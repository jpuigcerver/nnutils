import copy
import glob
import os
import re
import subprocess
import sys
import warnings

import setuptools
import torch
from setuptools.command.build_ext import build_ext


def _find_cuda_home():
    """Finds the CUDA install path."""
    # Guess #1
    cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    if cuda_home is None:
        # Guess #2
        if sys.platform == "win32":
            cuda_home = glob.glob(
                "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v*.*"
            )
        else:
            cuda_home = "/usr/local/cuda"
        if not os.path.exists(cuda_home):
            # Guess #3
            try:
                which = "where" if sys.platform == "win32" else "which"
                nvcc = subprocess.check_output([which, "nvcc"]).decode().rstrip("\r\n")
                cuda_home = os.path.dirname(os.path.dirname(nvcc))
            except Exception:
                cuda_home = None
    if cuda_home and not torch.cuda.is_available():
        print("No CUDA runtime is found, using CUDA_HOME='{}'".format(cuda_home))
    return cuda_home


MINIMUM_GCC_VERSION = (4, 9)
MINIMUM_MSVC_VERSION = (19, 0, 24215)
ABI_INCOMPATIBILITY_WARNING = """
                               !! WARNING !!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
Your compiler ({}) may be ABI-incompatible with PyTorch!
Please use a compiler that is ABI-compatible with GCC 4.9 and above.
See https://gcc.gnu.org/onlinedocs/libstdc++/manual/abi.html.
See https://gist.github.com/goldsborough/d466f43e8ffc948ff92de7486c5216d6
for instructions on how to install GCC 4.9 or higher.
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                              !! WARNING !!
"""
CUDA_HOME = _find_cuda_home()
# PyTorch releases have the version pattern major.minor.patch, whereas when
# PyTorch is built from source, we append the git commit hash, which gives
# it the below pattern.
BUILT_FROM_SOURCE_VERSION_PATTERN = re.compile(r"\d+\.\d+\.\d+\w+\+\w+")


def check_compiler_abi_compatibility(compiler):
    """
    Verifies that the given compiler is ABI-compatible with PyTorch.
    Arguments:
        compiler (str): The compiler executable name to check (e.g. ``g++``).
            Must be executable in a shell process.
    Returns:
        False if the compiler is (likely) ABI-incompatible with PyTorch,
        else True.
    """
    if BUILT_FROM_SOURCE_VERSION_PATTERN.match(torch.version.__version__):
        return True
    try:
        check_cmd = "{}" if sys.platform == "win32" else "{} --version"
        info = subprocess.check_output(
            check_cmd.format(compiler).split(), stderr=subprocess.STDOUT
        )
    except Exception:
        _, error, _ = sys.exc_info()
        warnings.warn("Error checking compiler version: {}".format(error))
    else:
        info = info.decode().lower()
        if "gcc" in info or "g++" in info:
            # Sometimes the version is given as "major.x" instead of semver.
            version = re.search(r"(\d+)\.(\d+|x)", info)
            if version is not None:
                major, minor = version.groups()
                minor = 0 if minor == "x" else int(minor)
                if (int(major), minor) >= MINIMUM_GCC_VERSION:
                    return True
                else:
                    # Append the detected version for the warning.
                    compiler = "{} {}".format(compiler, version.group(0))
        elif "Microsoft" in info:
            info = info.decode().lower()
            version = re.search(r"(\d+)\.(\d+)\.(\d+)", info)
            if version is not None:
                major, minor, revision = version.groups()
                if (int(major), int(minor), int(revision)) >= MINIMUM_MSVC_VERSION:
                    return True
                else:
                    # Append the detected version for the warning.
                    compiler = "{} {}".format(compiler, version.group(0))

    warnings.warn(ABI_INCOMPATIBILITY_WARNING.format(compiler))
    return False


class BuildExtension(build_ext):
    """
    A custom :mod:`setuptools` build extension .
    This :class:`setuptools.build_ext` subclass takes care of passing the
    minimum required compiler flags (e.g. ``-std=c++11``) as well as mixed
    C++/CUDA compilation (and support for CUDA files in general).
    When using :class:`BuildExtension`, it is allowed to supply a dictionary
    for ``extra_compile_args`` (rather than the usual list) that maps from
    languages (``cxx`` or ``cuda``) to a list of additional compiler flags to
    supply to the compiler. This makes it possible to supply different flags to
    the C++ and CUDA compiler during mixed compilation.
    """

    def build_extensions(self):
        self._check_abi()
        for extension in self.extensions:
            self._define_torch_extension_name(extension)

        # Register .cu and .cuh as valid source extensions.
        self.compiler.src_extensions += [".cu", ".cuh"]
        # Save the original _compile method for later.
        if self.compiler.compiler_type == "msvc":
            self.compiler._cpp_extensions += [".cu", ".cuh"]
            original_compile = self.compiler.compile
            original_spawn = self.compiler.spawn
        else:
            original_compile = self.compiler._compile

        def unix_wrap_compile(obj, src, ext, cc_args, extra_postargs, pp_opts):
            # Copy before we make any modifications.
            cflags = copy.deepcopy(extra_postargs)
            try:
                original_compiler = self.compiler.compiler_so
                if _is_cuda_file(src):
                    nvcc = _join_cuda_home("bin", "nvcc")
                    self.compiler.set_executable("compiler_so", nvcc)
                    if isinstance(cflags, dict):
                        cflags = cflags["nvcc"]
                    cflags += ["--compiler-options", "'-fPIC'"]
                elif isinstance(cflags, dict):
                    cflags = cflags["cxx"]
                # NVCC does not allow multiple -std to be passed, so we avoid
                # overriding the option if the user explicitly passed it.
                if not any(flag.startswith("-std=") for flag in cflags):
                    cflags.append("-std=c++11")

                original_compile(obj, src, ext, cc_args, cflags, pp_opts)
            finally:
                # Put the original compiler back in place.
                self.compiler.set_executable("compiler_so", original_compiler)

        def win_wrap_compile(
                sources,
                output_dir=None,
                macros=None,
                include_dirs=None,
                debug=0,
                extra_preargs=None,
                extra_postargs=None,
                depends=None,
        ):

            self.cflags = copy.deepcopy(extra_postargs)
            extra_postargs = None

            def spawn(cmd):
                orig_cmd = cmd
                # Using regex to match src, obj and include files

                src_regex = re.compile("/T(p|c)(.*)")
                src_list = [
                    m.group(2) for m in (src_regex.match(elem) for elem in cmd) if m
                ]

                obj_regex = re.compile("/Fo(.*)")
                obj_list = [
                    m.group(1) for m in (obj_regex.match(elem) for elem in cmd) if m
                ]

                include_regex = re.compile(r"((\-|\/)I.*)")
                include_list = [
                    m.group(1) for m in (include_regex.match(elem) for elem in cmd) if m
                ]

                if len(src_list) >= 1 and len(obj_list) >= 1:
                    src = src_list[0]
                    obj = obj_list[0]
                    if _is_cuda_file(src):
                        nvcc = _join_cuda_home("bin", "nvcc")
                        if isinstance(self.cflags, dict):
                            cflags = self.cflags["nvcc"]
                        elif isinstance(self.cflags, list):
                            cflags = self.cflags
                        else:
                            cflags = []
                        cmd = (
                                [
                                    nvcc,
                                    "-c",
                                    src,
                                    "-o",
                                    obj,
                                    "-Xcompiler",
                                    "/wd4819",
                                    "-Xcompiler",
                                    "/MD",
                                ]
                                + include_list
                                + cflags
                        )
                    elif isinstance(self.cflags, dict):
                        cflags = self.cflags["cxx"]
                        cmd += cflags
                    elif isinstance(self.cflags, list):
                        cflags = self.cflags
                        cmd += cflags

                return original_spawn(cmd)

            try:
                self.compiler.spawn = spawn
                return original_compile(
                    sources,
                    output_dir,
                    macros,
                    include_dirs,
                    debug,
                    extra_preargs,
                    extra_postargs,
                    depends,
                )
            finally:
                self.compiler.spawn = original_spawn

        # Monkey-patch the _compile method.
        if self.compiler.compiler_type == "msvc":
            self.compiler.compile = win_wrap_compile
        else:
            self.compiler._compile = unix_wrap_compile

        build_ext.build_extensions(self)

    def _check_abi(self):
        # On some platforms, like Windows, compiler_cxx is not available.
        if hasattr(self.compiler, "compiler_cxx"):
            compiler = self.compiler.compiler_cxx[0]
        elif sys.platform == "win32":
            compiler = os.environ.get("CXX", "cl")
        else:
            compiler = os.environ.get("CXX", "c++")
        check_compiler_abi_compatibility(compiler)

    def _define_torch_extension_name(self, extension):
        # pybind11 doesn't support dots in the names
        # so in order to support extensions in the packages
        # like torch._C, we take the last part of the string
        # as the library name
        names = extension.name.split(".")
        name = names[-1]
        define = "-DTORCH_EXTENSION_NAME={}".format(name)
        if isinstance(extension.extra_compile_args, dict):
            for args in extension.extra_compile_args.values():
                args.append(define)
        else:
            extension.extra_compile_args.append(define)


def include_paths(cuda=False):
    """
    Get the include paths required to build a C++ or CUDA extension.
    Args:
        cuda: If `True`, includes CUDA-specific include paths.
    Returns:
        A list of include path strings.
    """
    torch_path = os.path.dirname(os.path.abspath(torch.__file__))
    lib_include = os.path.join(torch_path, "lib", "include")
    # Some internal (old) Torch headers don't properly prefix their includes,
    # so we need to pass -Itorch/lib/include/TH as well.
    paths = [
        lib_include,
        os.path.join(lib_include, "TH"),
        os.path.join(lib_include, "THC"),
    ]
    if cuda:
        paths.append(_join_cuda_home("include"))
    return paths


def library_paths(cuda=False):
    """
    Get the library paths required to build a C++ or CUDA extension.
    Args:
        cuda: If `True`, includes CUDA-specific library paths.
    Returns:
        A list of library path strings.
    """
    paths = []

    if sys.platform == "win32":
        torch_path = os.path.dirname(os.path.abspath(torch.__file__))
        lib_path = os.path.join(torch_path, "lib")

        paths.append(lib_path)

    if cuda:
        lib_dir = "lib/x64" if sys.platform == "win32" else "lib64"
        paths.append(_join_cuda_home(lib_dir))
    return paths


def _join_cuda_home(*paths):
    """
    Joins paths with CUDA_HOME, or raises an error if it CUDA_HOME is not set.
    This is basically a lazy way of raising an error for missing $CUDA_HOME
    only once we need to get any CUDA-specific path.
    """
    if CUDA_HOME is None:
        raise EnvironmentError(
            "CUDA_HOME environment variable is not set. "
            "Please set it to your CUDA install root."
        )
    return os.path.join(CUDA_HOME, *paths)


def _is_cuda_file(path):
    return os.path.splitext(path)[1] in [".cu", ".cuh"]
