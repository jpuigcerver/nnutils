SET(PYTORCH_SETUP_PREFIX ""
  CACHE STRING "Prefix path passed to Python's setup.py.")

ADD_SUBDIRECTORY(src)

IF(PYTORCH_FOUND AND WITH_PYTORCH AND PYTHONINTERP_FOUND)
  SET(_EXTERNAL_LIBS)
  SET(_EXTERNAL_DIRS)

  IF(PYTORCH_CUDA AND NOT WITH_CUDA)
    MESSAGE(WARNING "Your PyTorch installation supports CUDA, but "
      "nnutils_pytorch won't. Are you sure about this?")
  ENDIF(PYTORCH_CUDA AND NOT WITH_CUDA)

  # Dependency with the wrapper code.
  LIST(APPEND _EXTERNAL_LIBS
    ${CMAKE_CURRENT_BINARY_DIR}/src/cpu/libpytorch_cpu_.a)
  IF(CUDA_FOUND AND WITH_CUDA)
    LIST(APPEND _EXTERNAL_LIBS
      ${CMAKE_CURRENT_BINARY_DIR}/src/gpu/libpytorch_gpu_.a)
  ENDIF(CUDA_FOUND AND WITH_CUDA)

  # Dependency with other libraries needed by the cpu and gpu implementations.
  GET_PROPERTY(_AUX_LIBS TARGET pytorch_cpu_
    PROPERTY LINK_LIBRARIES)
  GET_PROPERTY(_AUX_DIRS TARGET pytorch_cpu_
    PROPERTY INCLUDE_DIRECTORIES)
  LIST(APPEND _EXTERNAL_LIBS "${_AUX_LIBS}")
  LIST(APPEND _EXTERNAL_DIRS "${_AUX_DIRS}")
  IF(CUDA_FOUND AND WITH_CUDA)
    GET_PROPERTY(_AUX_LIBS TARGET pytorch_gpu_
      PROPERTY LINK_LIBRARIES)
    GET_PROPERTY(_AUX_DIRS TARGET pytorch_gpu_
      PROPERTY INCLUDE_DIRECTORIES)
    LIST(APPEND _EXTERNAL_LIBS "${_AUX_LIBS}")
    LIST(APPEND _EXTERNAL_DIRS "${_AUX_DIRS}")
  ENDIF(CUDA_FOUND AND WITH_CUDA)

  # A single target that compiles both pytorch_cpu_ and pytorch_gpu_
  ADD_CUSTOM_TARGET(pytorch_ ALL DEPENDS pytorch_cpu_)
  IF(CUDA_FOUND AND WITH_CUDA)
    ADD_DEPENDENCIES(pytorch_ pytorch_gpu_)
  ENDIF(CUDA_FOUND AND WITH_CUDA)

  # Prepare setup.py
  SET(SETUP_PY_IN "${CMAKE_CURRENT_SOURCE_DIR}/setup.py.in")
  SET(SETUP_PY "${CMAKE_CURRENT_BINARY_DIR}/setup.py")
  SET(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/build/timestamp")

  CONFIGURE_FILE("${SETUP_PY_IN}"  "${SETUP_PY}")
  ADD_CUSTOM_COMMAND(OUTPUT "${OUTPUT}"
    COMMAND ${PYTHON_EXECUTABLE} "${SETUP_PY}" build
    COMMAND ${CMAKE_COMMAND} -E touch "${OUTPUT}"
    DEPENDS nnutils_pytorch/__init__.py
    DEPENDS nnutils_pytorch/nnutils.py
    DEPENDS pytorch_
    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")
  # Custom target to run setup.py build during make
  ADD_CUSTOM_TARGET(nnutils_pytorch ALL DEPENDS "${OUTPUT}")

  # This command will "install" nnutils_pytorch after it's built into a
  # temporal directory, in order to be able to run the Python tests without
  # actually installing the library.
  ADD_CUSTOM_COMMAND(TARGET nnutils_pytorch POST_BUILD
    COMMAND ${PYTHON_EXECUTABLE} "${SETUP_PY}"
    install --prefix=${CMAKE_CURRENT_BINARY_DIR}/inst
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    VERBATIM)

  SET(_PYTHONPATH
    "${CMAKE_CURRENT_BINARY_DIR}/inst/lib/python${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR}/site-packages:$ENV{PYTHONPATH}")

  ADD_TEST(NAME pytorch_mask_image_from_size_test
    COMMAND ${PYTHON_EXECUTABLE}
    ${CMAKE_CURRENT_SOURCE_DIR}/test/mask_image_from_size_test.py)
  ADD_TEST(NAME pytorch_adaptive_avgpool_2d_test
    COMMAND ${PYTHON_EXECUTABLE}
    ${CMAKE_CURRENT_SOURCE_DIR}/test/adaptive_avgpool_2d_test.py)
  ADD_TEST(NAME pytorch_adaptive_maxpool_2d_test
    COMMAND ${PYTHON_EXECUTABLE}
    ${CMAKE_CURRENT_SOURCE_DIR}/test/adaptive_maxpool_2d_test.py)
  SET_TESTS_PROPERTIES(
    pytorch_mask_image_from_size_test
    pytorch_adaptive_avgpool_2d_test
    pytorch_adaptive_maxpool_2d_test
    PROPERTIES
    ENVIRONMENT PYTHONPATH=${_PYTHONPATH})
  # If no PYTORCH_SETUP_PREFIX variable is given by the user, we will use
  # CMAKE_INSTALL_PREFIX (if given).
  IF(NOT PYTORCH_SETUP_PREFIX)
    SET(PYTORCH_SETUP_PREFIX "${CMAKE_INSTALL_PREFIX}")
  ENDIF()

  # make install will just execute python setup.py.
  # The install prefix will be PYTORCH_SETUP_PREFIX, if given. Or the system's
  # default if not given.
  IF(PYTORCH_SETUP_PREFIX)
    INSTALL(
      CODE "EXECUTE_PROCESS(COMMAND ${PYTHON_EXECUTABLE} ${SETUP_PY} install --prefix=${PYTORCH_SETUP_PREFIX} WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})")
  ELSE()
    INSTALL(
      CODE "EXECUTE_PROCESS(COMMAND ${PYTHON_EXECUTABLE} ${SETUP_PY} install WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})")
  ENDIF(PYTORCH_SETUP_PREFIX)

  SET_PROPERTY(DIRECTORY APPEND PROPERTY ADDITIONAL_MAKE_CLEAN_FILES
    "${CMAKE_CURRENT_BINARY_DIR}/inst" "${CMAKE_CURRENT_BINARY_DIR}/build")
ENDIF(PYTORCH_FOUND AND WITH_PYTORCH AND PYTHONINTERP_FOUND)
