#!/bin/bash
set -e;

SDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)";
SOURCE_DIR=$(cd $SDIR/.. && pwd);

###########################################
## THIS CODE IS EXECUTED WITHIN THE HOST ##
###########################################
if [ ! -f /.dockerenv ]; then
  docker run --rm --log-driver none \
	 -v /tmp:/host/tmp \
	 -v ${SOURCE_DIR}:/host/src \
	 quay.io/pypa/manylinux2014_x86_64 \
	 /host/src/wheels/create_wheels_cpu.sh;
  exit 0;
fi;

#######################################################
## THIS CODE IS EXECUTED WITHIN THE DOCKER CONTAINER ##
#######################################################
set -ex;

source /opt/rh/devtoolset-9/enable;

# Copy host source directory, to avoid changes in the host.
cp -r /host/src /tmp/src;

ODIR="/host/tmp/nnutils_pytorch/whl/cpu";
mkdir -p "$ODIR";
wheels=();
for py in cp36-cp36m cp37-cp37m cp38-cp38; do
  export PYTHON=/opt/python/$py/bin/python;
  cd /tmp/src/pytorch;
  # Remove previous builds.
  rm -rf build dist;

  "$PYTHON" -m pip install --default-timeout=1000 -U pip;
  "$PYTHON" -m pip install --default-timeout=1000 -U wheel setuptools;

  echo "=== Installing requirements for $py with CPU-only ===";
  "$PYTHON" -m pip install --default-timeout=1000 https://download.pytorch.org/whl/cpu/torch-1.6.0%2Bcpu-${py}-linux_x86_64.whl
  "$PYTHON" -m pip install \
	    -r <(sed -r 's|^torch((>=\|>).*)?$||g;/^$/d' requirements.txt);

  echo "=== Building wheel for $py with CPU-only ===";
  $PYTHON setup.py clean;
  $PYTHON setup.py bdist_wheel;

  # No need to fix wheel for CPU

  # Move dev libraries to a different location to make sure that tests do
  # not use them.
  mv /opt/rh /opt/rh_tmp;

  echo "=== Installing wheel for $py with CPU-only ===";
  cd /tmp;
  $PYTHON -m pip uninstall -y nnutils_pytorch;
  $PYTHON -m pip install nnutils_pytorch --no-index -f /tmp/src/pytorch/dist \
	  --no-dependencies -v;

  echo "=== Testing wheel for $py with CPU-only ===";
  $PYTHON -m unittest nnutils_pytorch.mask_image_from_size_test;
  $PYTHON -m unittest nnutils_pytorch.adaptive_avgpool_2d_test;
  $PYTHON -m unittest nnutils_pytorch.adaptive_maxpool_2d_test;


  # Move dev libraries back to their original location after tests.
  mv /opt/rh_tmp /opt/rh;

  echo "=== Copying wheel for $py with CPU-only to the host ===";
  readarray -t whl < <(find /tmp/src/pytorch/dist -name "*.whl");
  whl_name="$(basename "$whl")";
  whl_name="${whl_name/-linux/-manylinux1}";
  mv "$whl" "${ODIR}/${whl_name}";
  wheels+=("${ODIR}/${whl_name}");
done;

echo "================================================================";
printf "=== %-56s ===\n" "Copied ${#wheels[@]} wheels to ${ODIR:5}";
echo "================================================================";
