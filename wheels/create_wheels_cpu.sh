#!/bin/bash
set -e;

SDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)";
SOURCE_DIR=$(cd $SDIR/.. && pwd);

if [ "$DOCKER" != 1 ]; then
  cd $SDIR;
  docker build --build-arg BASE_IMAGE=ubuntu:16.04 \
	 -t nnutils:cpu-base -f Dockerfile .;
  docker build -t nnutils:cpu -f Dockerfile-cpu .;
  docker run --rm --log-driver none \
	 -v /tmp:/host/tmp \
	 -v ${SOURCE_DIR}:/host/src \
	 nnutils:cpu /create_wheels_cpu.sh;
  exit 0;
fi;

## THIS CODE IS EXECUTED WITHIN THE DOCKER CONTAINER

# Copy source in the host to a temporal location.
cp -r /host/src /tmp/src;
cd /tmp/src;
git status;

PYTHON_VERSIONS=(
  python2.7
  python3.5
  python3.6
  python3.7
);
PYTHON_NUMBERS=(
  27
  35
  36
  37
);
PYTHON_SUFFIX=(
  cp27-cp27mu-linux_x86_64.whl
  cp35-cp35m-linux_x86_64.whl
  cp36-cp36m-linux_x86_64.whl
  cp37-cp37m-linux_x86_64.whl
);
for i in $(seq ${#PYTHON_VERSIONS[@]}); do
  export PYTHON=${PYTHON_VERSIONS[i - 1]}
  export PYV=${PYTHON_NUMBERS[i - 1]};
  ODIR=/host/tmp/nnutils/whl/cpu;
  mkdir -p "$ODIR";
  if [ $(find "$ODIR" -name "*-${PYTHON_SUFFIX[i-1]}" | wc -l) -eq 0 ]; then
    source "/py${PYV}-cpu/bin/activate";
    cd /tmp/src/pytorch;
    echo "=== Building for $PYTHON-cpu ==="
    python setup.py bdist_wheel;
    whl=$(find dist/ -name "*${PYTHON_SUFFIX[i-1]}");
    # Install nnutils wheel.
    pip install "$whl";
    # Move to the tmp directory to ensure that nothing gets imported from the
    # build directory.
    cp "$whl" "$ODIR";
    cd /tmp;
    # Test installed module.
    python -m unittest nnutils_pytorch.mask_image_from_size_test;
    python -m unittest nnutils_pytorch.adaptive_avgpool_2d_test;
    python -m unittest nnutils_pytorch.adaptive_maxpool_2d_test;
    deactivate;
  fi;
done;

echo "";
echo "";
echo "========================================================="
echo "== Python wheels located at /tmp/nnutils/whl/cpu       =="
echo "========================================================="
