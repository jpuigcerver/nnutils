#!/bin/bash
set -e;

SDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)";
SOURCE_DIR=$(cd $SDIR/.. && pwd);

if [ "$DOCKER" != 1 ]; then
  cd $SDIR;
  rm -rf /tmp/nnutils/wheels/cpu;
  mkdir -p /tmp/nnutils/wheels/cpu;
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

PYTHON_VERSIONS=(python2.7 python3.5 python3.6);
PYTHON_NUMBERS=(27 35 36);
for i in $(seq ${#PYTHON_VERSIONS[@]}); do
  export PYTHON=${PYTHON_VERSIONS[i - 1]}
  export PYV=${PYTHON_NUMBERS[i - 1]};
  source "/py${PYV}-cpu/bin/activate";

  mkdir /tmp/src/build-py$PYV-cpu;
  cd /tmp/src/build-py$PYV-cpu;
  cmake -DWITH_CUDA=OFF -DCMAKE_BUILD_TYPE=RELEASE ..;
  make;
  cd pytorch;
  python setup.py bdist_wheel;
  cp dist/*.whl /host/tmp/nnutils/wheels/cpu;

  # Install wheel.
  pip install $(find dist/ -name "*.whl");

  # Move to the tmp directory to ensure that nothing gets imported from the
  # build directory.
  cd /tmp;

  # Test installed module.
  python -m unittest nnutils_pytorch.mask_image_from_size_test;
  python -m unittest nnutils_pytorch.adaptive_avgpool_2d_test;
  python -m unittest nnutils_pytorch.adaptive_maxpool_2d_test;

  deactivate;
  cd /;
done;

echo "";
echo "";
echo "========================================================="
echo "== Python wheels located at /tmp/nnutils/wheels/cpu    =="
echo "========================================================="
