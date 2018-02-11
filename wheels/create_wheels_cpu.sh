#!/bin/bash
set -e;

SDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)";
SOURCE_DIR=$(cd $SDIR/.. && pwd);

if [ "$DOCKER" != 1 ]; then
  rm -rf /tmp/nnutils/wheels/cpu;
  mkdir /tmp/nnutils/wheels/cpu;
  docker build -t nnutils .;
  docker run --rm --log-driver none \
	 -v /tmp:/host/tmp \
	 -v ${SOURCE_DIR}:/host/src \
	 nnutils:latest /create_wheels_cpu.sh;
  exit 0;
fi;

## THIS CODE IS EXECUTED WITHIN THE DOCKER CONTAINER

# Copy source in the host to a temporal location.
cp -r /host/src /tmp/src;

PYTHON_VERSIONS=(python2.7 python3.5 python3.6);
PYTHON_NUMBERS=(27 35 36);
for i in $(seq ${#PYTHON_VERSIONS[@]}); do
  export PYTHON=${PYTHON_VERSIONS[i - 1]}
  export PYV=${PYTHON_NUMBERS[i - 1]};
  source "py${PYV}-cpu/bin/activate";

  mkdir /tmp/src/build-py$PYV-cpu;
  cd /tmp/src/build-py$PYV-cpu;
  cmake -DWITH_CUDA=OFF -DCMAKE_BUILD_TYPE=RELEASE ..;
  make;
  cd pytorch;
  python setup.py bdist_wheel;
  cp dist/*.whl /host/tmp/nnutils/wheels/cpu;

  deactivate;
  cd /;
done;

echo "";
echo "";
echo "========================================================="
echo "== Python wheels located at /tmp/nnutils/wheels/cpu    =="
echo "========================================================="
