#!/bin/bash
set -e;

if [[ "$DOCKER" != 1 ||
      -z "$CUDA_VERSION_SHORT" ]]; then
  echo "This script should be executed from a CUDA Docker container." >&2 && \
  exit 1;
fi;

PYTHON_VERSIONS=(python2.7 python3.5 python3.6 python3.7);
PYTHON_NUMBERS=(27 35 36 37);
PYTORCH_WHEELS=(
  http://download.pytorch.org/whl/cu${CUDA_VERSION_SHORT}/torch-0.4.1-cp27-cp27mu-linux_x86_64.whl
  http://download.pytorch.org/whl/cu${CUDA_VERSION_SHORT}/torch-0.4.1-cp35-cp35m-linux_x86_64.whl
  http://download.pytorch.org/whl/cu${CUDA_VERSION_SHORT}/torch-0.4.1-cp36-cp36m-linux_x86_64.whl
  http://download.pytorch.org/whl/cu${CUDA_VERSION_SHORT}/torch-0.4.1.post2-cp37-cp37m-linux_x86_64.whl
);

for i in $(seq ${#PYTHON_VERSIONS[@]}); do
  export PYTHON=${PYTHON_VERSIONS[i - 1]}
  export PYV=${PYTHON_NUMBERS[i - 1]};
  virtualenv --python=$PYTHON py${PYV}-cuda;
  source "py${PYV}-cuda/bin/activate";
  pip --version;
  pip install "${PYTORCH_WHEELS[i - 1]}";
  pip install torchvision;
  deactivate;
done;
