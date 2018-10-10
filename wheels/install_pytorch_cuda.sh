#!/bin/bash
set -e;

[[ -z "$CUDA_VERSION_S" ]] &&
echo "Missing environment variable CUDA_VERSION_S" >&2 && exit 1;

PYTORCH_WHL_PREFIX="http://download.pytorch.org/whl/${CUDA_VERSION_S}";
PYTORCH_WHEELS=(
  ${PYTORCH_WHL_PREFIX}/torch-0.4.1-cp27-cp27mu-linux_x86_64.whl
  ${PYTORCH_WHL_PREFIX}/torch-0.4.1-cp35-cp35m-linux_x86_64.whl
  ${PYTORCH_WHL_PREFIX}/torch-0.4.1-cp36-cp36m-linux_x86_64.whl
  ${PYTORCH_WHL_PREFIX}/torch-0.4.1.post2-cp37-cp37m-linux_x86_64.whl
);

i=0;
while [ $# -gt 0 ]; do
  export PYTHON="/opt/python/$1/bin/python";
  [ ! -f "$PYTHON" ] && echo "Python binary $PYTHON does not exist!" >&2 && exit 1;
  $PYTHON -m pip install numpy "${PYTORCH_WHEELS[i]}";
  shift;
  ((++i));
done;
