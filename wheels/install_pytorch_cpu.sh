#!/bin/bash
set -e;

PYTORCH_WHEELS=(
  http://download.pytorch.org/whl/cpu/torch-1.0.0-cp27-cp27mu-linux_x86_64.whl
  http://download.pytorch.org/whl/cpu/torch-1.0.0-cp35-cp35m-linux_x86_64.whl
  http://download.pytorch.org/whl/cpu/torch-1.0.0-cp36-cp36m-linux_x86_64.whl
  http://download.pytorch.org/whl/cpu/torch-1.0.0-cp37-cp37m-linux_x86_64.whl
);

i=0;
while [ $# -gt 0 ]; do
  export PYTHON="/opt/python/$1/bin/python";
  [ ! -f "$PYTHON" ] && echo "Python binary $PYTHON does not exist!" >&2 && exit 1;
  $PYTHON -m pip install numpy "${PYTORCH_WHEELS[i]}";
  shift;
  ((++i));
done;
