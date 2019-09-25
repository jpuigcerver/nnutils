#!/bin/bash
set -e;

[ $# -ne 1 ] && { echo "Missing python version!" >&2 && exit 1; }

BASE_URL="http://download.pytorch.org/whl/${CUDA_VERSION_S}";
if [[ "${CUDA_VERSION_S}" = cu92 ]]; then
  URL="${BASE_URL}/torch-1.2.0%2B${CUDA_VERSION_S}-$1-manylinux1_x86_64.whl";
else
  URL="${BASE_URL}/torch-1.2.0-$1-manylinux1_x86_64.whl";
fi;
"$PYTHON" -m pip install "$URL";
