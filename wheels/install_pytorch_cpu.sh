#!/bin/bash
set -e;

[ $# -ne 1 ] && { echo "Missing python version!" >&2 && exit 1; }

BASE_URL="http://download.pytorch.org/whl/cpu";
URL="${BASE_URL}/torch-1.2.0%2Bcpu-$1-manylinux1_x86_64.whl";
"$PYTHON" -m pip install "$URL";
