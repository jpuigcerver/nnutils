#!/bin/bash
set -e;

mkdir build;
cd build;
cmake -DWITH_CUDA=OFF -DWITH_PYTORCH=ON -DCMAKE_BUILD_TYPE=DEBUG ..;
VERBOSE=1 make;
