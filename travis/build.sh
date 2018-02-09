#!/bin/bash
set -e;

mkdir build;
cd build;
cmake -DWITH_CUDA=OFF -DWITH_PYTORCH=ON -DCMAKE_BUILD_TYPE=DEBUG ..;
make;
