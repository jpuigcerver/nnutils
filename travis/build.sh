#!/bin/bash
set -e;

mkdir build;
cd build;
cmake -DWITH_CUDA=OFF -DWITH_PYTORCH=ON -DCMAKE_BUILD_TYPE=DEBUG \
      -DPYTORCH_SETUP_PREFIX=$HOME/.local ..;
make VERBOSE=1;
