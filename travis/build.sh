#!/bin/bash
set -e;

mkdir build;
cd build;
cmake -DGTEST_ROOT="$GTEST_ROOT" -DGMOCK_ROOT="$GTEST_ROOT" \
      -DWITH_CUDA=OFF -DWITH_PYTORCH=ON -DCMAKE_BUILD_TYPE=DEBUG ..;
make;
