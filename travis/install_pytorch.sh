#!/bin/bash
set -e;

base_url=http://download.pytorch.org/whl;

if [ "$TRAVIS_OS_NAME" = linux ]; then
  if [ "$TRAVIS_PYTHON_VERSION" = "2.7" ]; then
    url="${base_url}/cpu/torch-1.0.0-cp27-cp27mu-linux_x86_64.whl";
  elif [ "$TRAVIS_PYTHON_VERSION" = "3.5" ]; then
    url="${base_url}/cpu/torch-1.0.0-cp35-cp35m-linux_x86_64.whl";
  elif [ "$TRAVIS_PYTHON_VERSION" = "3.6" ]; then
    url="${base_url}/cpu/torch-1.0.0-cp36-cp36m-linux_x86_64.whl";
  elif [ "$TRAVIS_PYTHON_VERSION" = "3.7" ]; then
    url="${base_url}/cpu/torch-1.0.0-cp37-cp37m-linux_x86_64.whl";
  fi;
elif [ "$TRAVIS_OS_NAME" = osx ]; then
  if [ "$TRAVIS_PYTHON_VERSION" = "2.7" ]; then
    url="$base_url/torch-1.0.0-cp27-none-macosx_10_6_x86_64.whl";
  elif [ "$TRAVIS_PYTHON_VERSION" = "3.5" ]; then
    url="$base_url/torch-1.0.0-cp35-cp35m-macosx_10_6_x86_64.whl";
  elif [ "$TRAVIS_PYTHON_VERSION" = "3.6" ]; then
    url="$base_url/torch-1.0.0-cp36-cp36m-macosx_10_7_x86_64.whl";
  fi;
fi;

python -m pip install --progress-bar off "$url" || exit 1;
