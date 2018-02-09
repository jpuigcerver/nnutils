#!/bin/bash
set -e;

if [ "$TRAVIS_OS_NAME" = linux ]; then
  :
elif [ "$TRAVIS_OS_NAME" = osx ]; then
  :
fi;
