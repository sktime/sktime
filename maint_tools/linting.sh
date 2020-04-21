#!/bin/bash

set -e -x
set -o pipefail

pwd
ls -lh sktime/

if ! flake8 --verbose --filename=*.py sktime/; then
  echo 'Linting failed.'
  exit 1
fi
