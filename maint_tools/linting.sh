#!/bin/bash

set -e -x
set -o pipefail

if ! flake8 --verbose --filename=*.py sktime/; then
  echo 'Linting failed.'
  exit 1
fi
