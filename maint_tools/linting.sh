#!/bin/bash

# author: Markus Löning
# code quality check using flake8

set -e -x
set -o pipefail

if ! flake8 --verbose --filename=*.py sktime/; then
  echo 'Linting failed.'
#  exit 1
fi
