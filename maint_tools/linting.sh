#!/bin/bash

# author: Markus Löning
# code quality check using flake8

set -e -x
set -o pipefail

if ! flake8 --verbose --filename=*.py sktime/; then
  echo 'Linting failed.'
<<<<<<< HEAD
#  exit 1
=======
  # CI will fail when linting fails
  exit 1
>>>>>>> 67c56be8b1e838f2628df829946f795b7dba9aed
fi
