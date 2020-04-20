#!/bin/bash

set -e -x
set -o pipefail

flake8 --verbose --filename=*.py sktime/
if [ $? -ne 0 ]; then
        exit 1
fi