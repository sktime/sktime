#!/bin/bash

# Script to search for missing init files.
set -euxo pipefail

if [[ -n `find ./sktime/ -type d '!' -exec test -e "{}/__init__.py" ";" -not -path "**/__pycache__" -not -path "**/datasets/data*" -not -path "**/contrib/*" -print` ]]
then
    echo "Missing init files detected."
    exit 1
fi
