#!/bin/bash

# Script to search for missing init FILES.
set -euxo pipefail

FILES=$( find ./sktime -type d '!' -exec test -e "{}/__init__.py" ";" -not -path "**/__pycache__" -not -path "**/datasets/data*" -not -path "**/contrib/*" -print )

if [[ -n "$FILES" ]]
then
    echo "Missing __init__.py files detected in the following modules:"
    echo "$FILES"
    exit 1
fi
