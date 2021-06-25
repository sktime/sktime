#!/bin/bash

# Script to run all example notebooks.
set -e -x
set -o pipefail
CMD="jupyter nbconvert --to notebook --inplace --execute --ExecutePreprocessor.timeout=600"

for notebook in examples/*method.ipynb; do
  echo "Running: $notebook"
  $CMD "$notebook"
done
