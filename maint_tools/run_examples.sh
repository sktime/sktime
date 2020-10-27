#!/bin/bash

# Script to run all example notebooks.

CMD="jupyter nbconvert --to notebook --inplace --execute --ExecutePreprocessor.timeout=600"

for notebook in examples/*.ipynb; do
  echo "Running: $notebook"
  $CMD "$notebook" || break
done
