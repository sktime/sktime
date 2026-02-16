#!/bin/bash

# Script to run all blog post notebooks.
set -euxo pipefail

CMD="jupyter nbconvert --to notebook --inplace --execute --ExecutePreprocessor.timeout=600"

for notebook in examples/blog_posts/*.ipynb; do
  echo "Running: $notebook"
  $CMD "$notebook"
done
