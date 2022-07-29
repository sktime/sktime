#!/bin/bash

# adapted from
# - https://github.com/scikit-hep/azure-wheel-helpers/blob/master/build-wheels.sh
# - https://github.com/pypa/python-manylinux-demo/blob/master/travis/build-wheels.sh

set -e -x
echo "$INCLUDED_VERSIONS"

# Split input string of included Python version by comma into array
IFS=',' read -ra INCLUDED_VERSIONS <<< "$INCLUDED_VERSIONS"

# Initialize empty array
PYTHON_VERSIONS=()

# Include Python versions
for VERSION in "${INCLUDED_VERSIONS[@]}"; do
  # Trim white space
  VERSION="$(echo -e "${VERSION}" | tr -d '[:space:]')"

  # Append version
  PYTHON_VERSIONS+=("/opt/python/$VERSION/bin")
done

echo "Included Python versions: " "${PYTHON_VERSIONS[@]}"

# Build wheels
cd /io/ # Change directory

for PYTHON in "${PYTHON_VERSIONS[@]}"; do
  "${PYTHON}/python" -m pip install ".[all_extras,dev]"
done

# Install built wheel and test
for PYTHON in "${PYTHON_VERSIONS[@]}"; do
  # Run tests
  "${PYTHON}/pytest" --junitxml=junit/test-results.xml
done
