#!/bin/bash

# adapted from
# - https://github.com/scikit-hep/azure-wheel-helpers/blob/master/build-wheels.sh
# - https://github.com/pypa/python-manylinux-demo/blob/master/travis/build-wheels.sh

set -e -x
echo "$REQUIREMENTS"
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
  # Install requirements
  "${PYTHON}/pip" install -r "$REQUIREMENTS"

  # Build wheel
  "${PYTHON}/python" setup.py bdist_wheel
done

# Bundle external shared libraries into the wheels using the auditwheel library
for wheel in dist/sktime-*.whl; do
  auditwheel repair --plat "$PLATFORM" --wheel-dir dist/ "$wheel"
done

# Install built wheel and test
for PYTHON in "${PYTHON_VERSIONS[@]}"; do
  # Install from wheel
  "${PYTHON}/pip" install --pre --no-index --no-deps --find-links dist/ sktime

  # Run tests
  "${PYTHON}/pytest" --showlocals --durations=20 --junitxml=junit/test-results.xml --cov=sktime --cov-report=xml --cov-report=html --pyargs sktime
done
