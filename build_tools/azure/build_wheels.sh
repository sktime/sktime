#!/bin/bash

# adapted from
# - https://github.com/scikit-hep/azure-wheel-helpers/blob/master/build-wheels.sh
# - https://github.com/pypa/python-manylinux-demo/blob/master/travis/build-wheels.sh

set -e -x
echo "$REQUIREMENTS"
echo "$EXCLUDE_PYTHON_VERSIONS"

# Collect the available Pythons versions
PYTHON_VERSIONS=(/opt/python/*/bin)
# echo "All Python versions: ${PYTHON_VERSIONS[@]}"

# Filter out Python versions
# echo "${EXCLUDE_PYTHON_VERSIONS[@]}"
# split string by comma into array
IFS=',' read -ra EXCLUDE_PYTHON_VERSIONS <<< "$EXCLUDE_PYTHON_VERSIONS"

for VERSION in "${EXCLUDE_PYTHON_VERSIONS[@]}"; do
  # remove dot and whitespace from version number
  VERSION="$(echo -e "${VERSION//./}" | tr -d '[:space:]')"

  # remove versions
  PYTHON_VERSIONS=(${PYTHON_VERSIONS[@]//*"$VERSION"*/})
done
echo "Included Python versions: " "${PYTHON_VERSIONS[@]}"

# Build wheels
cd /io/ # Change directory

for PYTHON in "${PYTHON_VERSIONS[@]}"; do
  # Install requirements
  "${PYTHON}/pip" install -r "$REQUIREMENTS"

  # Install pytorch and signatory for the signatures module.
  # Note: torch must be installed via conda and signatory must be installed
  # after pytorch.
  "${PYTHON}/pip" install torch==1.5.0
  "${PYTHON}/pip" install signatory==1.2.1.1.5.0 --no-cache-dir --force-reinstall

  # Build wheel
  "${PYTHON}/python" setup.py bdist_wheel
done

# Bundle external shared libraries into the wheels using the auditwheel library
for wheel in dist/sktime-*.whl; do
  auditwheel repair --plat "$PLATFORM" --wheel-dir dist/ "$wheel"
done

# Install built whee wheel and test
for PYTHON in "${PYTHON_VERSIONS[@]}"; do
  # Install from wheel
  "${PYTHON}/pip" install --pre --no-index --find-links dist/ sktime

  # Run tests
  "${PYTHON}/pytest" --showlocals --durations=20 --junitxml=junit/test-results.xml --cov=sktime --cov-report=xml --cov-report=html --pyargs sktime
done
