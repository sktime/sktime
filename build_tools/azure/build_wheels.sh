#!/bin/bash

# adapted from
# - https://github.com/scikit-hep/azure-wheel-helpers/blob/master/build-wheels.sh
# - https://github.com/pypa/python-manylinux-demo/blob/master/travis/build-wheels.sh

set -e -x
echo "$REQUIREMENTS"
echo "$EXCLUDE_PYTHON_VERSIONS"

# Collect the available Pythons versions
pys=(/opt/python/*/bin)
echo "All Python versions: ${pys[@]}"

# Filter out Python versions
#echo "${EXCLUDE_PYTHON_VERSIONS[@]}"
for VERSION in "${EXCLUDE_PYTHON_VERSIONS[@]}"; do
  VERSION="${VERSION//.}"  # strip dot from version number
  pys=(${pys[@]//*"$VERSION"*/})  # remove versions
done
echo "Included Python versions: ${pys[@]}"

#pys=(${pys[@]//*27*/})
#pys=(${pys[@]//*34*/})
#pys=(${pys[@]//*35*/})

# Build wheels
export CC=/usr/lib/ccache/gcc
export CXX=/usr/lib/ccache/g++

cd /io/

for PYTHON in "${pys[@]}"; do
    "${PYTHON}/pip" install -r "$REQUIREMENTS"
    "${PYTHON}/python" setup.py bdist_wheel
done

# Bundle external shared libraries into the wheels using the auditwheel library
for wheel in dist/sktime-*.whl; do
    auditwheel repair --plat "$PLATFORM" --wheel-dir dist/ "$wheel"
done

# Install built whee wheel and test
for PYTHON in "${pys[@]}"; do
    "${PYTHON}/pip" install --pre --no-index --find-links dist/ sktime
    "${PYTHON}/pytest" --showlocals --durations=20 --pyargs sktime
done
