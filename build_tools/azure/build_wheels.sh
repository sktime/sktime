#!/bin/bash

# adapted from
# - https://github.com/scikit-hep/azure-wheel-helpers/blob/master/build-wheels.sh
# - https://github.com/pypa/python-manylinux-demo/blob/master/travis/build-wheels.sh

set -e -x
echo "$requirements"

# Collect the pythons
pys=(/opt/python/*/bin)

# Print list of Python's available
echo "All Pythons: ${pys[@]}"

ls -lh /io/build_tools

# Filter out Python versions
pys=("${pys[@]//*27*/}")
pys=("${pys[@]//*34*/}")
pys=("${pys[@]//*35*/}")

# Print list of Python's being used
echo "Using Python versions: ${pys[@]}"

# Compile wheels
for PYBIN in "${pys[@]}"; do
    "${PYBIN}/pip" install -r /io/"$requirements"
    "${PYBIN}/pip" wheel -v /io/ -w wheelhouse/
done

# Bundle external shared libraries into the wheels using the auditwheel library
for whl in wheelhouse/"$PACKAGE_NAME"-*.whl; do
    auditwheel repair --plat $PLAT "$whl" -w /io/wheelhouse/
done

# Install packages and test
for PYBIN in "${pys[@]}"; do
    "${PYBIN}/pip" install -r /io/$test_requirements_file
    "${PYBIN}/pip" install "$PACKAGE_NAME" --no-index -f /io/wheelhouse
    if [ -d "/io/tests" ]; then
        "${PYBIN}/pytest" /io/tests
    else
        "${PYBIN}/pytest" --pyargs "$PACKAGE_NAME"
    fi
done