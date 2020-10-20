#!/bin/sh

set -e

VERSION=0.4.3

# Make temporary directory
echo "Making test directory ..."
mkdir "$HOME"/testdir
cd "$HOME"/testdir

# Create test environment
echo "Creating test environemnt ..."
source $(conda info --base)/etc/profile.d/conda.sh
conda create -n sktime_testenv python=3.7
conda activate testenv

# Install from test PyPI
echo "Install sktime from Test PyPI ..."
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple sktime=="$VERSION"

# Remove test directory and environment
echo "Clean up ..."
conda remove -n sktime_testenv --all -y
cd -
rm -r "$HOME"/testdir

echo "Done."
