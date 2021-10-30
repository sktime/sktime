#!/bin/sh

# Helper script to download and install sktime from test PyPI to check wheel
# and upload prior to new release

set -e

# Version to test, passed as input argument to script
VERSION=$1

# Make temporary directory
echo "Making test directory ..."
mkdir "$HOME"/testdir
cd "$HOME"/testdir

# Create test environment
echo "Creating test environment ..."
source $(conda info --base)/etc/profile.d/conda.sh  # set up conda
conda create -n sktime_testenv python=3.7
conda activate sktime_testenv

# Install from test PyPI
echo "Installing sktime from Test PyPI ..."
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple sktime=="$VERSION"
echo "Successfully installed sktime from Test PyPI."

# Clean up test directory and environment
echo "Cleaning up ..."
conda deactivate
conda remove -n sktime_testenv --all -y
rm -r "$HOME"/testdir

echo "Done."
