#!/bin/bash

# adapted from https://github.com/scikit-learn/scikit-learn/blob/master/build_tools/travis/install.sh

# This script is meant to be called by the "install" step defined in
# .travis.yml. See https://docs.travis-ci.com/ for more details.
# The behavior of the script is controlled by environment variabled defined
# in the .travis.yml in the top level folder of the project.

# License: 3-clause BSD

# Travis clone sktime repository in to a local repository.
# We use a cached directory with three sktime repositories (one for each
# matrix entry) from which we pull from local Travis repository. This allows
# us to keep build artefact for gcc + cython, and gain time

set -e

# Fail fast
build_tools/travis/travis_fastfail.sh

echo 'List files from cached directories'
echo 'pip:'
ls "$HOME"/.cache/pip

if [ "$TRAVIS_OS_NAME" = "linux" ]
then
	export CC=/usr/lib/ccache/gcc
	export CXX=/usr/lib/ccache/g++
	# Useful for debugging how ccache is used
	# export CCACHE_LOGFILE=/tmp/ccache.log
	# ~60M is used by .ccache when compiling from scratch at the time of writing
	ccache --max-size 100M --show-stats
elif [ "$TRAVIS_OS_NAME" = "osx" ]
then
    # enable OpenMP support for Apple-clang
    export CC=/usr/bin/clang
    export CXX=/usr/bin/clang++
    export CPPFLAGS="$CPPFLAGS -Xpreprocessor -fopenmp"
    export CFLAGS="$CFLAGS -I/usr/local/opt/libomp/include"
    export CXXFLAGS="$CXXFLAGS -I/usr/local/opt/libomp/include"
    export LDFLAGS="$LDFLAGS -L/usr/local/opt/libomp/lib -lomp"
    export DYLD_LIBRARY_PATH=/usr/local/opt/libomp/lib
		export PATH="/usr/local/opt/ccache/libexec:$PATH"
fi

make_conda() {
    # Deactivate the travis-provided virtual environment and setup a
    # conda-based environment instead
    # If Travvis has language=generic (e.g. for macOS), deactivate does not exist. `|| :` will pass.
    deactivate || :

    # Install miniconda
    wget https://repo.continuum.io/miniconda/"$MINICONDA_VERSION" -O miniconda.sh
    MINICONDA=$HOME/miniconda
    chmod +x miniconda.sh && ./miniconda.sh -b -p "$MINICONDA"
    export PATH=$MINICONDA/bin:$PATH
    conda config --set always_yes true
    conda update --quiet conda

    # Set up test environment
    conda create --name testenv python="$PYTHON_VERSION"

    # Activate environment
    source activate testenv

    # Install requirements from inside conda environment
    pip install -r "$REQUIREMENTS"

    # Install signatory after the requirements due to limitations with
    # signatory needing to be installed after pytorch
    pip install signatory==1.2.0.

    # List installed environment
    python --version
    conda list -n testenv
}

# requirements file
make_conda "$REQUIREMENTS"

if [ "$COVERAGE" == "true" ]
then
    pip install coverage codecov pytest-cov
fi

# Build sktime in the install.sh script to collapse the verbose
# build output in the travis output when it succeeds.
# conda list

# python setup.py develop  # invokes build_ext -i to compile files
python setup.py bdist_wheel
ls dist  # list build artifacts

# Install from built wheels
pip install --pre --no-index --no-deps --find-links dist/ sktime

# Useful for debugging how ccache is used
if [ "$TRAVIS_OS_NAME" = "linux" ]
then
	ccache --show-stats
fi
# cat $CCACHE_LOGFILE

# fast fail
build_tools/travis/travis_fastfail.sh

set +e
