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

# brew installs specified in .travis.yml
# if [ $TRAVIS_OS_NAME = "macos" ]
#then
#  # Install jq for travis_fastfail.sh script
#  brew install jq
#fi

# Fail fast
build_tools/travis/travis_fastfail.sh

echo 'List files from cached directories'
echo 'pip:'
ls $HOME/.cache/pip

if [ $TRAVIS_OS_NAME = "linux" ]
then
	export CC=/usr/lib/ccache/gcc
	export CXX=/usr/lib/ccache/g++
	# Useful for debugging how ccache is used
	# export CCACHE_LOGFILE=/tmp/ccache.log
	# ~60M is used by .ccache when compiling from scratch at the time of writing
	ccache --max-size 100M --show-stats
elif [ $TRAVIS_OS_NAME = "osx" ]
then
    # enable OpenMP support for Apple-clang
    export CC=/usr/bin/clang
    export CXX=/usr/bin/clang++
    export CPPFLAGS="$CPPFLAGS -Xpreprocessor -fopenmp"
    export CFLAGS="$CFLAGS -I/usr/local/opt/libomp/include"
    export CXXFLAGS="$CXXFLAGS -I/usr/local/opt/libomp/include"
    export LDFLAGS="$LDFLAGS -L/usr/local/opt/libomp/lib -lomp"
    export DYLD_LIBRARY_PATH=/usr/local/opt/libomp/lib

    # Install the OpenMP library
    # brew installs specified in .travis.yml
    #	brew install libomp

		# Install ccache manually for macOS environments
		# brew installs specified in .travis.yml
		# brew install ccache
		export PATH="/usr/local/opt/ccache/libexec:$PATH"
fi

make_conda() {
	TO_INSTALL="$@"
    # Deactivate the travis-provided virtual environment and setup a
    # conda-based environment instead
    # If Travvis has language=generic (e.g. for macOS), deactivate does not exist. `|| :` will pass.
    deactivate || :

    # Install miniconda
    if [ $TRAVIS_OS_NAME = "osx" ]
	  then
		fname=Miniconda3-latest-MacOSX-x86_64.sh
	  else
		fname=Miniconda3-latest-Linux-x86_64.sh
	  fi

    wget https://repo.continuum.io/miniconda/$fname \
        -O miniconda.sh
    MINICONDA_PATH=$HOME/miniconda
    chmod +x miniconda.sh && ./miniconda.sh -b -p $MINICONDA_PATH
    export PATH=$MINICONDA_PATH/bin:$PATH
    conda update --yes conda

    # Set up test environment
    conda create -n testenv -c conda-forge --yes $TO_INSTALL

    # Activate environment
    source activate testenv

    # Install packages not available via conda
    pip install scikit-posthocs==$SCIKIT_POSTHOCS_VERSION
    pip install joblib==$JOBLIB_VERSION
    pip install tsfresh
}

TO_INSTALL="python=$PYTHON_VERSION pip pytest \
            numpy=$NUMPY_VERSION scipy=$SCIPY_VERSION \
            cython=$CYTHON_VERSION scikit-learn=$SKLEARN_VERSION \
            pandas=$PANDAS_VERSION statsmodels=$STATSMODELS_VERSION \
            wheel \
						numba=$NUMBA_VERSION"
make_conda $TO_INSTALL

if [ "$COVERAGE" == "true" ]
then
    pip install coverage codecov pytest-cov
fi

# Build sktime in the install.sh script to collapse the verbose
# build output in the travis output when it succeeds.

# python setup.py develop  # invokes build_ext -i to compile files
python setup.py bdist_wheel
ls dist  # list build artifacts

# Install from built wheels
pip install --pre --no-index --find-links dist/ sktime

# Useful for debugging how ccache is used
if [ $TRAVIS_OS_NAME = "linux" ]
then
	ccache --show-stats
fi
# cat $CCACHE_LOGFILE

# fast fail
build_tools/travis/travis_fastfail.sh

set +e
