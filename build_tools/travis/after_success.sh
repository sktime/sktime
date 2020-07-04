#!/bin/bash

# adapted from https://github.com/scikit-learn/scikit-learn/blob/master/build_tools/travis/after_success.sh

# This script is meant to be called by the "after_success" step defined in
# .travis.yml. See https://docs.travis-ci.com/ for more details.

# License: 3-clause BSD

<<<<<<< HEAD
if [[ "$COVERAGE" == "true" ]];
=======
if [ "$COVERAGE" == "true" ];
>>>>>>> 67c56be8b1e838f2628df829946f795b7dba9aed
then
    # Need to run codecov from a git checkout, so we copy .coverage
    # from TEST_DIR where pytest has been run
    cp "$TEST_DIR"/.coverage "$TRAVIS_BUILD_DIR"

    # Ignore codecov failures as the codecov server is not
    # very reliable but we don't want travis to report a failure
    # in the github UI just because the coverage report failed to
    # be published.
<<<<<<< HEAD
    codecov --root "$TRAVIS_BUILD_DIR" || echo "codecov upload failed"
=======
    codecov --root "$TRAVIS_BUILD_DIR" || echo "Codecov upload failed"
else
  echo "Skipped codecov upload"
>>>>>>> 67c56be8b1e838f2628df829946f795b7dba9aed
fi

# Build website on master branch
if [ "$TRAVIS_JOB_NAME" == "$DEPLOY_JOB_NAME" ] && [ "$TRAVIS_BRANCH" == "$DEPLOY_BRANCH" ];
then
<<<<<<< HEAD

  # Add packages for docs generation, specified in EXTRAS_REQUIRE in setup.py
  pip install -e .[docs]

  # generate website
  make docs
=======
  # Add packages for docs generation, specified in EXTRAS_REQUIRE in setup.py
  pip install -e .[docs]

  # we have to manually install bug fix here to parse md docs
  # https://github.com/sphinx-doc/sphinx/issues/2840
  pip install git+https://github.com/crossnox/m2r@dev#egg=m2r

  # generate website
  make docs
else
  echo "Skipped building docs"
>>>>>>> 67c56be8b1e838f2628df829946f795b7dba9aed
fi
