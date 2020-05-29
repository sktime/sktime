#!/bin/bash

# adapted from https://github.com/scikit-learn/scikit-learn/blob/master/build_tools/travis/after_success.sh

# This script is meant to be called by the "after_success" step defined in
# .travis.yml. See https://docs.travis-ci.com/ for more details.

# License: 3-clause BSD

if [[ "$COVERAGE" == "true" ]];
then
    # Need to run codecov from a git checkout, so we copy .coverage
    # from TEST_DIR where pytest has been run
    cp "$TEST_DIR"/.coverage "$TRAVIS_BUILD_DIR"

    # Ignore codecov failures as the codecov server is not
    # very reliable but we don't want travis to report a failure
    # in the github UI just because the coverage report failed to
    # be published.
    codecov --root "$TRAVIS_BUILD_DIR" || echo "codecov upload failed"
fi

# Build website on master branch
if [[ "$TRAVIS_OS_NAME" == "$TRAVIS_DEPLOY_OS_NAME" ]] && [[ "$TRAVIS_BRANCH" == "$TRAVIS_DEPLOY_BRANCH" ]];
then
  # Add packages for docs generation, specified in EXTRAS_REQUIRE in setup.py
  pip install -e .[docs]

  # generate website
  make docs
fi
