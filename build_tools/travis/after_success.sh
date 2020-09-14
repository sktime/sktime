#!/bin/bash

# This script is meant to be called by the "after_success" step defined in
# .travis.yml. See https://docs.travis-ci.com/ for more details.

# License: 3-clause BSD

if [ "$COVERAGE" == "true" ];
then
    # see https://docs.codecov.io/docs/about-the-codecov-bash-uploader
    # Ignore codecov failures as the codecov server is not
    # very reliable but we don't want travis to report a failure
    # in the GitHub UI just because the coverage report failed to
    # be published.
    bash <(curl -s https://codecov.io/bash) -f "$TEST_DIR"/.coverage || echo "Codecov upload failed"
else
  echo "Skipped codecov upload"
fi

# Build website on master branch
if [ "$TRAVIS_JOB_NAME" == "$DEPLOY_JOB_NAME" ] && [ "$TRAVIS_BRANCH" == "$DEPLOY_BRANCH" ];
then
  # Add packages for docs generation, specified in EXTRAS_REQUIRE in setup.py
  pip install -e .[docs]

  # we have to manually install bug fix here to parse md docs
  # https://github.com/sphinx-doc/sphinx/issues/2840
  pip install git+https://github.com/crossnox/m2r@dev#egg=m2r

  # generate website
  make docs
else
  echo "Skipped building docs"
fi
