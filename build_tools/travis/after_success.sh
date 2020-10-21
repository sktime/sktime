#!/bin/bash

# This script is meant to be called by the "after_success" step defined in
# .travis.yml. See https://docs.travis-ci.com/ for more details.

# License: 3-clause BSD

if [ "$COVERAGE" == "true" ]; then
  # Ignore codecov failures as the codecov server is not
  # very reliable but we don't want travis to report a failure
  # in the GitHub UI just because the coverage report failed to
  # be published. Since we ran the tests in a separate repo, we need to
  # point the uploader to the generated coverage report.
  # see https://docs.codecov.io/docs/about-the-codecov-bash-uploader
  cp "$TEST_DIR"/.coverage .
  bash <(curl -s https://codecov.io/bash) -s "$TEST_DIR" || echo "Codecov upload failed"
else
  echo "Skipped codecov upload"
fi


# Docs are no longer deployed via travis but now instead via readthedocs
## Build website on master branch, also see deploy section in .travis.yml
#if [ "$TRAVIS_JOB_NAME" == "$DEPLOY_JOB_NAME" ] && [ "$TRAVIS_BRANCH" == "$DEPLOY_BRANCH" ]; then
#  # Add packages for docs generation, specified in EXTRAS_REQUIRE in setup.py
#  pip install -r docs/requirements.txt
#
#  # generate website
#  make docs
#else
#  echo "Skipped building docs"
#fi
