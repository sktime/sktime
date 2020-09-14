#!/bin/bash

# adapted from https://github.com/scikit-learn/scikit-learn/blob/master/build_tools/travis/test_script.sh

# This script is meant to be called by the "script" step defined in
# .travis.yml. See https://docs.travis-ci.com/ for more details.
# The behavior of the script is controlled by environment variabled defined
# in the .travis.yml in the top level folder of the project.

# License: 3-clause BSD

# exit the script if any statement returns a non-true return value
set -e

# print test environment
conda list -n testenv

run_tests() {
    TEST_CMD="pytest --showlocals --durations=10 --pyargs"

    # Optionally run coverage
    if [[ "$COVERAGE" == "true" ]]; then
        TEST_CMD="$TEST_CMD --cov-report=xml --cov=sktime"
    fi
    set -x  # print executed commands to the terminal

    $TEST_CMD sktime
}

run_tests
set +e
