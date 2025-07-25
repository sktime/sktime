#!/bin/bash -e

# This script is used to build the system dependencies for MacOS
# It is called by test workflow in .github/workflows/
# To build necessary system dependencies for MacOS, run:

# check if os is MacOS using uname
if [ "$(uname)" = "Darwin" ]; then
    # install necessary dependencies
    echo "installing necessary dependencies..."
    brew install libomp
    echo "Verifying libomp installation..."
    brew_prefix="$(brew --prefix libomp)"
    ls -l "${brew_prefix}/lib/libomp.dylib"
    {
        echo "DYLD_LIBRARY_PATH=${brew_prefix}/lib:\$DYLD_LIBRARY_PATH"
        echo "LDFLAGS=-L${brew_prefix}/lib"
        echo "CPPFLAGS=-I${brew_prefix}/include"
    } >> "$GITHUB_ENV"
else
    echo "This script is intended to run on macOS (Darwin)."
fi
