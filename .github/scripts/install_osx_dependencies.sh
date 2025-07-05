#!/bin/bash -e

# This script is used to build the system dependencies for MacOS
# It is called by test workflow in .github/workflows/
# To build necessary system dependencies for MacOS, run:

# check if os is MacOS using uname
if [ "$(uname)" = "Darwin" ]; then
    echo "installing necessary dependencies..."
    brew install libomp

    LIBOMP_PATH="$(brew --prefix libomp)"
    echo "Verifying libomp installation..."
    ls -l "$LIBOMP_PATH/lib/libomp.dylib"

    {
        echo "DYLD_LIBRARY_PATH=$LIBOMP_PATH/lib:\$DYLD_LIBRARY_PATH"
        echo "LDFLAGS=-L$LIBOMP_PATH/lib"
        echo "CPPFLAGS=-I$LIBOMP_PATH/include"
    } >> "$GITHUB_ENV"
else
    echo "This script is intended to run on macOS (Darwin)."
fi
