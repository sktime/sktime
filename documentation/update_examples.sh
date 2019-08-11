#!/usr/bin/env bash
# run from inside sktime/documentation/

# change directory to create symbolic links from within the target directory
cd source/examples/

# get all files in examples directory
FILES=../../../examples/*

for file in $FILES
do
  # create symbolic links from example files to target directory
  echo "Creating symbolic link to $file"
  ln -sf $file .
done

# return to original directory
cd ../..
