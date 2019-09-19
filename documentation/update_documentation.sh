#!/bin/bash


# create symbolic links to notebooks in example folder
cd source/examples/

FILES=../../../examples/*

for file in $FILES
do
  echo "Creating symbolic link to file $file"
  ln -s $file .
done

cd ../..