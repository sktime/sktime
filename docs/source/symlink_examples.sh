#!/bin/bash

# helper script to create symbolic links to
# notebooks in example folder

# remove all
rm -rf examples/*

# cd into website folder
cd examples/ || return

# create symbolic links in website folder
ln -s ../../examples/* .

# return to initial folder
cd - || exit
