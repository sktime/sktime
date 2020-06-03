#!/bin/bash

# helper script to create symbolic links to
# notebooks in example folder

# remove all
rm -r source/examples/*

# cd into website folder
cd source/examples/ || return

# create symbolic links in website folder
ln -s ../../../examples/* .

# return to initial folder
cd ../..
