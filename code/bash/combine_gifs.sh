#!/bin/bash

# Scripts takes two arguments, first is directory of GIFs to combine and second is name of output GIF

LIST=`ls -v $1`
cd $1
echo $LIST
convert $LIST $2
cd ..
