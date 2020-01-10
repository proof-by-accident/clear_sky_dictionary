#!/bin/bash

LIST=`ls -v $1`
cd $1
echo $LIST
convert $LIST $2
cd ..
