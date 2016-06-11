#!/bin/bash

blah=`ls ../run/*.h5_lores | grep 2006`
echo $blah

for f in $blah; do
    name=`echo $f | awk -F".h5" '{print $1}'`
    echo $name
    ipython representative.py $name
done
