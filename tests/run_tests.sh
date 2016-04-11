#!/bin/bash

p=`pwd`
rm -r output/*
nosetests
cd output/testInterp
pdfjoin *.pdf --outfile all.pdf
cd ..
#xpdf testWarpSparse/sedwarp.pdf
cd testBolo
xpdf test2lcs.pdf
cd $p
