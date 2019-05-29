#!/bin/bash

searchDir=$1
mhdFiles=$(find "$searchDir" -maxdepth 1 -type f -name '*.mhd')
for mhdFile in $mhdFiles
do
newFileName="${mhdFile%.*}.nrrd"
plastimatch convert --input $mhdFile --output-img $newFileName
rawFileName="${mhdFile%.*}.raw"
rm $mhdFile $rawFileName
done
