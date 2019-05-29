#!/bin/bash

BASEDIR=$(dirname "$0")
if [ $# -gt 0 ]
 then
 BASEDIR=$1
fi

pythonScript=/home/fechter/workspace/TorchSandbox/src/eval/JacobiStats.py
outputFile=jacobiEval.txt
if [ $# -gt 1 ]
 then
 outputFile=$2
fi

images=$(find $BASEDIR -type f -name 'deformationField*.nrrd')
for image in $images
do
	echo ${image}
	plastimatch jacobian --input ${image} --output-img defField_jac.nrrd
	python $pythonScript --jcFile=defField_jac.nrrd --output=$outputFile
  	rm defField_jac.nrrd
done


exit 0
