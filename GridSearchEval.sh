#!/bin/bash

BASEDIR=$(dirname "$0")
if [ $# -gt 0 ]
 then
 BASEDIR=$1
fi

pythonScript='/home/fechter/workspace/TorchSandbox/src/eval/LandmarkHandler.py'
origLMPath='/home/fechter/workspace/TorchSandbox/resources/PopiTmp'

cd ${BASEDIR}
echo 'ccw;smoothW;cycleW;vecLengthW;timeForTraining;finalLoss;numberOfIterations;newDist0;newDist1;oldDist0;oldDist1' > "GridSearchResult.csv"
directories=$(find . -maxdepth 1 -type d)
for directory in $directories
do
    if [ "$directory" != "." ]
    then
	cd ${directory}
	ccw=$(echo ${directory} | grep -o -P '(?<=ccW)[0-9.]+')
	smothW=$(echo ${directory} | grep -o -P '(?<=smoothW)[0-9.]+')
	cycleW=$(echo ${directory} | grep -o -P '(?<=cycleW)[0-9.]+')
	vecLenghtW=$(echo ${directory} | grep -o -P '(?<=vecLengthW)[0-9.]+')
	iterLossTime=$(cat lossIterLog.csv | grep -o -x -E '[0-9;.]+')
	cd ..
	python ${pythonScript} --path0=${origLMPath} --path1=${directory} --calcDiff
	newDist=$(cat distances.csv)
	python ${pythonScript} --path0=${origLMPath} --path1=${origLMPath} --calcDiff
	oldDist=$(cat distances.csv)
	rm distances.csv
	echo "${ccw};${smothW};${cycleW};${vecLenghtW};${iterLossTime};${newDist}${oldDist}" >> "GridSearchResult.csv"
    fi    
done
cd ${ROOTDIR}
exit 0
