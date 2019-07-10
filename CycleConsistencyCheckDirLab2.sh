#!/bin/bash

startTimeBin=$1
numberOfTimeBins=$2
dataSetNumber=$3

datasetNumberP1=$(echo "${dataSetNumber} + 1" | bc)
numberOfTimeBinsM1=$(echo "${numberOfTimeBins} - 1" | bc)
iterationRange=$(seq ${numberOfTimeBinsM1} -1 0 | xargs -i echo "(({} + ${startTimeBin} + 1)%${numberOfTimeBins})-1" | bc)

pointName="${startTimeBin}0"

for i in $iterationRange
do
python src/eval/LandmarkHandler.py --path0=resources/DirLab4DEval/Case"${datasetNumberP1}"/"${pointName}".pts --path1=results/DirLab4DEval/deformationFieldDataset"${dataSetNumber}"image0channel"${i}".nrrd --deformPoints
pointName="${pointName}deformed"

if [[ $i =~ ^(0|1|2|3|4|5)$ ]]
then
python src/eval/LandmarkHandler.py --path0=resources/DirLab4DEval/Case"${datasetNumberP1}"/"${i}"0.pts --path1=resources/DirLab4DEval/Case"${datasetNumberP1}"/"${pointName}".pts --calcDiff --calcDiff > distances.txt
else
echo "" >> distances.csv
fi
done

find resources/DirLab4DEval/ -name '*deformed.pts' -exec rm {} \; 

exit 0
