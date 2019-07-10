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

python src/eval/LandmarkHandler.py --path0=resources/Popi4DEval/Popi0"${datasetNumberP1}"/"${pointName}".pts --path1=results/Popi4DEval/deformationFieldDataset"${dataSetNumber}"image0channel"${i}".nrrd --deformPoints

pointName="${pointName}deformed"
if [ $i -eq "-1" ]
then
i=9
fi

if [ ${dataSetNumber} -lt 3 ] || [ ${i} -eq 0 ] || [ ${i} -eq 5 ]
then
python src/eval/LandmarkHandler.py --path0=resources/Popi4DEval/Popi0"${datasetNumberP1}"/"${i}"0.pts --path1=resources/Popi4DEval/Popi0"${datasetNumberP1}"/"${pointName}".pts --calcDiff --calcDiff > distances.txt
else
echo "" >> distances.csv
fi



done

find resources/Popi4DEval/ -name '*deformed.pts' -exec rm {} \; 


exit 0
