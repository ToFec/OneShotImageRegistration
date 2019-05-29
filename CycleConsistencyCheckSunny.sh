#!/bin/bash

startTimeBin=$1
numberOfTimeBins=$2
dataSetNumber=$3

numberOfTimeBinsM1=$(echo "${numberOfTimeBins} - 1" | bc)
iterationRange=$(seq 0 ${numberOfTimeBinsM1} | xargs -i echo "(({} + ${startTimeBin} + 1)%${numberOfTimeBins})-1" | bc)

if [ ${startTimeBin} -eq ${numberOfTimeBinsM1} ]
then
startTimeBin=-1
fi


labelName="origLabelDataset"${dataSetNumber}"image0channel"${startTimeBin}""

for i in $iterationRange
do

iP1=$(echo "${i} + 1" | bc)
if [ ${iP1} -eq ${numberOfTimeBinsM1} ]
then
iP1=-1
fi

python src/eval/ImageWarper.py -i results/SunnyBrook4D/${labelName}.nrrd -d results/SunnyBrook4D/deformationFieldDataset${dataSetNumber}image0channel${i}.nrrd -o results/SunnyBrook4D/${labelName}def.nrrd -b

labelName="${labelName}def"


python src/eval/EvalBinaryImage.py -f results/SunnyBrook4D/origLabelDataset${dataSetNumber}image0channel${iP1}.nrrd -m results/SunnyBrook4D/${labelName}.nrrd -o dsc.csv



done


exit 0
