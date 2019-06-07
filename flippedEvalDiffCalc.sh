#/bin/bash

python src/eval/EvalBinaryImage.py -f results/tmpSunnyEval/origLabelDataset0image0channel-1.nrrd -m results/tmpSunnyEval/deformedLabelDataset0image0channel0.nrrd -o dsc.csv
python src/eval/EvalBinaryImage.py -f results/tmpSunnyEval/origLabelDataset0image0channel0.nrrd -m results/tmpSunnyEval/deformedLabelDataset0image0channel-1.nrrd -o dsc.csv

python src/eval/EvalBinaryImage.py -f results/flippedEval/origLabelDataset1image0channel-1.nrrd -m results/flippedEval/deformedLabelDataset1image0channel0.nrrd -o dsc.csv
python src/eval/EvalBinaryImage.py -f results/flippedEval/origLabelDataset1image0channel0.nrrd -m results/flippedEval/deformedLabelDataset1image0channel-1.nrrd -o dsc.csv

python src/eval/EvalBinaryImage.py -f results/flippedEval/origLabelDataset2image0channel-1.nrrd -m results/flippedEval/deformedLabelDataset2image0channel0.nrrd -o dsc.csv
python src/eval/EvalBinaryImage.py -f results/flippedEval/origLabelDataset2image0channel0.nrrd -m results/flippedEval/deformedLabelDataset2image0channel-1.nrrd -o dsc.csv

python src/eval/EvalBinaryImage.py -f results/flippedEval/origLabelDataset3image0channel-1.nrrd -m results/flippedEval/deformedLabelDataset3image0channel0.nrrd -o dsc.csv
python src/eval/EvalBinaryImage.py -f results/flippedEval/origLabelDataset3image0channel0.nrrd -m results/flippedEval/deformedLabelDataset3image0channel-1.nrrd -o dsc.csv

exit 0

python src/eval/LandmarkHandler.py --path0=resources/DirLab3DEval/Case6/10.pts --path1=results/flippedEval/dataset0channel10deformed.pts --calcDiff > distances0to1Orig.txt
python src/eval/LandmarkHandler.py --path0=resources/DirLab3DEval/Case6/00.pts --path1=results/flippedEval/dataset0channel00deformed.pts --calcDiff > distances10toOrig.txt

python src/eval/LandmarkHandler.py --path0=resources/DirLab3DEval/Case7/10.pts --path1=results/flippedEval/dataset1channel10deformed.pts --calcDiff > distances0to1Orig.txt
python src/eval/LandmarkHandler.py --path0=resources/DirLab3DEval/Case7/00.pts --path1=results/flippedEval/dataset1channel00deformed.pts --calcDiff > distances10toOrig.txt

exit 0

python src/eval/LandmarkHandler.py --path0=resources/DirLab3DEval/Case8/10.pts --path1=results/flippedEval/dataset0channel00deformed.pts --calcDiff > distances0to1Orig.txt
python src/eval/LandmarkHandler.py --path0=resources/DirLab3DEval/Case8/00.pts --path1=results/flippedEval/dataset0channel10deformed.pts --calcDiff > distances10toOrig.txt

python src/eval/LandmarkHandler.py --path0=resources/DirLab3DEval/Case8/10.pts --path1=results/flippedEval/dataset1channel10deformed.pts --calcDiff > distances0to10Flipped.txt
python src/eval/LandmarkHandler.py --path0=resources/DirLab3DEval/Case8/00.pts --path1=results/flippedEval/dataset1channel00deformed.pts --calcDiff > distances10to0Flipped.txt

exit 0
