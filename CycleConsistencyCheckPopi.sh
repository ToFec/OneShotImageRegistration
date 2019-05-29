#!/bin/bash
echo "Dataset 00"

python src/eval/LandmarkHandler.py --path0=resources/Popi4DEval/Popi01/00.pts --path1=results/Popi4DEval/deformationFieldDataset0image0channel-1.nrrd --deformPoints
python src/eval/LandmarkHandler.py --path0=resources/Popi4DEval/Popi01/90.pts --path1=resources/Popi4DEval/Popi01/00deformed.pts --calcDiff --calcDiff > distances.txt

python src/eval/LandmarkHandler.py --path0=resources/Popi4DEval/Popi01/00deformed.pts --path1=results/Popi4DEval/deformationFieldDataset0image0channel8.nrrd --deformPoints
python src/eval/LandmarkHandler.py --path0=resources/Popi4DEval/Popi01/80.pts --path1=resources/Popi4DEval/Popi01/00deformeddeformed.pts --calcDiff --calcDiff > distances.txt

python src/eval/LandmarkHandler.py --path0=resources/Popi4DEval/Popi01/00deformeddeformed.pts --path1=results/Popi4DEval/deformationFieldDataset0image0channel7.nrrd --deformPoints
python src/eval/LandmarkHandler.py --path0=resources/Popi4DEval/Popi01/70.pts --path1=resources/Popi4DEval/Popi01/00deformeddeformeddeformed.pts --calcDiff --calcDiff > distances.txt

python src/eval/LandmarkHandler.py --path0=resources/Popi4DEval/Popi01/00deformeddeformeddeformed.pts --path1=results/Popi4DEval/deformationFieldDataset0image0channel6.nrrd --deformPoints
python src/eval/LandmarkHandler.py --path0=resources/Popi4DEval/Popi01/60.pts --path1=resources/Popi4DEval/Popi01/00deformeddeformeddeformeddeformed.pts --calcDiff --calcDiff > distances.txt

python src/eval/LandmarkHandler.py --path0=resources/Popi4DEval/Popi01/00deformeddeformeddeformeddeformed.pts --path1=results/Popi4DEval/deformationFieldDataset0image0channel5.nrrd --deformPoints
python src/eval/LandmarkHandler.py --path0=resources/Popi4DEval/Popi01/50.pts --path1=resources/Popi4DEval/Popi01/00deformeddeformeddeformeddeformeddeformed.pts --calcDiff --calcDiff > distances.txt

python src/eval/LandmarkHandler.py --path0=resources/Popi4DEval/Popi01/00deformeddeformeddeformeddeformeddeformed.pts --path1=results/Popi4DEval/deformationFieldDataset0image0channel4.nrrd --deformPoints
python src/eval/LandmarkHandler.py --path0=resources/Popi4DEval/Popi01/40.pts --path1=resources/Popi4DEval/Popi01/00deformeddeformeddeformeddeformeddeformeddeformed.pts --calcDiff --calcDiff > distances.txt

python src/eval/LandmarkHandler.py --path0=resources/Popi4DEval/Popi01/00deformeddeformeddeformeddeformeddeformeddeformed.pts --path1=results/Popi4DEval/deformationFieldDataset0image0channel3.nrrd --deformPoints
python src/eval/LandmarkHandler.py --path0=resources/Popi4DEval/Popi01/30.pts --path1=resources/Popi4DEval/Popi01/00deformeddeformeddeformeddeformeddeformeddeformeddeformed.pts --calcDiff --calcDiff > distances.txt

python src/eval/LandmarkHandler.py --path0=resources/Popi4DEval/Popi01/00deformeddeformeddeformeddeformeddeformeddeformeddeformed.pts --path1=results/Popi4DEval/deformationFieldDataset0image0channel2.nrrd --deformPoints
python src/eval/LandmarkHandler.py --path0=resources/Popi4DEval/Popi01/20.pts --path1=resources/Popi4DEval/Popi01/00deformeddeformeddeformeddeformeddeformeddeformeddeformeddeformed.pts --calcDiff > distances.txt

python src/eval/LandmarkHandler.py --path0=resources/Popi4DEval/Popi01/00deformeddeformeddeformeddeformeddeformeddeformeddeformeddeformed.pts --path1=results/Popi4DEval/deformationFieldDataset0image0channel1.nrrd --deformPoints
python src/eval/LandmarkHandler.py --path0=resources/Popi4DEval/Popi01/10.pts --path1=resources/Popi4DEval/Popi01/00deformeddeformeddeformeddeformeddeformeddeformeddeformeddeformeddeformed.pts --calcDiff --calcDiff > distances.txt

python src/eval/LandmarkHandler.py --path0=resources/Popi4DEval/Popi01/00deformeddeformeddeformeddeformeddeformeddeformeddeformeddeformeddeformed.pts --path1=results/Popi4DEval/deformationFieldDataset0image0channel0.nrrd --deformPoints
python src/eval/LandmarkHandler.py --path0=resources/Popi4DEval/Popi01/00.pts --path1=resources/Popi4DEval/Popi01/00deformeddeformeddeformeddeformeddeformeddeformeddeformeddeformeddeformeddeformed.pts --calcDiff --calcDiff > distances.txt

echo "Dataset 01"

python src/eval/LandmarkHandler.py --path0=resources/Popi4DEval/Popi02/00.pts --path1=results/Popi4DEval/deformationFieldDataset1image0channel-1.nrrd --deformPoints
python src/eval/LandmarkHandler.py --path0=resources/Popi4DEval/Popi02/90.pts --path1=resources/Popi4DEval/Popi02/00deformed.pts --calcDiff --calcDiff > distances.txt

python src/eval/LandmarkHandler.py --path0=resources/Popi4DEval/Popi02/00deformed.pts --path1=results/Popi4DEval/deformationFieldDataset1image0channel8.nrrd --deformPoints
python src/eval/LandmarkHandler.py --path0=resources/Popi4DEval/Popi02/80.pts --path1=resources/Popi4DEval/Popi02/00deformeddeformed.pts --calcDiff --calcDiff > distances.txt

python src/eval/LandmarkHandler.py --path0=resources/Popi4DEval/Popi02/00deformeddeformed.pts --path1=results/Popi4DEval/deformationFieldDataset1image0channel7.nrrd --deformPoints
python src/eval/LandmarkHandler.py --path0=resources/Popi4DEval/Popi02/70.pts --path1=resources/Popi4DEval/Popi02/00deformeddeformeddeformed.pts --calcDiff --calcDiff > distances.txt

python src/eval/LandmarkHandler.py --path0=resources/Popi4DEval/Popi02/00deformeddeformeddeformed.pts --path1=results/Popi4DEval/deformationFieldDataset1image0channel6.nrrd --deformPoints
python src/eval/LandmarkHandler.py --path0=resources/Popi4DEval/Popi02/60.pts --path1=resources/Popi4DEval/Popi02/00deformeddeformeddeformeddeformed.pts --calcDiff --calcDiff > distances.txt

python src/eval/LandmarkHandler.py --path0=resources/Popi4DEval/Popi02/00deformeddeformeddeformeddeformed.pts --path1=results/Popi4DEval/deformationFieldDataset1image0channel5.nrrd --deformPoints
python src/eval/LandmarkHandler.py --path0=resources/Popi4DEval/Popi02/50.pts --path1=resources/Popi4DEval/Popi02/00deformeddeformeddeformeddeformeddeformed.pts --calcDiff --calcDiff > distances.txt

python src/eval/LandmarkHandler.py --path0=resources/Popi4DEval/Popi02/00deformeddeformeddeformeddeformeddeformed.pts --path1=results/Popi4DEval/deformationFieldDataset1image0channel4.nrrd --deformPoints
python src/eval/LandmarkHandler.py --path0=resources/Popi4DEval/Popi02/40.pts --path1=resources/Popi4DEval/Popi02/00deformeddeformeddeformeddeformeddeformeddeformed.pts --calcDiff --calcDiff > distances.txt

python src/eval/LandmarkHandler.py --path0=resources/Popi4DEval/Popi02/00deformeddeformeddeformeddeformeddeformeddeformed.pts --path1=results/Popi4DEval/deformationFieldDataset1image0channel3.nrrd --deformPoints
python src/eval/LandmarkHandler.py --path0=resources/Popi4DEval/Popi02/30.pts --path1=resources/Popi4DEval/Popi02/00deformeddeformeddeformeddeformeddeformeddeformeddeformed.pts --calcDiff --calcDiff > distances.txt

python src/eval/LandmarkHandler.py --path0=resources/Popi4DEval/Popi02/00deformeddeformeddeformeddeformeddeformeddeformeddeformed.pts --path1=results/Popi4DEval/deformationFieldDataset1image0channel2.nrrd --deformPoints
python src/eval/LandmarkHandler.py --path0=resources/Popi4DEval/Popi02/20.pts --path1=resources/Popi4DEval/Popi02/00deformeddeformeddeformeddeformeddeformeddeformeddeformeddeformed.pts --calcDiff > distances.txt

python src/eval/LandmarkHandler.py --path0=resources/Popi4DEval/Popi02/00deformeddeformeddeformeddeformeddeformeddeformeddeformeddeformed.pts --path1=results/Popi4DEval/deformationFieldDataset1image0channel1.nrrd --deformPoints
python src/eval/LandmarkHandler.py --path0=resources/Popi4DEval/Popi02/10.pts --path1=resources/Popi4DEval/Popi02/00deformeddeformeddeformeddeformeddeformeddeformeddeformeddeformeddeformed.pts --calcDiff --calcDiff > distances.txt

python src/eval/LandmarkHandler.py --path0=resources/Popi4DEval/Popi02/00deformeddeformeddeformeddeformeddeformeddeformeddeformeddeformeddeformed.pts --path1=results/Popi4DEval/deformationFieldDataset1image0channel0.nrrd --deformPoints
python src/eval/LandmarkHandler.py --path0=resources/Popi4DEval/Popi02/00.pts --path1=resources/Popi4DEval/Popi02/00deformeddeformeddeformeddeformeddeformeddeformeddeformeddeformeddeformeddeformed.pts --calcDiff --calcDiff > distances.txt

echo "Dataset 02"

python src/eval/LandmarkHandler.py --path0=resources/Popi4DEval/Popi03/00.pts --path1=results/Popi4DEval/deformationFieldDataset2image0channel-1.nrrd --deformPoints
python src/eval/LandmarkHandler.py --path0=resources/Popi4DEval/Popi03/90.pts --path1=resources/Popi4DEval/Popi03/00deformed.pts --calcDiff --calcDiff > distances.txt

python src/eval/LandmarkHandler.py --path0=resources/Popi4DEval/Popi03/00deformed.pts --path1=results/Popi4DEval/deformationFieldDataset2image0channel8.nrrd --deformPoints
python src/eval/LandmarkHandler.py --path0=resources/Popi4DEval/Popi03/80.pts --path1=resources/Popi4DEval/Popi03/00deformeddeformed.pts --calcDiff --calcDiff > distances.txt

python src/eval/LandmarkHandler.py --path0=resources/Popi4DEval/Popi03/00deformeddeformed.pts --path1=results/Popi4DEval/deformationFieldDataset2image0channel7.nrrd --deformPoints
python src/eval/LandmarkHandler.py --path0=resources/Popi4DEval/Popi03/70.pts --path1=resources/Popi4DEval/Popi03/00deformeddeformeddeformed.pts --calcDiff --calcDiff > distances.txt

python src/eval/LandmarkHandler.py --path0=resources/Popi4DEval/Popi03/00deformeddeformeddeformed.pts --path1=results/Popi4DEval/deformationFieldDataset2image0channel6.nrrd --deformPoints
python src/eval/LandmarkHandler.py --path0=resources/Popi4DEval/Popi03/60.pts --path1=resources/Popi4DEval/Popi03/00deformeddeformeddeformeddeformed.pts --calcDiff --calcDiff > distances.txt

python src/eval/LandmarkHandler.py --path0=resources/Popi4DEval/Popi03/00deformeddeformeddeformeddeformed.pts --path1=results/Popi4DEval/deformationFieldDataset2image0channel5.nrrd --deformPoints
python src/eval/LandmarkHandler.py --path0=resources/Popi4DEval/Popi03/50.pts --path1=resources/Popi4DEval/Popi03/00deformeddeformeddeformeddeformeddeformed.pts --calcDiff --calcDiff > distances.txt

python src/eval/LandmarkHandler.py --path0=resources/Popi4DEval/Popi03/00deformeddeformeddeformeddeformeddeformed.pts --path1=results/Popi4DEval/deformationFieldDataset2image0channel4.nrrd --deformPoints
python src/eval/LandmarkHandler.py --path0=resources/Popi4DEval/Popi03/40.pts --path1=resources/Popi4DEval/Popi03/00deformeddeformeddeformeddeformeddeformeddeformed.pts --calcDiff --calcDiff > distances.txt

python src/eval/LandmarkHandler.py --path0=resources/Popi4DEval/Popi03/00deformeddeformeddeformeddeformeddeformeddeformed.pts --path1=results/Popi4DEval/deformationFieldDataset2image0channel3.nrrd --deformPoints
python src/eval/LandmarkHandler.py --path0=resources/Popi4DEval/Popi03/30.pts --path1=resources/Popi4DEval/Popi03/00deformeddeformeddeformeddeformeddeformeddeformeddeformed.pts --calcDiff --calcDiff > distances.txt

python src/eval/LandmarkHandler.py --path0=resources/Popi4DEval/Popi03/00deformeddeformeddeformeddeformeddeformeddeformeddeformed.pts --path1=results/Popi4DEval/deformationFieldDataset2image0channel2.nrrd --deformPoints
python src/eval/LandmarkHandler.py --path0=resources/Popi4DEval/Popi03/20.pts --path1=resources/Popi4DEval/Popi03/00deformeddeformeddeformeddeformeddeformeddeformeddeformeddeformed.pts --calcDiff > distances.txt

python src/eval/LandmarkHandler.py --path0=resources/Popi4DEval/Popi03/00deformeddeformeddeformeddeformeddeformeddeformeddeformeddeformed.pts --path1=results/Popi4DEval/deformationFieldDataset2image0channel1.nrrd --deformPoints
python src/eval/LandmarkHandler.py --path0=resources/Popi4DEval/Popi03/10.pts --path1=resources/Popi4DEval/Popi03/00deformeddeformeddeformeddeformeddeformeddeformeddeformeddeformeddeformed.pts --calcDiff --calcDiff > distances.txt

python src/eval/LandmarkHandler.py --path0=resources/Popi4DEval/Popi03/00deformeddeformeddeformeddeformeddeformeddeformeddeformeddeformeddeformed.pts --path1=results/Popi4DEval/deformationFieldDataset2image0channel0.nrrd --deformPoints
python src/eval/LandmarkHandler.py --path0=resources/Popi4DEval/Popi03/00.pts --path1=resources/Popi4DEval/Popi03/00deformeddeformeddeformeddeformeddeformeddeformeddeformeddeformeddeformeddeformed.pts --calcDiff --calcDiff > distances.txt
