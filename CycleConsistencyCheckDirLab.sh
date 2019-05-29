#/bin/bash
echo "Dataset0"

python src/eval/LandmarkHandler.py --path0=resources/DirLab4DEval/Case1/00.pts --path1=results/DirLab4DEval/deformationFieldDataset0image0channel-1.nrrd --deformPoints

python src/eval/LandmarkHandler.py --path0=resources/DirLab4DEval/Case1/00deformed.pts --path1=results/DirLab4DEval/deformationFieldDataset0image0channel8.nrrd --deformPoints

python src/eval/LandmarkHandler.py --path0=resources/DirLab4DEval/Case1/00deformeddeformed.pts --path1=results/DirLab4DEval/deformationFieldDataset0image0channel7.nrrd --deformPoints

python src/eval/LandmarkHandler.py --path0=resources/DirLab4DEval/Case1/00deformeddeformeddeformed.pts --path1=results/DirLab4DEval/deformationFieldDataset0image0channel6.nrrd --deformPoints

python src/eval/LandmarkHandler.py --path0=resources/DirLab4DEval/Case1/00deformeddeformeddeformeddeformed.pts --path1=results/DirLab4DEval/deformationFieldDataset0image0channel5.nrrd --deformPoints
python src/eval/LandmarkHandler.py --path0=resources/DirLab4DEval/Case1/50.pts --path1=resources/DirLab4DEval/Case1/00deformeddeformeddeformeddeformeddeformed.pts --calcDiff --calcDiff > distances.txt

python src/eval/LandmarkHandler.py --path0=resources/DirLab4DEval/Case1/00deformeddeformeddeformeddeformeddeformed.pts --path1=results/DirLab4DEval/deformationFieldDataset0image0channel4.nrrd --deformPoints
python src/eval/LandmarkHandler.py --path0=resources/DirLab4DEval/Case1/40.pts --path1=resources/DirLab4DEval/Case1/00deformeddeformeddeformeddeformeddeformeddeformed.pts --calcDiff --calcDiff > distances.txt

python src/eval/LandmarkHandler.py --path0=resources/DirLab4DEval/Case1/00deformeddeformeddeformeddeformeddeformeddeformed.pts --path1=results/DirLab4DEval/deformationFieldDataset0image0channel3.nrrd --deformPoints
python src/eval/LandmarkHandler.py --path0=resources/DirLab4DEval/Case1/30.pts --path1=resources/DirLab4DEval/Case1/00deformeddeformeddeformeddeformeddeformeddeformeddeformed.pts --calcDiff --calcDiff > distances.txt

python src/eval/LandmarkHandler.py --path0=resources/DirLab4DEval/Case1/00deformeddeformeddeformeddeformeddeformeddeformeddeformed.pts --path1=results/DirLab4DEval/deformationFieldDataset0image0channel2.nrrd --deformPoints
python src/eval/LandmarkHandler.py --path0=resources/DirLab4DEval/Case1/20.pts --path1=resources/DirLab4DEval/Case1/00deformeddeformeddeformeddeformeddeformeddeformeddeformeddeformed.pts --calcDiff > distances.txt

python src/eval/LandmarkHandler.py --path0=resources/DirLab4DEval/Case1/00deformeddeformeddeformeddeformeddeformeddeformeddeformeddeformed.pts --path1=results/DirLab4DEval/deformationFieldDataset0image0channel1.nrrd --deformPoints
python src/eval/LandmarkHandler.py --path0=resources/DirLab4DEval/Case1/10.pts --path1=resources/DirLab4DEval/Case1/00deformeddeformeddeformeddeformeddeformeddeformeddeformeddeformeddeformed.pts --calcDiff --calcDiff > distances.txt

python src/eval/LandmarkHandler.py --path0=resources/DirLab4DEval/Case1/00deformeddeformeddeformeddeformeddeformeddeformeddeformeddeformeddeformed.pts --path1=results/DirLab4DEval/deformationFieldDataset0image0channel0.nrrd --deformPoints
python src/eval/LandmarkHandler.py --path0=resources/DirLab4DEval/Case1/00.pts --path1=resources/DirLab4DEval/Case1/00deformeddeformeddeformeddeformeddeformeddeformeddeformeddeformeddeformeddeformed.pts --calcDiff --calcDiff > distances.txt

echo "Dataset1"

python src/eval/LandmarkHandler.py --path0=resources/DirLab4DEval/Case2/00.pts --path1=results/DirLab4DEval/deformationFieldDataset1image0channel-1.nrrd --deformPoints

python src/eval/LandmarkHandler.py --path0=resources/DirLab4DEval/Case2/00deformed.pts --path1=results/DirLab4DEval/deformationFieldDataset1image0channel8.nrrd --deformPoints

python src/eval/LandmarkHandler.py --path0=resources/DirLab4DEval/Case2/00deformeddeformed.pts --path1=results/DirLab4DEval/deformationFieldDataset1image0channel7.nrrd --deformPoints

python src/eval/LandmarkHandler.py --path0=resources/DirLab4DEval/Case2/00deformeddeformeddeformed.pts --path1=results/DirLab4DEval/deformationFieldDataset1image0channel6.nrrd --deformPoints

python src/eval/LandmarkHandler.py --path0=resources/DirLab4DEval/Case2/00deformeddeformeddeformeddeformed.pts --path1=results/DirLab4DEval/deformationFieldDataset1image0channel5.nrrd --deformPoints
python src/eval/LandmarkHandler.py --path0=resources/DirLab4DEval/Case2/50.pts --path1=resources/DirLab4DEval/Case2/00deformeddeformeddeformeddeformeddeformed.pts --calcDiff --calcDiff > distances.txt

python src/eval/LandmarkHandler.py --path0=resources/DirLab4DEval/Case2/00deformeddeformeddeformeddeformeddeformed.pts --path1=results/DirLab4DEval/deformationFieldDataset1image0channel4.nrrd --deformPoints
python src/eval/LandmarkHandler.py --path0=resources/DirLab4DEval/Case2/40.pts --path1=resources/DirLab4DEval/Case2/00deformeddeformeddeformeddeformeddeformeddeformed.pts --calcDiff --calcDiff > distances.txt

python src/eval/LandmarkHandler.py --path0=resources/DirLab4DEval/Case2/00deformeddeformeddeformeddeformeddeformeddeformed.pts --path1=results/DirLab4DEval/deformationFieldDataset1image0channel3.nrrd --deformPoints
python src/eval/LandmarkHandler.py --path0=resources/DirLab4DEval/Case2/30.pts --path1=resources/DirLab4DEval/Case2/00deformeddeformeddeformeddeformeddeformeddeformeddeformed.pts --calcDiff --calcDiff > distances.txt

python src/eval/LandmarkHandler.py --path0=resources/DirLab4DEval/Case2/00deformeddeformeddeformeddeformeddeformeddeformeddeformed.pts --path1=results/DirLab4DEval/deformationFieldDataset1image0channel2.nrrd --deformPoints
python src/eval/LandmarkHandler.py --path0=resources/DirLab4DEval/Case2/20.pts --path1=resources/DirLab4DEval/Case2/00deformeddeformeddeformeddeformeddeformeddeformeddeformeddeformed.pts --calcDiff > distances.txt

python src/eval/LandmarkHandler.py --path0=resources/DirLab4DEval/Case2/00deformeddeformeddeformeddeformeddeformeddeformeddeformeddeformed.pts --path1=results/DirLab4DEval/deformationFieldDataset1image0channel1.nrrd --deformPoints
python src/eval/LandmarkHandler.py --path0=resources/DirLab4DEval/Case2/10.pts --path1=resources/DirLab4DEval/Case2/00deformeddeformeddeformeddeformeddeformeddeformeddeformeddeformeddeformed.pts --calcDiff --calcDiff > distances.txt

python src/eval/LandmarkHandler.py --path0=resources/DirLab4DEval/Case2/00deformeddeformeddeformeddeformeddeformeddeformeddeformeddeformeddeformed.pts --path1=results/DirLab4DEval/deformationFieldDataset1image0channel0.nrrd --deformPoints
python src/eval/LandmarkHandler.py --path0=resources/DirLab4DEval/Case2/00.pts --path1=resources/DirLab4DEval/Case2/00deformeddeformeddeformeddeformeddeformeddeformeddeformeddeformeddeformeddeformed.pts --calcDiff --calcDiff > distances.txt

echo "Dataset 2"

python src/eval/LandmarkHandler.py --path0=resources/DirLab4DEval/Case3/00.pts --path1=results/DirLab4DEval/deformationFieldDataset2image0channel-1.nrrd --deformPoints

python src/eval/LandmarkHandler.py --path0=resources/DirLab4DEval/Case3/00deformed.pts --path1=results/DirLab4DEval/deformationFieldDataset2image0channel8.nrrd --deformPoints

python src/eval/LandmarkHandler.py --path0=resources/DirLab4DEval/Case3/00deformeddeformed.pts --path1=results/DirLab4DEval/deformationFieldDataset2image0channel7.nrrd --deformPoints

python src/eval/LandmarkHandler.py --path0=resources/DirLab4DEval/Case3/00deformeddeformeddeformed.pts --path1=results/DirLab4DEval/deformationFieldDataset2image0channel6.nrrd --deformPoints

python src/eval/LandmarkHandler.py --path0=resources/DirLab4DEval/Case3/00deformeddeformeddeformeddeformed.pts --path1=results/DirLab4DEval/deformationFieldDataset2image0channel5.nrrd --deformPoints
python src/eval/LandmarkHandler.py --path0=resources/DirLab4DEval/Case3/50.pts --path1=resources/DirLab4DEval/Case3/00deformeddeformeddeformeddeformeddeformed.pts --calcDiff --calcDiff > distances.txt

python src/eval/LandmarkHandler.py --path0=resources/DirLab4DEval/Case3/00deformeddeformeddeformeddeformeddeformed.pts --path1=results/DirLab4DEval/deformationFieldDataset2image0channel4.nrrd --deformPoints
python src/eval/LandmarkHandler.py --path0=resources/DirLab4DEval/Case3/40.pts --path1=resources/DirLab4DEval/Case3/00deformeddeformeddeformeddeformeddeformeddeformed.pts --calcDiff --calcDiff > distances.txt

python src/eval/LandmarkHandler.py --path0=resources/DirLab4DEval/Case3/00deformeddeformeddeformeddeformeddeformeddeformed.pts --path1=results/DirLab4DEval/deformationFieldDataset2image0channel3.nrrd --deformPoints
python src/eval/LandmarkHandler.py --path0=resources/DirLab4DEval/Case3/30.pts --path1=resources/DirLab4DEval/Case3/00deformeddeformeddeformeddeformeddeformeddeformeddeformed.pts --calcDiff --calcDiff > distances.txt

python src/eval/LandmarkHandler.py --path0=resources/DirLab4DEval/Case3/00deformeddeformeddeformeddeformeddeformeddeformeddeformed.pts --path1=results/DirLab4DEval/deformationFieldDataset2image0channel2.nrrd --deformPoints
python src/eval/LandmarkHandler.py --path0=resources/DirLab4DEval/Case3/20.pts --path1=resources/DirLab4DEval/Case3/00deformeddeformeddeformeddeformeddeformeddeformeddeformeddeformed.pts --calcDiff > distances.txt

python src/eval/LandmarkHandler.py --path0=resources/DirLab4DEval/Case3/00deformeddeformeddeformeddeformeddeformeddeformeddeformeddeformed.pts --path1=results/DirLab4DEval/deformationFieldDataset2image0channel1.nrrd --deformPoints
python src/eval/LandmarkHandler.py --path0=resources/DirLab4DEval/Case3/10.pts --path1=resources/DirLab4DEval/Case3/00deformeddeformeddeformeddeformeddeformeddeformeddeformeddeformeddeformed.pts --calcDiff --calcDiff > distances.txt

python src/eval/LandmarkHandler.py --path0=resources/DirLab4DEval/Case3/00deformeddeformeddeformeddeformeddeformeddeformeddeformeddeformeddeformed.pts --path1=results/DirLab4DEval/deformationFieldDataset2image0channel0.nrrd --deformPoints
python src/eval/LandmarkHandler.py --path0=resources/DirLab4DEval/Case3/00.pts --path1=resources/DirLab4DEval/Case3/00deformeddeformeddeformeddeformeddeformeddeformeddeformeddeformeddeformeddeformed.pts --calcDiff --calcDiff > distances.txt

echo "Dataset 3"

python src/eval/LandmarkHandler.py --path0=resources/DirLab4DEval/Case4/00.pts --path1=results/DirLab4DEval/deformationFieldDataset3image0channel-1.nrrd --deformPoints

python src/eval/LandmarkHandler.py --path0=resources/DirLab4DEval/Case4/00deformed.pts --path1=results/DirLab4DEval/deformationFieldDataset3image0channel8.nrrd --deformPoints

python src/eval/LandmarkHandler.py --path0=resources/DirLab4DEval/Case4/00deformeddeformed.pts --path1=results/DirLab4DEval/deformationFieldDataset3image0channel7.nrrd --deformPoints

python src/eval/LandmarkHandler.py --path0=resources/DirLab4DEval/Case4/00deformeddeformeddeformed.pts --path1=results/DirLab4DEval/deformationFieldDataset3image0channel6.nrrd --deformPoints

python src/eval/LandmarkHandler.py --path0=resources/DirLab4DEval/Case4/00deformeddeformeddeformeddeformed.pts --path1=results/DirLab4DEval/deformationFieldDataset3image0channel5.nrrd --deformPoints
python src/eval/LandmarkHandler.py --path0=resources/DirLab4DEval/Case4/50.pts --path1=resources/DirLab4DEval/Case4/00deformeddeformeddeformeddeformeddeformed.pts --calcDiff --calcDiff > distances.txt

python src/eval/LandmarkHandler.py --path0=resources/DirLab4DEval/Case4/00deformeddeformeddeformeddeformeddeformed.pts --path1=results/DirLab4DEval/deformationFieldDataset3image0channel4.nrrd --deformPoints
python src/eval/LandmarkHandler.py --path0=resources/DirLab4DEval/Case4/40.pts --path1=resources/DirLab4DEval/Case4/00deformeddeformeddeformeddeformeddeformeddeformed.pts --calcDiff --calcDiff > distances.txt

python src/eval/LandmarkHandler.py --path0=resources/DirLab4DEval/Case4/00deformeddeformeddeformeddeformeddeformeddeformed.pts --path1=results/DirLab4DEval/deformationFieldDataset3image0channel3.nrrd --deformPoints
python src/eval/LandmarkHandler.py --path0=resources/DirLab4DEval/Case4/30.pts --path1=resources/DirLab4DEval/Case4/00deformeddeformeddeformeddeformeddeformeddeformeddeformed.pts --calcDiff --calcDiff > distances.txt

python src/eval/LandmarkHandler.py --path0=resources/DirLab4DEval/Case4/00deformeddeformeddeformeddeformeddeformeddeformeddeformed.pts --path1=results/DirLab4DEval/deformationFieldDataset3image0channel2.nrrd --deformPoints
python src/eval/LandmarkHandler.py --path0=resources/DirLab4DEval/Case4/20.pts --path1=resources/DirLab4DEval/Case4/00deformeddeformeddeformeddeformeddeformeddeformeddeformeddeformed.pts --calcDiff > distances.txt

python src/eval/LandmarkHandler.py --path0=resources/DirLab4DEval/Case4/00deformeddeformeddeformeddeformeddeformeddeformeddeformeddeformed.pts --path1=results/DirLab4DEval/deformationFieldDataset3image0channel1.nrrd --deformPoints
python src/eval/LandmarkHandler.py --path0=resources/DirLab4DEval/Case4/10.pts --path1=resources/DirLab4DEval/Case4/00deformeddeformeddeformeddeformeddeformeddeformeddeformeddeformeddeformed.pts --calcDiff --calcDiff > distances.txt

python src/eval/LandmarkHandler.py --path0=resources/DirLab4DEval/Case4/00deformeddeformeddeformeddeformeddeformeddeformeddeformeddeformeddeformed.pts --path1=results/DirLab4DEval/deformationFieldDataset3image0channel0.nrrd --deformPoints
python src/eval/LandmarkHandler.py --path0=resources/DirLab4DEval/Case4/00.pts --path1=resources/DirLab4DEval/Case4/00deformeddeformeddeformeddeformeddeformeddeformeddeformeddeformeddeformeddeformed.pts --calcDiff --calcDiff > distances.txt

echo "Dataset 4"

python src/eval/LandmarkHandler.py --path0=resources/DirLab4DEval/Case5/00.pts --path1=results/DirLab4DEval/deformationFieldDataset4image0channel-1.nrrd --deformPoints

python src/eval/LandmarkHandler.py --path0=resources/DirLab4DEval/Case5/00deformed.pts --path1=results/DirLab4DEval/deformationFieldDataset4image0channel8.nrrd --deformPoints

python src/eval/LandmarkHandler.py --path0=resources/DirLab4DEval/Case5/00deformeddeformed.pts --path1=results/DirLab4DEval/deformationFieldDataset4image0channel7.nrrd --deformPoints

python src/eval/LandmarkHandler.py --path0=resources/DirLab4DEval/Case5/00deformeddeformeddeformed.pts --path1=results/DirLab4DEval/deformationFieldDataset4image0channel6.nrrd --deformPoints

python src/eval/LandmarkHandler.py --path0=resources/DirLab4DEval/Case5/00deformeddeformeddeformeddeformed.pts --path1=results/DirLab4DEval/deformationFieldDataset4image0channel5.nrrd --deformPoints
python src/eval/LandmarkHandler.py --path0=resources/DirLab4DEval/Case5/50.pts --path1=resources/DirLab4DEval/Case5/00deformeddeformeddeformeddeformeddeformed.pts --calcDiff --calcDiff > distances.txt

python src/eval/LandmarkHandler.py --path0=resources/DirLab4DEval/Case5/00deformeddeformeddeformeddeformeddeformed.pts --path1=results/DirLab4DEval/deformationFieldDataset4image0channel4.nrrd --deformPoints
python src/eval/LandmarkHandler.py --path0=resources/DirLab4DEval/Case5/40.pts --path1=resources/DirLab4DEval/Case5/00deformeddeformeddeformeddeformeddeformeddeformed.pts --calcDiff --calcDiff > distances.txt

python src/eval/LandmarkHandler.py --path0=resources/DirLab4DEval/Case5/00deformeddeformeddeformeddeformeddeformeddeformed.pts --path1=results/DirLab4DEval/deformationFieldDataset4image0channel3.nrrd --deformPoints
python src/eval/LandmarkHandler.py --path0=resources/DirLab4DEval/Case5/30.pts --path1=resources/DirLab4DEval/Case5/00deformeddeformeddeformeddeformeddeformeddeformeddeformed.pts --calcDiff --calcDiff > distances.txt

python src/eval/LandmarkHandler.py --path0=resources/DirLab4DEval/Case5/00deformeddeformeddeformeddeformeddeformeddeformeddeformed.pts --path1=results/DirLab4DEval/deformationFieldDataset4image0channel2.nrrd --deformPoints
python src/eval/LandmarkHandler.py --path0=resources/DirLab4DEval/Case5/20.pts --path1=resources/DirLab4DEval/Case5/00deformeddeformeddeformeddeformeddeformeddeformeddeformeddeformed.pts --calcDiff > distances.txt

python src/eval/LandmarkHandler.py --path0=resources/DirLab4DEval/Case5/00deformeddeformeddeformeddeformeddeformeddeformeddeformeddeformed.pts --path1=results/DirLab4DEval/deformationFieldDataset4image0channel1.nrrd --deformPoints
python src/eval/LandmarkHandler.py --path0=resources/DirLab4DEval/Case5/10.pts --path1=resources/DirLab4DEval/Case5/00deformeddeformeddeformeddeformeddeformeddeformeddeformeddeformeddeformed.pts --calcDiff --calcDiff > distances.txt

python src/eval/LandmarkHandler.py --path0=resources/DirLab4DEval/Case5/00deformeddeformeddeformeddeformeddeformeddeformeddeformeddeformeddeformed.pts --path1=results/DirLab4DEval/deformationFieldDataset4image0channel0.nrrd --deformPoints
python src/eval/LandmarkHandler.py --path0=resources/DirLab4DEval/Case5/00.pts --path1=resources/DirLab4DEval/Case5/00deformeddeformeddeformeddeformeddeformeddeformeddeformeddeformeddeformeddeformed.pts --calcDiff --calcDiff > distances.txt

echo "Dataset 5"

python src/eval/LandmarkHandler.py --path0=resources/DirLab4DEval/Case6/00.pts --path1=results/DirLab4DEval/deformationFieldDataset5image0channel-1.nrrd --deformPoints

python src/eval/LandmarkHandler.py --path0=resources/DirLab4DEval/Case6/00deformed.pts --path1=results/DirLab4DEval/deformationFieldDataset5image0channel8.nrrd --deformPoints

python src/eval/LandmarkHandler.py --path0=resources/DirLab4DEval/Case6/00deformeddeformed.pts --path1=results/DirLab4DEval/deformationFieldDataset5image0channel7.nrrd --deformPoints

python src/eval/LandmarkHandler.py --path0=resources/DirLab4DEval/Case6/00deformeddeformeddeformed.pts --path1=results/DirLab4DEval/deformationFieldDataset5image0channel6.nrrd --deformPoints

python src/eval/LandmarkHandler.py --path0=resources/DirLab4DEval/Case6/00deformeddeformeddeformeddeformed.pts --path1=results/DirLab4DEval/deformationFieldDataset5image0channel5.nrrd --deformPoints
python src/eval/LandmarkHandler.py --path0=resources/DirLab4DEval/Case6/50.pts --path1=resources/DirLab4DEval/Case6/00deformeddeformeddeformeddeformeddeformed.pts --calcDiff --calcDiff > distances.txt

python src/eval/LandmarkHandler.py --path0=resources/DirLab4DEval/Case6/00deformeddeformeddeformeddeformeddeformed.pts --path1=results/DirLab4DEval/deformationFieldDataset5image0channel4.nrrd --deformPoints
python src/eval/LandmarkHandler.py --path0=resources/DirLab4DEval/Case6/40.pts --path1=resources/DirLab4DEval/Case6/00deformeddeformeddeformeddeformeddeformeddeformed.pts --calcDiff --calcDiff > distances.txt

python src/eval/LandmarkHandler.py --path0=resources/DirLab4DEval/Case6/00deformeddeformeddeformeddeformeddeformeddeformed.pts --path1=results/DirLab4DEval/deformationFieldDataset5image0channel3.nrrd --deformPoints
python src/eval/LandmarkHandler.py --path0=resources/DirLab4DEval/Case6/30.pts --path1=resources/DirLab4DEval/Case6/00deformeddeformeddeformeddeformeddeformeddeformeddeformed.pts --calcDiff --calcDiff > distances.txt

python src/eval/LandmarkHandler.py --path0=resources/DirLab4DEval/Case6/00deformeddeformeddeformeddeformeddeformeddeformeddeformed.pts --path1=results/DirLab4DEval/deformationFieldDataset5image0channel2.nrrd --deformPoints
python src/eval/LandmarkHandler.py --path0=resources/DirLab4DEval/Case6/20.pts --path1=resources/DirLab4DEval/Case6/00deformeddeformeddeformeddeformeddeformeddeformeddeformeddeformed.pts --calcDiff > distances.txt

python src/eval/LandmarkHandler.py --path0=resources/DirLab4DEval/Case6/00deformeddeformeddeformeddeformeddeformeddeformeddeformeddeformed.pts --path1=results/DirLab4DEval/deformationFieldDataset5image0channel1.nrrd --deformPoints
python src/eval/LandmarkHandler.py --path0=resources/DirLab4DEval/Case6/10.pts --path1=resources/DirLab4DEval/Case6/00deformeddeformeddeformeddeformeddeformeddeformeddeformeddeformeddeformed.pts --calcDiff --calcDiff > distances.txt

python src/eval/LandmarkHandler.py --path0=resources/DirLab4DEval/Case6/00deformeddeformeddeformeddeformeddeformeddeformeddeformeddeformeddeformed.pts --path1=results/DirLab4DEval/deformationFieldDataset5image0channel0.nrrd --deformPoints
python src/eval/LandmarkHandler.py --path0=resources/DirLab4DEval/Case6/00.pts --path1=resources/DirLab4DEval/Case6/00deformeddeformeddeformeddeformeddeformeddeformeddeformeddeformeddeformeddeformed.pts --calcDiff --calcDiff > distances.txt

echo "Dataset 6"

python src/eval/LandmarkHandler.py --path0=resources/DirLab4DEval/Case7/00.pts --path1=results/DirLab4DEval/deformationFieldDataset6image0channel-1.nrrd --deformPoints

python src/eval/LandmarkHandler.py --path0=resources/DirLab4DEval/Case7/00deformed.pts --path1=results/DirLab4DEval/deformationFieldDataset6image0channel8.nrrd --deformPoints

python src/eval/LandmarkHandler.py --path0=resources/DirLab4DEval/Case7/00deformeddeformed.pts --path1=results/DirLab4DEval/deformationFieldDataset6image0channel7.nrrd --deformPoints

python src/eval/LandmarkHandler.py --path0=resources/DirLab4DEval/Case7/00deformeddeformeddeformed.pts --path1=results/DirLab4DEval/deformationFieldDataset6image0channel6.nrrd --deformPoints

python src/eval/LandmarkHandler.py --path0=resources/DirLab4DEval/Case7/00deformeddeformeddeformeddeformed.pts --path1=results/DirLab4DEval/deformationFieldDataset6image0channel5.nrrd --deformPoints
python src/eval/LandmarkHandler.py --path0=resources/DirLab4DEval/Case7/50.pts --path1=resources/DirLab4DEval/Case7/00deformeddeformeddeformeddeformeddeformed.pts --calcDiff --calcDiff > distances.txt

python src/eval/LandmarkHandler.py --path0=resources/DirLab4DEval/Case7/00deformeddeformeddeformeddeformeddeformed.pts --path1=results/DirLab4DEval/deformationFieldDataset6image0channel4.nrrd --deformPoints
python src/eval/LandmarkHandler.py --path0=resources/DirLab4DEval/Case7/40.pts --path1=resources/DirLab4DEval/Case7/00deformeddeformeddeformeddeformeddeformeddeformed.pts --calcDiff --calcDiff > distances.txt

python src/eval/LandmarkHandler.py --path0=resources/DirLab4DEval/Case7/00deformeddeformeddeformeddeformeddeformeddeformed.pts --path1=results/DirLab4DEval/deformationFieldDataset6image0channel3.nrrd --deformPoints
python src/eval/LandmarkHandler.py --path0=resources/DirLab4DEval/Case7/30.pts --path1=resources/DirLab4DEval/Case7/00deformeddeformeddeformeddeformeddeformeddeformeddeformed.pts --calcDiff --calcDiff > distances.txt

python src/eval/LandmarkHandler.py --path0=resources/DirLab4DEval/Case7/00deformeddeformeddeformeddeformeddeformeddeformeddeformed.pts --path1=results/DirLab4DEval/deformationFieldDataset6image0channel2.nrrd --deformPoints
python src/eval/LandmarkHandler.py --path0=resources/DirLab4DEval/Case7/20.pts --path1=resources/DirLab4DEval/Case7/00deformeddeformeddeformeddeformeddeformeddeformeddeformeddeformed.pts --calcDiff > distances.txt

python src/eval/LandmarkHandler.py --path0=resources/DirLab4DEval/Case7/00deformeddeformeddeformeddeformeddeformeddeformeddeformeddeformed.pts --path1=results/DirLab4DEval/deformationFieldDataset6image0channel1.nrrd --deformPoints
python src/eval/LandmarkHandler.py --path0=resources/DirLab4DEval/Case7/10.pts --path1=resources/DirLab4DEval/Case7/00deformeddeformeddeformeddeformeddeformeddeformeddeformeddeformeddeformed.pts --calcDiff --calcDiff > distances.txt

python src/eval/LandmarkHandler.py --path0=resources/DirLab4DEval/Case7/00deformeddeformeddeformeddeformeddeformeddeformeddeformeddeformeddeformed.pts --path1=results/DirLab4DEval/deformationFieldDataset6image0channel0.nrrd --deformPoints
python src/eval/LandmarkHandler.py --path0=resources/DirLab4DEval/Case7/00.pts --path1=resources/DirLab4DEval/Case7/00deformeddeformeddeformeddeformeddeformeddeformeddeformeddeformeddeformeddeformed.pts --calcDiff --calcDiff > distances.txt

echo "Dataset 7"

python src/eval/LandmarkHandler.py --path0=resources/DirLab4DEval/Case8/00.pts --path1=results/DirLab4DEval/deformationFieldDataset7image0channel-1.nrrd --deformPoints

python src/eval/LandmarkHandler.py --path0=resources/DirLab4DEval/Case8/00deformed.pts --path1=results/DirLab4DEval/deformationFieldDataset7image0channel8.nrrd --deformPoints

python src/eval/LandmarkHandler.py --path0=resources/DirLab4DEval/Case8/00deformeddeformed.pts --path1=results/DirLab4DEval/deformationFieldDataset7image0channel7.nrrd --deformPoints

python src/eval/LandmarkHandler.py --path0=resources/DirLab4DEval/Case8/00deformeddeformeddeformed.pts --path1=results/DirLab4DEval/deformationFieldDataset7image0channel6.nrrd --deformPoints

python src/eval/LandmarkHandler.py --path0=resources/DirLab4DEval/Case8/00deformeddeformeddeformeddeformed.pts --path1=results/DirLab4DEval/deformationFieldDataset7image0channel5.nrrd --deformPoints
python src/eval/LandmarkHandler.py --path0=resources/DirLab4DEval/Case8/50.pts --path1=resources/DirLab4DEval/Case8/00deformeddeformeddeformeddeformeddeformed.pts --calcDiff --calcDiff > distances.txt

python src/eval/LandmarkHandler.py --path0=resources/DirLab4DEval/Case8/00deformeddeformeddeformeddeformeddeformed.pts --path1=results/DirLab4DEval/deformationFieldDataset7image0channel4.nrrd --deformPoints
python src/eval/LandmarkHandler.py --path0=resources/DirLab4DEval/Case8/40.pts --path1=resources/DirLab4DEval/Case8/00deformeddeformeddeformeddeformeddeformeddeformed.pts --calcDiff --calcDiff > distances.txt

python src/eval/LandmarkHandler.py --path0=resources/DirLab4DEval/Case8/00deformeddeformeddeformeddeformeddeformeddeformed.pts --path1=results/DirLab4DEval/deformationFieldDataset7image0channel3.nrrd --deformPoints
python src/eval/LandmarkHandler.py --path0=resources/DirLab4DEval/Case8/30.pts --path1=resources/DirLab4DEval/Case8/00deformeddeformeddeformeddeformeddeformeddeformeddeformed.pts --calcDiff --calcDiff > distances.txt

python src/eval/LandmarkHandler.py --path0=resources/DirLab4DEval/Case8/00deformeddeformeddeformeddeformeddeformeddeformeddeformed.pts --path1=results/DirLab4DEval/deformationFieldDataset7image0channel2.nrrd --deformPoints
python src/eval/LandmarkHandler.py --path0=resources/DirLab4DEval/Case8/20.pts --path1=resources/DirLab4DEval/Case8/00deformeddeformeddeformeddeformeddeformeddeformeddeformeddeformed.pts --calcDiff > distances.txt

python src/eval/LandmarkHandler.py --path0=resources/DirLab4DEval/Case8/00deformeddeformeddeformeddeformeddeformeddeformeddeformeddeformed.pts --path1=results/DirLab4DEval/deformationFieldDataset7image0channel1.nrrd --deformPoints
python src/eval/LandmarkHandler.py --path0=resources/DirLab4DEval/Case8/10.pts --path1=resources/DirLab4DEval/Case8/00deformeddeformeddeformeddeformeddeformeddeformeddeformeddeformeddeformed.pts --calcDiff --calcDiff > distances.txt

python src/eval/LandmarkHandler.py --path0=resources/DirLab4DEval/Case8/00deformeddeformeddeformeddeformeddeformeddeformeddeformeddeformeddeformed.pts --path1=results/DirLab4DEval/deformationFieldDataset7image0channel0.nrrd --deformPoints
python src/eval/LandmarkHandler.py --path0=resources/DirLab4DEval/Case8/00.pts --path1=resources/DirLab4DEval/Case8/00deformeddeformeddeformeddeformeddeformeddeformeddeformeddeformeddeformeddeformed.pts --calcDiff --calcDiff > distances.txt

echo "Dataset 8"

python src/eval/LandmarkHandler.py --path0=resources/DirLab4DEval/Case9/00.pts --path1=results/DirLab4DEval/deformationFieldDataset8image0channel-1.nrrd --deformPoints

python src/eval/LandmarkHandler.py --path0=resources/DirLab4DEval/Case9/00deformed.pts --path1=results/DirLab4DEval/deformationFieldDataset8image0channel8.nrrd --deformPoints

python src/eval/LandmarkHandler.py --path0=resources/DirLab4DEval/Case9/00deformeddeformed.pts --path1=results/DirLab4DEval/deformationFieldDataset8image0channel7.nrrd --deformPoints

python src/eval/LandmarkHandler.py --path0=resources/DirLab4DEval/Case9/00deformeddeformeddeformed.pts --path1=results/DirLab4DEval/deformationFieldDataset8image0channel6.nrrd --deformPoints

python src/eval/LandmarkHandler.py --path0=resources/DirLab4DEval/Case9/00deformeddeformeddeformeddeformed.pts --path1=results/DirLab4DEval/deformationFieldDataset8image0channel5.nrrd --deformPoints
python src/eval/LandmarkHandler.py --path0=resources/DirLab4DEval/Case9/50.pts --path1=resources/DirLab4DEval/Case9/00deformeddeformeddeformeddeformeddeformed.pts --calcDiff --calcDiff > distances.txt

python src/eval/LandmarkHandler.py --path0=resources/DirLab4DEval/Case9/00deformeddeformeddeformeddeformeddeformed.pts --path1=results/DirLab4DEval/deformationFieldDataset8image0channel4.nrrd --deformPoints
python src/eval/LandmarkHandler.py --path0=resources/DirLab4DEval/Case9/40.pts --path1=resources/DirLab4DEval/Case9/00deformeddeformeddeformeddeformeddeformeddeformed.pts --calcDiff --calcDiff > distances.txt

python src/eval/LandmarkHandler.py --path0=resources/DirLab4DEval/Case9/00deformeddeformeddeformeddeformeddeformeddeformed.pts --path1=results/DirLab4DEval/deformationFieldDataset8image0channel3.nrrd --deformPoints
python src/eval/LandmarkHandler.py --path0=resources/DirLab4DEval/Case9/30.pts --path1=resources/DirLab4DEval/Case9/00deformeddeformeddeformeddeformeddeformeddeformeddeformed.pts --calcDiff --calcDiff > distances.txt

python src/eval/LandmarkHandler.py --path0=resources/DirLab4DEval/Case9/00deformeddeformeddeformeddeformeddeformeddeformeddeformed.pts --path1=results/DirLab4DEval/deformationFieldDataset8image0channel2.nrrd --deformPoints
python src/eval/LandmarkHandler.py --path0=resources/DirLab4DEval/Case9/20.pts --path1=resources/DirLab4DEval/Case9/00deformeddeformeddeformeddeformeddeformeddeformeddeformeddeformed.pts --calcDiff > distances.txt

python src/eval/LandmarkHandler.py --path0=resources/DirLab4DEval/Case9/00deformeddeformeddeformeddeformeddeformeddeformeddeformeddeformed.pts --path1=results/DirLab4DEval/deformationFieldDataset8image0channel1.nrrd --deformPoints
python src/eval/LandmarkHandler.py --path0=resources/DirLab4DEval/Case9/10.pts --path1=resources/DirLab4DEval/Case9/00deformeddeformeddeformeddeformeddeformeddeformeddeformeddeformeddeformed.pts --calcDiff --calcDiff > distances.txt

python src/eval/LandmarkHandler.py --path0=resources/DirLab4DEval/Case9/00deformeddeformeddeformeddeformeddeformeddeformeddeformeddeformeddeformed.pts --path1=results/DirLab4DEval/deformationFieldDataset8image0channel0.nrrd --deformPoints
python src/eval/LandmarkHandler.py --path0=resources/DirLab4DEval/Case9/00.pts --path1=resources/DirLab4DEval/Case9/00deformeddeformeddeformeddeformeddeformeddeformeddeformeddeformeddeformeddeformed.pts --calcDiff --calcDiff > distances.txt

echo "Dataset 9"

python src/eval/LandmarkHandler.py --path0=resources/DirLab4DEval/Case10/00.pts --path1=results/DirLab4DEval/deformationFieldDataset9image0channel-1.nrrd --deformPoints

python src/eval/LandmarkHandler.py --path0=resources/DirLab4DEval/Case10/00deformed.pts --path1=results/DirLab4DEval/deformationFieldDataset9image0channel8.nrrd --deformPoints

python src/eval/LandmarkHandler.py --path0=resources/DirLab4DEval/Case10/00deformeddeformed.pts --path1=results/DirLab4DEval/deformationFieldDataset9image0channel7.nrrd --deformPoints

python src/eval/LandmarkHandler.py --path0=resources/DirLab4DEval/Case10/00deformeddeformeddeformed.pts --path1=results/DirLab4DEval/deformationFieldDataset9image0channel6.nrrd --deformPoints

python src/eval/LandmarkHandler.py --path0=resources/DirLab4DEval/Case10/00deformeddeformeddeformeddeformed.pts --path1=results/DirLab4DEval/deformationFieldDataset9image0channel5.nrrd --deformPoints
python src/eval/LandmarkHandler.py --path0=resources/DirLab4DEval/Case10/50.pts --path1=resources/DirLab4DEval/Case10/00deformeddeformeddeformeddeformeddeformed.pts --calcDiff --calcDiff > distances.txt

python src/eval/LandmarkHandler.py --path0=resources/DirLab4DEval/Case10/00deformeddeformeddeformeddeformeddeformed.pts --path1=results/DirLab4DEval/deformationFieldDataset9image0channel4.nrrd --deformPoints
python src/eval/LandmarkHandler.py --path0=resources/DirLab4DEval/Case10/40.pts --path1=resources/DirLab4DEval/Case10/00deformeddeformeddeformeddeformeddeformeddeformed.pts --calcDiff --calcDiff > distances.txt

python src/eval/LandmarkHandler.py --path0=resources/DirLab4DEval/Case10/00deformeddeformeddeformeddeformeddeformeddeformed.pts --path1=results/DirLab4DEval/deformationFieldDataset9image0channel3.nrrd --deformPoints
python src/eval/LandmarkHandler.py --path0=resources/DirLab4DEval/Case10/30.pts --path1=resources/DirLab4DEval/Case10/00deformeddeformeddeformeddeformeddeformeddeformeddeformed.pts --calcDiff --calcDiff > distances.txt

python src/eval/LandmarkHandler.py --path0=resources/DirLab4DEval/Case10/00deformeddeformeddeformeddeformeddeformeddeformeddeformed.pts --path1=results/DirLab4DEval/deformationFieldDataset9image0channel2.nrrd --deformPoints
python src/eval/LandmarkHandler.py --path0=resources/DirLab4DEval/Case10/20.pts --path1=resources/DirLab4DEval/Case10/00deformeddeformeddeformeddeformeddeformeddeformeddeformeddeformed.pts --calcDiff > distances.txt

python src/eval/LandmarkHandler.py --path0=resources/DirLab4DEval/Case10/00deformeddeformeddeformeddeformeddeformeddeformeddeformeddeformed.pts --path1=results/DirLab4DEval/deformationFieldDataset9image0channel1.nrrd --deformPoints
python src/eval/LandmarkHandler.py --path0=resources/DirLab4DEval/Case10/10.pts --path1=resources/DirLab4DEval/Case10/00deformeddeformeddeformeddeformeddeformeddeformeddeformeddeformeddeformed.pts --calcDiff --calcDiff > distances.txt

python src/eval/LandmarkHandler.py --path0=resources/DirLab4DEval/Case10/00deformeddeformeddeformeddeformeddeformeddeformeddeformeddeformeddeformed.pts --path1=results/DirLab4DEval/deformationFieldDataset9image0channel0.nrrd --deformPoints
python src/eval/LandmarkHandler.py --path0=resources/DirLab4DEval/Case10/00.pts --path1=resources/DirLab4DEval/Case10/00deformeddeformeddeformeddeformeddeformeddeformeddeformeddeformeddeformeddeformed.pts --calcDiff --calcDiff > distances.txt
