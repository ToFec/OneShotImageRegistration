python src/eval/LandmarkHandler.py --path0=resources/DirLab3DEval/Case1/10.pts --path1=results/DirLab3DEvalSW1/dataset0channel00deformed.pts --calcDiff > distances0.txt
python src/eval/LandmarkHandler.py --path0=resources/DirLab3DEval/Case1/00.pts --path1=results/DirLab3DEvalSW1/dataset0channel10deformed.pts --calcDiff > distances1.txt

python src/eval/LandmarkHandler.py --path0=resources/DirLab3DEval/Case2/10.pts --path1=results/DirLab3DEvalSW1/dataset1channel00deformed.pts --calcDiff >> distances0.txt
python src/eval/LandmarkHandler.py --path0=resources/DirLab3DEval/Case2/00.pts --path1=results/DirLab3DEvalSW1/dataset1channel10deformed.pts --calcDiff >> distances1.txt

python src/eval/LandmarkHandler.py --path0=resources/DirLab3DEval/Case3/10.pts --path1=results/DirLab3DEvalSW1/dataset2channel00deformed.pts --calcDiff >> distances0.txt
python src/eval/LandmarkHandler.py --path0=resources/DirLab3DEval/Case3/00.pts --path1=results/DirLab3DEvalSW1/dataset2channel10deformed.pts --calcDiff >> distances1.txt

python src/eval/LandmarkHandler.py --path0=resources/DirLab3DEval/Case4/10.pts --path1=results/DirLab3DEvalSW1/dataset3channel00deformed.pts --calcDiff >> distances0.txt
python src/eval/LandmarkHandler.py --path0=resources/DirLab3DEval/Case4/00.pts --path1=results/DirLab3DEvalSW1/dataset3channel10deformed.pts --calcDiff >> distances1.txt

python src/eval/LandmarkHandler.py --path0=resources/DirLab3DEval/Case5/10.pts --path1=results/DirLab3DEvalSW1/dataset4channel00deformed.pts --calcDiff >> distances0.txt
python src/eval/LandmarkHandler.py --path0=resources/DirLab3DEval/Case5/00.pts --path1=results/DirLab3DEvalSW1/dataset4channel10deformed.pts --calcDiff >> distances1.txt

python src/eval/LandmarkHandler.py --path0=resources/DirLab3DEval/Case6/10.pts --path1=results/DirLab3DEvalSW1/dataset5channel00deformed.pts --calcDiff >> distances0.txt
python src/eval/LandmarkHandler.py --path0=resources/DirLab3DEval/Case6/00.pts --path1=results/DirLab3DEvalSW1/dataset5channel10deformed.pts --calcDiff >> distances1.txt

python src/eval/LandmarkHandler.py --path0=resources/DirLab3DEval/Case7/10.pts --path1=results/DirLab3DEvalSW1/dataset6channel00deformed.pts --calcDiff >> distances0.txt
python src/eval/LandmarkHandler.py --path0=resources/DirLab3DEval/Case7/00.pts --path1=results/DirLab3DEvalSW1/dataset6channel10deformed.pts --calcDiff >> distances1.txt

python src/eval/LandmarkHandler.py --path0=resources/DirLab3DEval/Case8/10.pts --path1=results/DirLab3DEvalSW1/dataset7channel00deformed.pts --calcDiff >> distances0.txt
python src/eval/LandmarkHandler.py --path0=resources/DirLab3DEval/Case8/00.pts --path1=results/DirLab3DEvalSW1/dataset7channel10deformed.pts --calcDiff >> distances1.txt

python src/eval/LandmarkHandler.py --path0=resources/DirLab3DEval/Case9/10.pts --path1=results/DirLab3DEvalSW1/dataset8channel00deformed.pts --calcDiff >> distances0.txt
python src/eval/LandmarkHandler.py --path0=resources/DirLab3DEval/Case9/00.pts --path1=results/DirLab3DEvalSW1/dataset8channel10deformed.pts --calcDiff >> distances1.txt

python src/eval/LandmarkHandler.py --path0=resources/DirLab3DEval/Case10/10.pts --path1=results/DirLab3DEvalSW1/dataset9channel00deformed.pts --calcDiff >> distances0.txt
python src/eval/LandmarkHandler.py --path0=resources/DirLab3DEval/Case10/00.pts --path1=results/DirLab3DEvalSW1/dataset9channel10deformed.pts --calcDiff >> distances1.txt

exit 0

python src/eval/LandmarkHandler.py --path0=resources/DirLab3DEval/Case1/10.pts --path1=resources/DirLab3DEval/Case1/00.pts --calcDiff > distances.txt
#python src/eval/LandmarkHandler.py --path0=resources/DirLab3DEval/Case1/00.pts --path1=resources/DirLab3DEval/Case1/10.pts --calcDiff > distances.txt
echo "dataset01" >> distances.txt
python src/eval/LandmarkHandler.py --path0=resources/DirLab3DEval/Case2/10.pts --path1=resources/DirLab3DEval/Case2/00.pts --calcDiff >> distances.txt
#python src/eval/LandmarkHandler.py --path0=resources/DirLab3DEval/Case2/00.pts --path1=resources/DirLab3DEval/Case2/10.pts --calcDiff > distances.txt
echo "dataset02" >> distances.txt
python src/eval/LandmarkHandler.py --path0=resources/DirLab3DEval/Case3/10.pts --path1=resources/DirLab3DEval/Case3/00.pts --calcDiff >> distances.txt
#python src/eval/LandmarkHandler.py --path0=resources/DirLab3DEval/Case3/00.pts --path1=resources/DirLab3DEval/Case3/10.pts --calcDiff > distances.txt
echo "dataset03" >> distances.txt
python src/eval/LandmarkHandler.py --path0=resources/DirLab3DEval/Case4/10.pts --path1=resources/DirLab3DEval/Case4/00.pts --calcDiff >> distances.txt
#python src/eval/LandmarkHandler.py --path0=resources/DirLab3DEval/Case4/00.pts --path1=resources/DirLab3DEval/Case4/10.pts --calcDiff > distances.txt
echo "dataset04" >> distances.txt
python src/eval/LandmarkHandler.py --path0=resources/DirLab3DEval/Case5/10.pts --path1=resources/DirLab3DEval/Case5/00.pts --calcDiff >> distances.txt
#python src/eval/LandmarkHandler.py --path0=resources/DirLab3DEval/Case5/00.pts --path1=resources/DirLab3DEval/Case5/10.pts --calcDiff > distances.txt
echo "dataset05" >> distances.txt
python src/eval/LandmarkHandler.py --path0=resources/DirLab3DEval/Case6/10.pts --path1=resources/DirLab3DEval/Case6/00.pts --calcDiff >> distances.txt
#python src/eval/LandmarkHandler.py --path0=resources/DirLab3DEval/Case6/00.pts --path1=resources/DirLab3DEval/Case6/10.pts --calcDiff > distances.txt
echo "dataset06" >> distances.txt
python src/eval/LandmarkHandler.py --path0=resources/DirLab3DEval/Case7/10.pts --path1=resources/DirLab3DEval/Case7/00.pts --calcDiff >> distances.txt
#python src/eval/LandmarkHandler.py --path0=resources/DirLab3DEval/Case7/00.pts --path1=resources/DirLab3DEval/Case7/10.pts --calcDiff > distances.txt
echo "dataset07" >> distances.txt
python src/eval/LandmarkHandler.py --path0=resources/DirLab3DEval/Case8/10.pts --path1=resources/DirLab3DEval/Case8/00.pts --calcDiff >> distances.txt
#python src/eval/LandmarkHandler.py --path0=resources/DirLab3DEval/Case8/00.pts --path1=resources/DirLab3DEval/Case8/10.pts --calcDiff > distances.txt
echo "dataset08" >> distances.txt
python src/eval/LandmarkHandler.py --path0=resources/DirLab3DEval/Case9/10.pts --path1=resources/DirLab3DEval/Case9/00.pts --calcDiff >> distances.txt
#python src/eval/LandmarkHandler.py --path0=resources/DirLab3DEval/Case9/00.pts --path1=resources/DirLab3DEval/Case9/10.pts --calcDiff > distances.txt
echo "dataset09" >> distances.txt
python src/eval/LandmarkHandler.py --path0=resources/DirLab3DEval/Case10/10.pts --path1=resources/DirLab3DEval/Case10/00.pts --calcDiff >> distances.txt
#python src/eval/LandmarkHandler.py --path0=resources/DirLab3DEval/Case10/00.pts --path1=resources/DirLab3DEval/Case10/10.pts --calcDiff > distances.txt

exit 0
