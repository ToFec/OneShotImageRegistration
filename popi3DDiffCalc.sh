python src/eval/LandmarkHandler.py --path0=resources/Popi3DEval/Popi01/10.pts --path1=results/Popi3DEval/dataset0channel00deformed.pts --calcDiff > distances0.txt
python src/eval/LandmarkHandler.py --path0=resources/Popi3DEval/Popi01/00.pts --path1=results/Popi3DEval/dataset0channel10deformed.pts --calcDiff > distances1.txt

python src/eval/LandmarkHandler.py --path0=resources/Popi3DEval/Popi02/10.pts --path1=results/Popi3DEval/dataset1channel00deformed.pts --calcDiff > distances0.txt
python src/eval/LandmarkHandler.py --path0=resources/Popi3DEval/Popi02/00.pts --path1=results/Popi3DEval/dataset1channel10deformed.pts --calcDiff > distances1.txt

python src/eval/LandmarkHandler.py --path0=resources/Popi3DEval/Popi03/10.pts --path1=results/Popi3DEval/dataset2channel00deformed.pts --calcDiff > distances0.txt
python src/eval/LandmarkHandler.py --path0=resources/Popi3DEval/Popi03/00.pts --path1=results/Popi3DEval/dataset2channel10deformed.pts --calcDiff > distances1.txt

python src/eval/LandmarkHandler.py --path0=resources/Popi3DEval/Popi04/10.pts --path1=results/Popi3DEval/dataset3channel00deformed.pts --calcDiff > distances0.txt
python src/eval/LandmarkHandler.py --path0=resources/Popi3DEval/Popi04/00.pts --path1=results/Popi3DEval/dataset3channel10deformed.pts --calcDiff > distances1.txt

python src/eval/LandmarkHandler.py --path0=resources/Popi3DEval/Popi05/10.pts --path1=results/Popi3DEval/dataset4channel00deformed.pts --calcDiff > distances0.txt
python src/eval/LandmarkHandler.py --path0=resources/Popi3DEval/Popi05/00.pts --path1=results/Popi3DEval/dataset4channel10deformed.pts --calcDiff > distances1.txt

python src/eval/LandmarkHandler.py --path0=resources/Popi3DEval/Popi06/10.pts --path1=results/Popi3DEval/dataset5channel00deformed.pts --calcDiff > distances0.txt
python src/eval/LandmarkHandler.py --path0=resources/Popi3DEval/Popi06/00.pts --path1=results/Popi3DEval/dataset5channel10deformed.pts --calcDiff > distances1.txt

exit 0

python src/eval/LandmarkHandler.py --path0=resources/Popi3DEval/Popi01/10.pts --path1=resources/Popi3DEval/Popi01/00.pts --calcDiff > distances.txt
#python src/eval/LandmarkHandler.py --path0=resources/Popi3DEval/Popi01/00.pts --path1=resources/Popi3DEval/Popi01/10.pts --calcDiff > distances.txt
echo "dataset01" >> distances.txt
python src/eval/LandmarkHandler.py --path0=resources/Popi3DEval/Popi02/10.pts --path1=resources/Popi3DEval/Popi02/00.pts --calcDiff >> distances.txt
#python src/eval/LandmarkHandler.py --path0=resources/Popi3DEval/Popi02/00.pts --path1=resources/Popi3DEval/Popi02/10.pts --calcDiff > distances.txt
echo "dataset02" >> distances.txt
python src/eval/LandmarkHandler.py --path0=resources/Popi3DEval/Popi03/10.pts --path1=resources/Popi3DEval/Popi03/00.pts --calcDiff >> distances.txt
#python src/eval/LandmarkHandler.py --path0=resources/Popi3DEval/Popi03/00.pts --path1=resources/Popi3DEval/Popi03/10.pts --calcDiff > distances.txt
echo "dataset03" >> distances.txt
python src/eval/LandmarkHandler.py --path0=resources/Popi3DEval/Popi04/10.pts --path1=resources/Popi3DEval/Popi04/00.pts --calcDiff >> distances.txt
#python src/eval/LandmarkHandler.py --path0=resources/Popi3DEval/Popi04/00.pts --path1=resources/Popi3DEval/Popi04/10.pts --calcDiff > distances.txt
echo "dataset04" >> distances.txt
python src/eval/LandmarkHandler.py --path0=resources/Popi3DEval/Popi05/10.pts --path1=resources/Popi3DEval/Popi05/00.pts --calcDiff >> distances.txt
#python src/eval/LandmarkHandler.py --path0=resources/Popi3DEval/Popi05/00.pts --path1=resources/Popi3DEval/Popi05/10.pts --calcDiff > distances.txt
echo "dataset05" >> distances.txt
python src/eval/LandmarkHandler.py --path0=resources/Popi3DEval/Popi06/10.pts --path1=resources/Popi3DEval/Popi06/00.pts --calcDiff >> distances.txt
#python src/eval/LandmarkHandler.py --path0=resources/Popi3DEval/Popi06/00.pts --path1=resources/Popi3DEval/Popi06/10.pts --calcDiff > distances.txt


exit 0
