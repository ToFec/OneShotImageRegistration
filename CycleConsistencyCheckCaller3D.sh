#!/bin/bash

echo "Sunny Cycle Eval Start" >> dsc.csv

for i in {0..14}
do

for j in {0..1}
do
./CycleConsistencyCheckSunny3D.sh $j 2 $i
done

done

echo "Popi Start" >> distances.csv
for i in {0..5}
do

for j in {0..1}
do
./CycleConsistencyCheckPopi3D.sh $j 2 $i
done

done


echo "Dirlab Start" >> distances.csv

for i in {0..9}
do
./CycleConsistencyCheckDirLab3D.sh 0 2 $i
./CycleConsistencyCheckDirLab3D.sh 1 2 $i
done

exit 0
