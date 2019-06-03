#!/bin/bash


echo "Dirlab Start" >> distances.csv

for i in {0..9}
do
./CycleConsistencyCheckDirLab2.sh 0 10 $i
./CycleConsistencyCheckDirLab2.sh 1 10 $i
./CycleConsistencyCheckDirLab2.sh 2 10 $i
./CycleConsistencyCheckDirLab2.sh 3 10 $i
./CycleConsistencyCheckDirLab2.sh 4 10 $i
./CycleConsistencyCheckDirLab2.sh 5 10 $i
done

exit 0


echo "Sunny Cycle Eval Start" >> dsc.csv

for i in {0..14}
do

for j in {0..19}
do
./CycleConsistencyCheckSunny.sh $j 20 $i
done

done

exit 0


echo "Popi Start" >> distances.csv
for i in {0..5}
do

for j in {0..9}
do
if [ ${i} -lt 3 ] || [ ${j} -eq 0 ] || [ ${j} -eq 5 ]
then
./CycleConsistencyCheckPopi2.sh $j 10 $i
fi
done

done
