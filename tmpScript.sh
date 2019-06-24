#!/bin/bash

python src/OnePatchShot.py --trainingFiles=/home/fechter/workspace/TorchSandbox/resources/DirLab3DEval.csv --outputPath=results/DirLab3DEval
python src/OnePatchShot.py --trainingFiles=/home/fechter/workspace/TorchSandbox/resources/Popi3DEval.csv --outputPath=results/Popi3DEval
python src/OnePatchShot.py --trainingFiles=/home/fechter/workspace/TorchSandbox/resources/sunnyBrook3D.csv --outputPath=results/SunnyBrook3D --maskOutZeros

python src/OnePatchShot.py --trainingFiles=/home/fechter/workspace/TorchSandbox/resources/Popi4DEval.csv  --outputPath=results/Popi4DEval
python src/OnePatchShot.py --trainingFiles=/home/fechter/workspace/TorchSandbox/resources/DirLab4DEval.csv --outputPath=results/DirLab4DEval
python src/OnePatchShot.py --trainingFiles=/home/fechter/workspace/TorchSandbox/resources/sunnyBrook4D.csv --outputPath=results/SunnyBrook4D --maskOutZeros
exit 0
