#!/bin/bash

python src/OnePatchShot.py --trainingFiles=/home/fechter/workspace/TorchSandbox/resources/DirLab3DEval.csv --outputPath=results/layer12/DirLab3DEval
python src/OnePatchShot.py --trainingFiles=/home/fechter/workspace/TorchSandbox/resources/Popi3DEval.csv --outputPath=results/layer12/Popi3DEval
python src/OnePatchShot.py --trainingFiles=/home/fechter/workspace/TorchSandbox/resources/sunnyBrook3D.csv --outputPath=results/layer12/SunnyBrook3D --maskOutZeros

python src/OnePatchShot.py --trainingFiles=/home/fechter/workspace/TorchSandbox/resources/Popi4DEval.csv  --outputPath=results/layer12/Popi4DEval
python src/OnePatchShot.py --trainingFiles=/home/fechter/workspace/TorchSandbox/resources/DirLab4DEval.csv --outputPath=results/layer12/DirLab4DEval
python src/OnePatchShot.py --trainingFiles=/home/fechter/workspace/TorchSandbox/resources/sunnyBrook4D.csv --outputPath=results/layer12/SunnyBrook4D --maskOutZeros
exit 0
