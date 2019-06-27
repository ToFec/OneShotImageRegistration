#!/bin/bash

python src/OnePatchShot.py --trainingFiles=/home/fechter/workspace/TorchSandbox/resources/DirLab3DEval.csv --outputPath=results/StopAfter2/DirLab3DEval
python src/OnePatchShot.py --trainingFiles=/home/fechter/workspace/TorchSandbox/resources/Popi3DEval.csv --outputPath=results/StopAfter2/Popi3DEval
python src/OnePatchShot.py --trainingFiles=/home/fechter/workspace/TorchSandbox/resources/sunnyBrook3D.csv --outputPath=results/StopAfter2/SunnyBrook3D --maskOutZeros

python src/OnePatchShot.py --trainingFiles=/home/fechter/workspace/TorchSandbox/resources/Popi4DEval.csv  --outputPath=results/StopAfter2/Popi4DEval
python src/OnePatchShot.py --trainingFiles=/home/fechter/workspace/TorchSandbox/resources/DirLab4DEval.csv --outputPath=results/StopAfter2/DirLab4DEval
python src/OnePatchShot.py --trainingFiles=/home/fechter/workspace/TorchSandbox/resources/sunnyBrook4D.csv --outputPath=results/StopAfter2/SunnyBrook4D --maskOutZeros
exit 0
