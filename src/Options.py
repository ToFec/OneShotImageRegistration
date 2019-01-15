import torch


numberOfEpochs = 500
testMode = False
trainMode = False
overFit = False
ccW=1.0
smoothW = 0.0
vecLengthW = 0.0
cycleW = 0.0
trainingFileNamesCSV=''
device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
outputPath=''
patchSize=64
maxNumberOfSamples=6 # samples for one batch must be < maxNumberOfSamples
