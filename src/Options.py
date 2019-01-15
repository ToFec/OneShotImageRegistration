import torch


numberOfEpochs = 500
testMode = False
trainMode = False
overFit = False
ccW=0.79
smoothW = 0.1
vecLengthW = 0.01
cycleW = 0.1
trainingFileNamesCSV=''
device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
outputPath=''
patchSize=64
maxNumberOfSamples=6 # samples for one batch must be < maxNumberOfSamples
