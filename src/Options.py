import torch


numberOfEpochs = 100
testMode = True
trainMode = True
oneShot = True
ccW=0.9
smoothW = 0.05
vecLengthW = 0.00
cycleW = 0.05
trainingFileNamesCSV=''
device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
outputPath='.'
patchSize=64
maxNumberOfSamples=6 # samples for one batch must be < maxNumberOfSamples
netDepth=3
trainTillConvergence = True
lossTollerance=0.00000001
