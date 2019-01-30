import torch


numberOfEpochs = 50
testMode = True
trainMode = True
oneShot = True
ccW=0.9
smoothW = 0.1
vecLengthW = 1.00
cycleW = 0.00
trainingFileNamesCSV=''
device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
outputPath='.'
patchSize=32
maxNumberOfSamples=6 # samples for one batch must be < maxNumberOfSamples
netDepth=3
trainTillConvergence = True
lossTollerance=0.00000001
