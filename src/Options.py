import torch


numberOfEpochs = 50
testMode = True
trainMode = True
oneShot = True
ccW=1.0
smoothW = 0.00
vecLengthW = 0.00
cycleW = 0.00
trainingFileNamesCSV=''
device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
outputPath='.'
patchSize=72
maxNumberOfSamples=6 # samples for one batch must be < maxNumberOfSamples
netDepth=4
trainTillConvergence = True
lossTollerance=0.00000001
