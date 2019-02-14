import torch


numberOfEpochs = 50
testMode = True
trainMode = True
oneShot = True
usePaddedNet=False
ccW=0.999
smoothW = 0.001
vecLengthW = 0.0
cycleW = 0.00
trainingFileNamesCSV=''
device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
outputPath='.'
patchSize=128
maxNumberOfSamples=6 # samples for one batch must be < maxNumberOfSamples
netDepth=3
receptiveField = (44, 44, 44) #adapt depth and receptive field according to ReceptiveFieldSizeCalculator in repository
netMinPatchSize = 44
normImgPatches=False
trainTillConvergence = True
lossTollerance=0.00001



