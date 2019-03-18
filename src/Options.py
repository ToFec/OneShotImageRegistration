import torch
test = True

numberOfEpochs = 500
testMode = True
trainMode = True
oneShot = True
usePaddedNet=True

vecLengthW = 0.0
cycleW = 0.00
trainingFileNamesCSV=''
device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
outputPath='.'
patchSize=80#152
maxNumberOfSamples=6 # samples for one batch must be < maxNumberOfSamples
netDepth=3
numberOfFiltersFirstLayer=32
receptiveField = (44, 44, 44) #adapt depth and receptive field according to ReceptiveFieldSizeCalculator in repository
netMinPatchSize = 48
netMinPatchSizePadded = 4
normImgPatches=False
trainTillConvergence = True

cumulativeLossTollerance=100#0.001
if test:
  ccW=1.0
  downSampleRates = (0.25,0.5,1.0)
  boundarySmoothnessW=(0.0, 10.0, 0.0, 10.0, 0.0, 10.0)
  smoothW = (0.001,0.001,0.001,0.001,0.001,0.001)
  lossTollerances=(0.00001,0.00001)
else:
  ccW=1.0
  downSampleRates = (0.25,0.5,1.0)
  boundarySmoothnessW=(1.0,1.0,1.0)
  smoothW = (0.001,0.001,0.001)
  lossTollerances=(0.001,0.00001)
useMedianForSampling = (False,False,True)



