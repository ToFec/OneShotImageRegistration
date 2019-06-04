import torch
layers4 = True
useContext = True

numberOfEpochs = 5000
testMode = True
trainMode = True
oneShot = True
usePaddedNet=True

vecLengthW = 0.0
trainingFileNamesCSV=''
device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
outputPath='.'

maxNumberOfSamples=6 # samples for one batch must be < maxNumberOfSamples
netDepth=3
numberOfFiltersFirstLayer=32
receptiveField = (44, 44, 44) #adapt depth and receptive field according to ReceptiveFieldSizeCalculator in repository
netMinPatchSize = 48
netMinPatchSizePadded = 8
normImgPatches=False
trainTillConvergence = True
lossArraySize=50
ccCalcNN=True
maskOutZeros=False
cumulativeLossTollerance=100#0.001
if layers4:
  patchSize=(80,80,80,80)
  ccW=1.0
  downSampleSteps = 3# a size of 2 means 3 iterations with the following downsampling factors: (0.25,0.5,1.0)
  stoptAtSampleStep=4
  boundarySmoothnessW=(0.0,1.0,1.0,1.0)
  smoothW = (0.001,0.001,0.001,0.001)
  cycleW = 0.01
  lossTollerances=(0.00001,)
  useMedianForSampling = (False,False,False,True)
else:
  patchSize=(80,80,80)
  ccW=1.0
  downSampleSteps = 2# a size of 2 means 3 iterations with the following downsampling factors: (0.25,0.5,1.0)
  stoptAtSampleStep=3
  boundarySmoothnessW=(0.0,1.0,1.0)
  smoothW = (0.001,0.001,0.001)
  cycleW = 0.01
  lossTollerances=(0.00001,)
  useMedianForSampling = (False,False,True)




