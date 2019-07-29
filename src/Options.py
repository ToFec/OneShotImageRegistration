import torch
useContext = True
debugMode=True
numberOfEpochs = 1000
usePaddedNet=True

trainingFileNamesCSV=''
device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
outputPath='.'
diffeomorphicRegistration = True
netDepth=3
numberOfFiltersFirstLayer=32
netMinPatchSize = 48
netMinPatchSizePadded = 8
normImgPatches=False
trainTillConvergence = True
maskOutZeros=False
patchSize=(80,80,80)
ccW=1.0
dscWeight=0.0
downSampleSteps = 2# a size of 2 means 3 iterations with the following downsampling factors: (0.25,0.5,1.0)
stoptAtSampleStep=3
boundarySmoothnessW=(0.0,0.1,0.1)
cycleW = 0.01
if diffeomorphicRegistration:
  smoothW = (0.0001,0.0001,0.001)
else:
  smoothW = (0.0001,0.0001,0.0001)
lossTollerances=(0.00001,)
useMedianForSampling = (False,False,True)




