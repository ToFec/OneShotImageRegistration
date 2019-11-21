import torch
useContext = True
debugMode=True
numberOfEpochs = 1000
numberOfEpochs = [1000,1000,1000]
#numberOfEpochs = [200,250,1000]#with training of 3D pairs
#numberOfEpochs = [150,110,500]#with random training data from 4D

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
ccCalcNN=True
maskOutZeros=False
patchSize=(80,80,80)
ccW=1.0
dscWeight=0.0
downSampleSteps = 2# a size of 2 means 3 iterations with the following downsampling factors: (0.25,0.5,1.0)
stoptAtSampleStep=3
boundarySmoothnessW=(0.0,0.1,0.1)

cycleW = 0.01
finalGaussKernelSize=7
finalGaussKernelStd=2
sasSteps=5
if diffeomorphicRegistration:
  smoothW = (0.0005,0.0005,0.0005)
  smoothVF = True
else:
  smoothVF = True
  smoothW = (0.0005,0.0005,0.0005)
lossTollerances=(0.00001,)
useMedianForSampling = (False,False,True)

logLandmarkDistance=True

# only for training the network
validationIntervall=10
# numberOfEpochs = [75,75,75]
addVectorFields=True
netMinPatchSize = patchSize[0]
fineTuneOldModel=(False, False, False) # works only in combination with "previousModels" parameter
randomSampling=(False,False,False)




