import torch
useContext = True

numberOfEpochs = [75,75,75]
#numberOfEpochs = [200,250,1000]#with training of 3D pairs
#numberOfEpochs = [150,110,500]#with random training data from 4D
usePaddedNet=True

trainingFileNamesCSV=''
device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
outputPath='.'

netDepth=3
numberOfFiltersFirstLayer=32
netMinPatchSize = 48
netMinPatchSizePadded = 8
normImgPatches=False
trainTillConvergence = True
ccCalcNN=True
maskOutZeros=False
#patchSize=(80,80,120) for the training we also tried a patch size of 120
patchSize=(80,80,80)
ccW=1.0
downSampleSteps = 2# a size of 2 means 3 iterations with the following downsampling factors: (0.25,0.5,1.0)
stoptAtSampleStep=2
boundarySmoothnessW=(0.0,0.1,0.1)
smoothW = (0.0001,0.0001,0.0001)
cycleW = 0.01
lossTollerances=(0.00001,)
useMedianForSampling = (False,False,True)

# only for training the network
validationIntervall=10
addVectorFields=True
netMinPatchSize = patchSize[0]
fineTuneOldModel=(False, False, False) # works only in combination with "previousModels" parameter
randomSampling=(False,False,False)




