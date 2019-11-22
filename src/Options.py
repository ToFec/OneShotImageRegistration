import torch
#
#general options
#
useContext = True
debugMode=True
numberOfEpochs = [1000,1000,1000]
#numberOfEpochs = [200,250,1000]#with training of 3D pairs
trainingFileNamesCSV=''
device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
outputPath='.'

netDepth=3
numberOfFiltersFirstLayer=32
netMinPatchSizePadded = 8
patchSize=(80,80,80)
netMinPatchSize = patchSize[0]
normImgPatches=False
trainTillConvergence = True
ccCalcNN=True
maskOutZeros=False
useMedianForSampling = (False,False,True)

#
#cost functon parameters
#
ccW=1.0
dscWeight=0.0
downSampleSteps = 2# a size of 2 means 3 iterations with the following downsampling factors: (0.25,0.5,1.0)
stoptAtSampleStep=3
boundarySmoothnessW=(0.0,0.1,0.1)
smoothW = (0.0001,0.0001,0.0001)
cycleW = 0.01
lossTollerance=0.00001

#
#difeomorphic version parameters
#
diffeomorphicRegistration = True
overlappingPatches=True
finalGaussKernelSize=7
finalGaussKernelStd=2
sasSteps=5
smoothVF = True


#
#training parameters
#
validationIntervall=10
# numberOfEpochs = [75,75,75]
addVectorFields=True
fineTuneOldModel=(False, False, False) # works only in combination with "previousModels" parameter
randomSampling=(False,False,False)




