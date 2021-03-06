import torch
from torch.utils.data import DataLoader
import torch.multiprocessing as mp

import os
import sys, getopt
import numpy as np

from OptimizeCaller import OptimizeCaller

import time

from HeadAndNeckDataset import HeadAndNeckDataset, ToTensor
from Net import UNet
import Options as userOpts

def main(argv):
  
  #torch.backends.cudnn.enabled = False
  #CUDA_LAUNCH_BLOCKING = 1
  callString = 'OnePatchShot.py --trainingFiles=files.csv --device=device --outputPath=PATH'
  
  try:
    opts, args = getopt.getopt(argv, '', ['trainingFiles=', 'testModels', 'fineTune=', 'randomSampling=', 'validationFiles=', 'previousModels=', 'device=', 'maskOutZeros', 'noDiff', 'outputPath=', 'stoptAtSampleStep=', 'downSampleSteps=', 'cycleW=', 'smoothW=', 'boundarySmoothW='])
  except getopt.GetoptError as e:#python3
    print(e)
    print(callString)
    return
    
  if not (len(opts)):
    print(callString)
    return

  oldModelList = None
  testModels = False
  for opt, arg in opts:
    if opt == '--trainingFiles':
      userOpts.trainingFileNamesCSV = arg
    elif opt == '--validationFiles':
      userOpts.validationFileNameCSV = arg      
    elif opt == '--device':
      userOpts.device = arg      
    elif opt == '--outputPath':
      userOpts.outputPath = arg      
    elif opt == '--maskOutZeros':
      userOpts.maskOutZeros = True
    elif opt == '--stoptAtSampleStep':
      userOpts.stoptAtSampleStep = int(arg)
    elif opt == '--downSampleSteps':
      userOpts.downSampleSteps = int(arg)      
    elif opt == '--cycleW':
      userOpts.cycleW = float(arg)
    elif opt == '--noDiff':
      userOpts.diffeomorphicRegistration = False  
      userOpts.overlappingPatches = False    
    elif opt == '--smoothW':
      stringList = arg.split()
      userOpts.smoothW = [float(i) for i in stringList]
    elif opt == '--boundarySmoothW':
      stringList = arg.split()
      userOpts.boundarySmoothnessW = [float(i) for i in stringList]
    elif opt == '--previousModels':
      oldModelList = arg.split()
    elif opt == '--testModels':
      testModels = True    
    elif opt == '--fineTune':
      stringList = arg.split()
      userOpts.fineTuneOldModel=[i.lower() in("true","t") for i in stringList]
    elif opt == '--randomSampling':
      stringList = arg.split()
      userOpts.randomSampling = [i.lower() in("true","t") for i in stringList]
              
      
  torch.manual_seed(0)
  np.random.seed(0)
  torch.cuda.manual_seed(0)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False
  if userOpts.debugMode:
    torch.autograd.set_detect_anomaly(True)

  if not os.path.isdir(userOpts.outputPath):
    os.makedirs(userOpts.outputPath)

  headAndNeckTrainSet = HeadAndNeckDataset(userOpts.trainingFileNamesCSV, ToTensor(), True)
  if hasattr(userOpts, 'validationFileNameCSV'):
    validationSet = HeadAndNeckDataset(userOpts.validationFileNameCSV, ToTensor(), True)
    validationDataLoader = DataLoader(validationSet, batch_size=1, shuffle=False, num_workers=0)
  
  dataloader = DataLoader(headAndNeckTrainSet, batch_size=1,
                        shuffle=False, num_workers=0)
  
  net = UNet(headAndNeckTrainSet.getChannels(), True, False, userOpts.netDepth, userOpts.numberOfFiltersFirstLayer, useDeepSelfSupervision=False)
  with OptimizeCaller(net, userOpts) as trainTestOptimize:
    print(net)
    start = time.time()
    if oldModelList is not None:
      trainTestOptimize.setOldModels(oldModelList)
    if False:#userOpts.device == "cpu":
      net.share_memory()
      processes = []
      num_processes = 2
      for rank in range(num_processes):
        if testModels:
          p = mp.Process(target=trainTestOptimize.testNetDownSamplePatch, args=(dataloader))
        elif hasattr(userOpts, 'validationFileNameCSV'):
          p = mp.Process(target=trainTestOptimize.trainNetDownSamplePatch, args=(dataloader, validationDataLoader))
        else:
          p = mp.Process(target=trainTestOptimize.trainTestNetDownSamplePatch, args=(dataloader))
        p.start()
        processes.append(p)
      for p in processes:
        p.join()
          
    else:
      if testModels:
        trainTestOptimize.testNetDownSamplePatch(dataloader)
      elif hasattr(userOpts, 'validationFileNameCSV'):
        trainTestOptimize.trainNetDownSamplePatch(dataloader, validationDataLoader)
      else:
        trainTestOptimize.trainTestNetDownSamplePatch(dataloader)
    end = time.time()
    print('Registration took overall:', end - start, 'seconds')
    

if __name__ == "__main__":
  main(sys.argv[1:]) 
