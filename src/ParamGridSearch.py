import torch
from torch.utils.data import DataLoader

import subprocess

import os
import sys, getopt
import numpy as np

from Optimize import Optimize

import time

from HeadAndNeckDataset import HeadAndNeckDataset, ToTensor
from Net import UNet
import Options as userOpts

def main(argv):
  
  #torch.backends.cudnn.enabled = False
  #CUDA_LAUNCH_BLOCKING = 1
  callString = 'ParamGridSearch.py --trainingFiles=files.csv --outputPath=PATH'
  
  try:
    opts, args = getopt.getopt(argv, '', ['trainingFiles=', 'outputPath='])
  except getopt.GetoptError, e:
    print(e)
    print(callString)
    return
    
  if not (len(opts)):
    print(callString)
    return

  outputPath = 'RegResults'  
  for opt, arg in opts:
    if opt == '--trainingFiles':
      userOpts.trainingFileNamesCSV = arg
    elif opt == '--outputPath':
      userOpts.outputPath = arg      
      
  torch.manual_seed(0)
  np.random.seed(0)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False

  headAndNeckTrainSet = HeadAndNeckDataset(userOpts.trainingFileNamesCSV, ToTensor())
  
  dataloader = DataLoader(headAndNeckTrainSet, batch_size=1,
                        shuffle=False, num_workers=0)
  
  if not os.path.isdir(outputPath):
    os.makedirs(outputPath)
  
  cCW = 0.6
  ccWStep = 0.05
  smoothW = 0.0
  smoothWStep = 0.01
  cycleW = 0.0
  cycleWStep = 0.01
  vecLengthW = 0.0
  vecLengthWSetp = 0.01
  
  net = UNet(headAndNeckTrainSet.getChannels(), True, False, userOpts.netDepth)
  
  while cCW <= 1.0:
    while smoothW <= 0.2:
      while cycleW <= 0.2:
        while vecLengthW <= 0.2:
          if (cCW + smoothW + cycleW + vecLengthW == 1.0):
            userOpts.smoothW = smoothW
            userOpts.cycleW = cycleW
            userOpts.ccW = cCW
            userOpts.vecLengthW = vecLengthW
            
            trainTestOptimize = Optimize(net, userOpts)
            start = time.time()
            trainTestOptimize.trainNet(dataloader)
            end = time.time()
            
            timeForTraining = end - start
            finalLoss = trainTestOptimize.finalLoss
            numberofiterations = trainTestOptimize.finalNumberIterations
            
            trainTestOptimize.testNet(dataloader)
            
            print subprocess.check_output(['plastimatch','-l'])
            
          vecLengthW += vecLengthWSetp
        vecLengthW = 0.0
        cycleW += cycleWStep
      cycleW = 0.0
      smoothW += smoothWStep
    smoothW = 0.0
    cCW += ccWStep
          


if __name__ == "__main__":
  main(sys.argv[1:]) 