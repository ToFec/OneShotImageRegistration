import torch
from torch.utils.data import DataLoader

import subprocess

import os
from glob import glob
import sys, getopt
import numpy as np

from Optimize import Optimize

import time

from HeadAndNeckDataset import HeadAndNeckDataset, ToTensor, SmoothImage
from Net import UNet
import Options as userOpts
from eval.LandmarkHandler import PointProcessor

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

  for opt, arg in opts:
    if opt == '--trainingFiles':
      userOpts.trainingFileNamesCSV = arg
    elif opt == '--outputPath':
      userOpts.outputPath = arg      
      
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False

  headAndNeckTrainSet = HeadAndNeckDataset(userOpts.trainingFileNamesCSV, ToTensor(), True, SmoothImage())
  
  dataloader = DataLoader(headAndNeckTrainSet, batch_size=1,
                        shuffle=False, num_workers=0)
  
  if not os.path.isdir(userOpts.outputPath):
    os.makedirs(userOpts.outputPath)
  
  cCW = 0.7
  ccWStep = 0.1
  smoothW = 0.0
  smoothWStep = 0.05
  cycleW = 0.0
  cycleWStep = 0.05
  vecLengthW = 0.0
  vecLengthWSetp = 0.05
  
  net = UNet(headAndNeckTrainSet.getChannels(), True, False, userOpts.netDepth)
  
  rootDir = userOpts.outputPath
  while cCW <= 1.0:
    while smoothW <= 0.1:
      while cycleW <= 0.1:
        while vecLengthW <= 0.1:
          currDir = rootDir + os.path.sep + 'ccW' + str(cCW) + 'smoothW' + str(smoothW) + 'cycleW' + str(cycleW) + 'vecLengthW' + str(vecLengthW)
          if not os.path.isdir(currDir):
            os.makedirs(currDir)
          userOpts.outputPath = currDir
          
          
          torch.manual_seed(0)
          np.random.seed(0)
          
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
          
          logfile = userOpts.outputPath + os.path.sep + 'lossIterLog.csv'    
          logFile = open(logfile,'w', buffering=0)
          logFile.write('timeForTraining;finalLoss;numberofiterations\n')
          logFile.write(str(timeForTraining) + ';')
          logFile.write(str(finalLoss.item()) + ';')
          logFile.write(str(numberofiterations))
          logFile.close()
          
          trainTestOptimize.testNet(dataloader)
          
          for f in glob (userOpts.outputPath + os.path.sep + 'deformedImgDataset*'):
            os.unlink (f)

          for f in glob (userOpts.outputPath + os.path.sep + 'origImgDataset*'):
            os.unlink (f)
            
          for f in glob (userOpts.outputPath + os.path.sep + 'deformationFieldDataset*'):
            os.unlink (f)            
          
          vecLengthW += vecLengthWSetp
        vecLengthW = 0.0
        cycleW += cycleWStep
      cycleW = 0.0
      smoothW += smoothWStep
    smoothW = 0.0
    cCW += ccWStep
          


if __name__ == "__main__":
  main(sys.argv[1:]) 