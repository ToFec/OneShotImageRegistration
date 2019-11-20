'''
Created on Oct 24, 2019

@author: fechter
'''

from torch.utils.data import DataLoader

import os
import sys, getopt
import torch
import time
from HeadAndNeckDataset import HeadAndNeckDataset
import SimpleITK as sitk

def computeDefField(parameterFile, outputDir, imageToReg):
  transformixImageFilter = sitk.TransformixImageFilter()
  parameterMap = transformixImageFilter.ReadParameterFile(parameterFile)
  transformixImageFilter.SetTransformParameterMap(parameterMap)
  transformixImageFilter.SetMovingImage(imageToReg)
  transformixImageFilter.ComputeDeformationFieldOn()
  transformixImageFilter.SetOutputDirectory(outputDir)
  transformixImageFilter.Execute()
  
  
def main(argv):
  
  #torch.backends.cudnn.enabled = False
  #CUDA_LAUNCH_BLOCKING = 1
  callString = 'ElasticCycle.py --regFiles=files.csv --outputPath=PATH'
  
  try:
    opts, args = getopt.getopt(argv, '', ['regFiles=', 'outputPath=', 'defFieldConf=', 'noMask'])
  except getopt.GetoptError as e:#python3
    print(e)
    print(callString)
    return
    
  if not (len(opts)):
    print(callString)
    return

  inputFiles=None
  compDefField=None
  doMask = True
  for opt, arg in opts:
    if opt == '--regFiles':
      inputFiles = arg
    elif opt == '--outputPath':
      outputPath = arg      
    elif opt == '--defFieldConf':
      compDefField = arg 
    elif opt == '--noMask':
      doMask = False 
      
  if not os.path.isdir(outputPath):
    os.makedirs(outputPath)

    
  headAndNeckTrainSet = HeadAndNeckDataset(inputFiles, loadOnInstantiation=False,normlizeImages=False)
  dataloader = DataLoader(headAndNeckTrainSet, batch_size=1, shuffle=False, num_workers=0)
  for dataIdx, data in enumerate(dataloader, 0):
    outputDir = outputPath + os.path.sep + str(dataIdx)
    if not os.path.isdir(outputDir):
      os.makedirs(outputDir)
    img = data['image']
    if doMask:
      mask = data['mask']
      mask[mask > 0] = 1
    
    timebins = img.shape[1]
    vectorOfImages = sitk.VectorOfImage()
    vectorOfMasks = sitk.VectorOfImage()
    for t in range(0,timebins):
      vectorOfImages.push_back(sitk.GetImageFromArray(img.to(torch.short)[0,t,...]))
      if doMask:
        vectorOfMasks.push_back(sitk.GetImageFromArray(mask[0,t,...]))
      
    start = time.time()
    
    
    
    
    
    image = sitk.JoinSeries(vectorOfImages)
    
    
    image.SetSpacing( headAndNeckTrainSet.getSpacing(dataIdx) + (1,) )
    image.SetOrigin( headAndNeckTrainSet.getOrigin(dataIdx) + (0,) )
    

    if compDefField is not None:
      computeDefField(compDefField, outputPath, image)
    else:
        sitk.WriteImage(image, outputDir + os.path.sep + 'image4D.nrrd')
        
        if doMask:
          mask = sitk.JoinSeries(vectorOfMasks)
          mask.SetSpacing( headAndNeckTrainSet.getSpacing(dataIdx) + (1,) )
          mask.SetOrigin( headAndNeckTrainSet.getOrigin(dataIdx) + (0,) )  
          sitk.WriteImage(mask, outputDir + os.path.sep + 'mask4D.nrrd')
        
        
        continue
        elastixImageFilter = sitk.ElastixImageFilter()
        
        parameterMapForward = elastixImageFilter.ReadParameterFile('/home/fechter/workspace/OneShotImageRegistration/Elastix_ct-lungs/proposed/spacing1-cylic/par000.forward.txt')
        parameterMapBackward = elastixImageFilter.ReadParameterFile('/home/fechter/workspace/OneShotImageRegistration/Elastix_ct-lungs/proposed/spacing1-cylic/par001.inverse.txt')
        
        elastixImageFilter.SetParameterMap(parameterMapForward)
        elastixImageFilter.AddParameterMap(parameterMapBackward)
        
        elastixImageFilter.SetFixedImage(image)
        elastixImageFilter.SetFixedMask(mask)
        elastixImageFilter.SetMovingImage(image)
        #     elastixImageFilter.SetMovingMask(mask)
        
        elastixImageFilter.SetOutputDirectory(outputDir)
    
        elastixImageFilter.LogToConsoleOn()
        elastixImageFilter.Execute()
      
  
  #   
  #   resultImages = elastixImageFilter.GetResultImage()
  #   image3dSize = list(resultImages.GetSize())
  #   image3dSize[3] = 0
  #   for resultImgIdx in range(0, resultImages.GetSize()[3]):
  #     Extractor = sitk.ExtractImageFilter()
  #     Extractor.SetSize( image3dSize )
  #     Extractor.SetIndex( [0,0,0,resultImgIdx] )
  #     sitk.WriteImage( Extractor.Execute( resultImages ), outputPath + os.path.sep + "result_timeBin_" + str(resultImgIdx) )
  
  
  end = time.time()
  print('Registration took overall:', end - start, 'seconds')
    

if __name__ == '__main__':
  main(sys.argv[1:]) 