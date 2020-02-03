import sys
import getopt
import SimpleITK as sitk
import numpy as np
from OneShotImageRegistration.src import Utils
import torch


# TODO: really slow at the moment; vectorize!
def invertDirectionOfField(defField):
  fieldShape = defField.shape[2:]
  nuOfItems=fieldShape[0]*fieldShape[1]*fieldShape[2]
  vectorList = [None]*nuOfItems
  for idx0 in range(fieldShape[0]):
    for idx1 in range(fieldShape[1]):
      for idx2 in range(fieldShape[2]):
        defVec = defField[0, :, idx0, idx1, idx2]
        newPos0 = idx0+defVec[2]
        if newPos0 < 0:
          newPos0 = 0
        elif newPos0 > fieldShape[0] -2:
          newPos0 = fieldShape[0] -2
        newPos1 = idx1+defVec[1]
        if newPos1 < 0:
          newPos1 = 0
        elif newPos1 > fieldShape[1] -2:
          newPos1 = fieldShape[1] -2     
        newPos2 = idx2+defVec[0]
        if newPos2 < 0:
          newPos2 = 0
        elif newPos2 > fieldShape[2] -2:
          newPos2 = fieldShape[2] -2
        
        roundLow0 = int(newPos0)
        roundUp0 = roundLow0 + 1
        roundLow1 = int(newPos1)
        roundUp1 = roundLow1 + 1
        roundLow2 = int(newPos2)
        roundUp2 = roundLow2 +1
        
        
        upperPart0 = newPos0 - roundLow0
        lowerPart0 = 1 - upperPart0
        
        upperPart1 = newPos1 - roundLow1
        lowerPart1 = 1 - upperPart1
        
        upperPart2 = newPos2 - roundLow2
        lowerPart2 = 1 - upperPart2
        
        idx000 = roundLow0 + roundLow1 * fieldShape[0] + roundLow2 * fieldShape[0] * fieldShape[1]
        idx001 = roundLow0 + roundLow1 * fieldShape[0] + roundUp2 * fieldShape[0] * fieldShape[1]
        idx010 = roundLow0 + roundUp1 * fieldShape[0] + roundUp2 * fieldShape[0] * fieldShape[1]
        idx100 = roundUp0 + roundLow1 * fieldShape[0] + roundLow2 * fieldShape[0] * fieldShape[1]
        idx011 = roundLow0 + roundUp1 * fieldShape[0] + roundUp2 * fieldShape[0] * fieldShape[1]
        idx101 = roundUp0 + roundLow1 * fieldShape[0] + roundUp2 * fieldShape[0] * fieldShape[1]
        idx110 = roundUp0 + roundUp1 * fieldShape[0] + roundLow2 * fieldShape[0] * fieldShape[1]
        idx111 = roundUp0 + roundUp1 * fieldShape[0] + roundUp2 * fieldShape[0] * fieldShape[1]
        
        if vectorList[idx000] is None:
          vectorList[idx000] = [[(lowerPart0, lowerPart1, lowerPart2),(-defVec[0], -defVec[1], -defVec[2]) ]]
        else:
          vectorList[idx000].append([(lowerPart0, lowerPart1, lowerPart2),(-defVec[0], -defVec[1], -defVec[2]) ])

        if vectorList[idx001] is None:
          vectorList[idx001] = [[(lowerPart0, lowerPart1, upperPart2),(-defVec[0], -defVec[1], -defVec[2]) ]]
        else:
          vectorList[idx001].append([(lowerPart0, lowerPart1, upperPart2),(-defVec[0], -defVec[1], -defVec[2]) ])
          
        if vectorList[idx010] is None:
          vectorList[idx010] = [[(lowerPart0, upperPart1, lowerPart2),(-defVec[0], -defVec[1], -defVec[2]) ]]
        else:
          vectorList[idx010].append([(lowerPart0, upperPart1, lowerPart2),(-defVec[0], -defVec[1], -defVec[2]) ])
          
        if vectorList[idx100] is None:
          vectorList[idx100] = [[(upperPart0, lowerPart1, lowerPart2),(-defVec[0], -defVec[1], -defVec[2]) ]]
        else:
          vectorList[idx100].append([(upperPart0, lowerPart1, lowerPart2),(-defVec[0], -defVec[1], -defVec[2]) ])
          
        if vectorList[idx011] is None:
          vectorList[idx011] = [[(lowerPart0, upperPart1, upperPart2),(-defVec[0], -defVec[1], -defVec[2]) ]]
        else:
          vectorList[idx011].append([(lowerPart0, upperPart1, upperPart2),(-defVec[0], -defVec[1], -defVec[2]) ])
          
        if vectorList[idx101] is None:
          vectorList[idx101] = [[(upperPart0, lowerPart1, upperPart2),(-defVec[0], -defVec[1], -defVec[2]) ]]
        else:
          vectorList[idx101].append([(upperPart0, lowerPart1, upperPart2),(-defVec[0], -defVec[1], -defVec[2]) ]) 
          
        if vectorList[idx110] is None:
          vectorList[idx110] = [[(upperPart0, upperPart1, lowerPart2),(-defVec[0], -defVec[1], -defVec[2]) ]]
        else:
          vectorList[idx110].append([(upperPart0, upperPart1, lowerPart2),(-defVec[0], -defVec[1], -defVec[2]) ])    
          
        if vectorList[idx111] is None:
          vectorList[idx111] = [[(upperPart0, upperPart1, upperPart2),(-defVec[0], -defVec[1], -defVec[2]) ]]
        else:
          vectorList[idx111].append([(upperPart0, upperPart1, upperPart2),(-defVec[0], -defVec[1], -defVec[2]) ])
   
  newField = torch.zeros_like(defField)       
  for idx0 in range(fieldShape[0]):
    for idx1 in range(fieldShape[1]):
      for idx2 in range(fieldShape[2]):
        
        idx = idx0 + idx1 * fieldShape[0] + idx2 * fieldShape[0] * fieldShape[1]
        vectorParsList = vectorList[idx]
        if vectorParsList is not None:
          partSum0, partSum1, partSum2 = 0.0, 0.0, 0.0
          for vectPart in vectorParsList:
            partSum0 = partSum0 + vectPart[0][0]
            partSum1 = partSum1 + vectPart[0][1]
            partSum2 = partSum2 + vectPart[0][2]
          
          vecPart0, vecPart1, vecPart2 = 0.0, 0.0, 0.0
          for vectPart in vectorParsList:
            if partSum0 > 0:
              vecPart2 = vecPart2 + (vectPart[0][0] / partSum0) * vectPart[1][2]
            if partSum1 > 0:
              vecPart1 = vecPart1 + (vectPart[0][1] / partSum1) * vectPart[1][1]
            if partSum2 > 0:
              vecPart0 = vecPart0 + (vectPart[0][2] / partSum2) * vectPart[1][0]
          
          newField[0,0,idx0, idx1, idx2]=vecPart0
          newField[0,1,idx0, idx1, idx2]=vecPart1
          newField[0,2,idx0, idx1, idx2]=vecPart2
        
  return newField
          
          
def combineDeformationFields(filenames):
  oldNet = None
  for filename in filenames:
    defFieldITK = sitk.ReadImage(str(filename))
    defFieldSpacing = defFieldITK.GetSpacing()
    defFieldOrigin = defFieldITK.GetOrigin()
    defFieldDirection = defFieldITK.GetDirection()
    
    defField = sitk.GetArrayFromImage(defFieldITK)
    defField[...,0] = (defField[...,0] / defFieldSpacing[0]) * defFieldDirection[0]
    defField[...,1] = (defField[...,1] / defFieldSpacing[1]) * defFieldDirection[4]
    defField[...,2] = (defField[...,2] / defFieldSpacing[2]) * defFieldDirection[8]
    
    defField = np.moveaxis(defField, -1, 0)
    defField = np.expand_dims(defField, axis=0)
    defField = torch.from_numpy(defField)
    if oldNet is None:
      oldNet = defField
    else:
      oldNet = Utils.combineDeformationFields(defField, oldNet)
  
  return oldNet, defFieldSpacing, defFieldOrigin, defFieldDirection                                                          

def main(argv=None): # IGNORE:C0111
  try:
    opts, _ = getopt.getopt(argv, 'f:o:i', ['files=', 'output=', 'invert'])
  except getopt.GetoptError:
    return

  filenames = None
  outputFilename = "combinedDefField.nrrd"
  invertField = False
  for opt, arg in opts:
    if opt in ("-f", "--files"):
      filenames = arg.split()
    elif opt in ("-o", "--output"):
      outputFilename = str(arg)
    elif opt in ("-i", "--invert"):
      invertField = True      
      

  if filenames is not None:
    oldNet, defFieldSpacing, defFieldOrigin, defFieldDirection  = combineDeformationFields(filenames)
    
    
    if oldNet is not None:
      if invertField:
        oldNet = invertDirectionOfField(oldNet)
      
      defX = oldNet[0, 0, ].detach() * defFieldSpacing[0] * defFieldDirection[0]
      defY = oldNet[0, 1, ].detach() * defFieldSpacing[1] * defFieldDirection[4]
      defZ = oldNet[0, 2, ].detach() * defFieldSpacing[2] * defFieldDirection[8]
      defField = Utils.getDefField(defX, defY, defZ)
      

      
      defDataToSave = sitk.GetImageFromArray(defField, isVector=True)      
      
      defDataToSave.SetSpacing( defFieldSpacing )
      defDataToSave.SetOrigin( defFieldOrigin )
      defDataToSave.SetDirection( defFieldDirection )
      
      sitk.WriteImage(defDataToSave, outputFilename)      
  

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))