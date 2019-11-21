import getopt, sys, os
import SimpleITK as sitk
import torch
import numpy as np
import src.ScalingAndSquaringTmp as sas

sys.path.insert(0,os.path.realpath('.'))
import src.Utils

def main(argv):
  
  try:
    opts, args = getopt.getopt(argv, 'd:o:', ['defField=', 'output=' ])
  except getopt.GetoptError as e:#python3
    print(e)
    return  
  
  outputfile = None
  defFieldFileName = None
  for opt, arg in opts:
    if opt in ("-d", "--defField"):
      defFieldFileName = arg
    elif opt in ("-o", "--output"):
      outputfile = arg
  
  if outputfile is not None and defFieldFileName is not None:      
    defFieldITK = sitk.ReadImage(str(defFieldFileName))
    defField = sitk.GetArrayFromImage(defFieldITK)
    defFieldOrigin = defFieldITK.GetOrigin()
    defFieldSpacing = defFieldITK.GetSpacing()
    defFieldDirection = defFieldITK.GetDirection()
    defField[...,0] = (defField[...,0]) * defFieldDirection[0]
    defField[...,1] = (defField[...,1]) * defFieldDirection[4]
    defField[...,2] = (defField[...,2]) * defFieldDirection[8]

    defField = np.moveaxis(defField, -1, 0)
    defField = np.expand_dims(defField, axis=0) 
    
    defFieldTorch = torch.from_numpy(defField) 
    
    scalingSquaring = sas.ScalingAndSquaring(num_steps=3)
    
    deformationField = scalingSquaring(defFieldTorch)
    
    defX = deformationField[0, 0, ]* defFieldDirection[0]
    defY = deformationField[0, 1, ]* defFieldDirection[4]
    defZ = deformationField[0, 2, ] * defFieldDirection[8]
    defField = src.Utils.getDefField(defX, defY, defZ)
    defDataToSave = sitk.GetImageFromArray(defField, isVector=True)    
    defDataToSave.SetSpacing( defFieldSpacing )
    defDataToSave.SetOrigin( defFieldOrigin )
    defDataToSave.SetDirection(defFieldDirection)
    sitk.WriteImage(defDataToSave, outputfile)
  

  
if __name__ == "__main__":
  main(sys.argv[1:])    