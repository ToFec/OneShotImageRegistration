import sys
import getopt
import SimpleITK as sitk
import numpy as np
from src import Utils
import torch

def main(argv=None): # IGNORE:C0111
  try:
    opts, _ = getopt.getopt(argv, 'f:o:', ['files=', 'output='])
  except getopt.GetoptError:
    return

  filenames = None
  outputFilename = "combinedDefField.nrrd"
  for opt, arg in opts:
    if opt in ("-f", "--files"):
      filenames = arg.split()
    elif opt in ("-o", "--output"):
      outputFilename = str(arg)

  if filenames is not None:
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
    
    if oldNet is not None:
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