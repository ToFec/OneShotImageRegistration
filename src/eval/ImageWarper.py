import getopt, sys
import SimpleITK as sitk
import torch
import numpy as np

import src.Utils

def main(argv):
  try:
    opts, _ = getopt.getopt(argv, 'i:d:o:b', ['img=', 'defField=', 'output=', 'binary'])
  except getopt.GetoptError as e:#python3
    print(e)
    return
  
  outputfile = None
  imageFileName = None
  defFieldFileName = None
  isBinary=False
  for opt, arg in opts:
    if opt in ("-i", "--img"):
      imageFileName = arg
    elif opt in ("-d", "--defField"):
      defFieldFileName = arg
    elif opt in ("-o", "--output"):
      outputfile = arg
    elif opt in ("-b", "--binary"):
      isBinary = True
  
  
  if outputfile is not None and imageFileName is not None and defFieldFileName is not None:      
    defFieldITK = sitk.ReadImage(str(defFieldFileName))
    defField = sitk.GetArrayFromImage(defFieldITK)
      
#     defField = np.moveaxis(defField, 0, 2)
#     defField = np.moveaxis(defField, 0, 1)
    defFieldSpacing = defFieldITK.GetSpacing()
    defFieldDirection = defFieldITK.GetDirection()
    defField[...,0] = (defField[...,0] / defFieldSpacing[0]) * defFieldDirection[0]
    defField[...,1] = (defField[...,1] / defFieldSpacing[1]) * defFieldDirection[4]
    defField[...,2] = (defField[...,2] / defFieldSpacing[2]) * defFieldDirection[8]

    defField = np.moveaxis(defField, -1, 0)
    defField = np.expand_dims(defField, axis=0)

    
    imageITK = sitk.ReadImage(str(imageFileName))
    image = sitk.GetArrayFromImage(imageITK)
    image = np.expand_dims(image, axis=0)
    image = np.expand_dims(image, axis=0)
    imageSpacing = imageITK.GetSpacing()
    imageOrigin = imageITK.GetOrigin()
    imageDirection = imageITK.GetDirection()
    
    imageTorch = torch.from_numpy(image)
    imageType = imageTorch.dtype
    imageTorch = imageTorch.float()
    defFieldTorch = torch.from_numpy(defField)
    

    
    
    if isBinary:
      deformedImage = src.Utils.deformWithNearestNeighborInterpolation(imageTorch, defFieldTorch, 'cpu')
    else:
      deformedImage = src.Utils.deformImage(imageTorch, defFieldTorch, 'cpu')
    deformedImageITK = sitk.GetImageFromArray(torch.tensor(deformedImage[0, 0, ],dtype=imageType))
    
    deformedImageITK.SetSpacing( imageSpacing )
    deformedImageITK.SetOrigin( imageOrigin )
    deformedImageITK.SetDirection( imageDirection )
    sitk.WriteImage(deformedImageITK, outputfile)
  
  
if __name__ == "__main__":
  main(sys.argv[1:])    