import numpy as np
import SimpleITK as sitk
import torch

def deform(inputVol, x1, y1, z1):
  ##http://simpleitk.github.io/SimpleITK-Notebooks/01_Image_Basics.html
  imgToDeform = sitk.GetImageFromArray(inputVol)
  defField = np.stack([x1, y1, z1])
  defField = np.moveaxis(defField, 0, -1)
  itkVecImg = sitk.GetImageFromArray(defField.astype('f8'))
  displacement = sitk.DisplacementFieldTransform(itkVecImg)

  resampler = sitk.ResampleImageFilter()
  resampler.SetReferenceImage(imgToDeform)
  resampler.SetInterpolator(sitk.sitkLinear)
  resampler.SetDefaultPixelValue(0)
  resampler.SetTransform(displacement)
  moving = resampler.Execute(imgToDeform)
  
  deformedVol = sitk.GetArrayFromImage(moving) 
  
  return deformedVol

def getDefField(x1, y1, z1):
  defField = np.stack([x1, y1, z1])
  defField = np.moveaxis(defField, 0, -1)
  return defField

#deformation field for grid_sample function with zero deformation
def getZeroDefField(imagShape):
  m0=np.linspace(-1, 1, imagShape[2], dtype=np.float32)
  m1=np.linspace(-1, 1, imagShape[3], dtype=np.float32)
  m2=np.linspace(-1, 1, imagShape[4], dtype=np.float32)
  grid0, grid1, grid2 = np.meshgrid(m0,m1,m2,indexing='ij')
  defField = np.stack([grid2, grid1, grid0], axis=3)
  defField = np.expand_dims(defField, axis=0)
  defField = np.tile(defField, (imagShape[0],1,1,1,1))
  defField = torch.from_numpy(defField)
  return defField