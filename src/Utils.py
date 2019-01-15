import numpy as np
import SimpleITK as sitk
import torch
from GaussSmoothing import GaussianSmoothing

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

def smoothArray3D(inputArray, device):
    smoothing = GaussianSmoothing(1, 5, 2, 3, device)
    input = torch.nn.functional.pad(inputArray, (2,2,2,2,2,2))
    input = input[None, None, ]
    input = smoothing(input)
    input = torch.nn.functional.pad(input, (2,2,2,2,2,2))
    return smoothing(input)[0,0]
  
def getMaxIdxs(imgShape, imgPatchSize):
  if imgShape[2] - imgPatchSize > 0:
    maxidx0 = imgShape[2] - imgPatchSize + 1
  else:
    maxidx0 = 1
  
  if imgShape[3] - imgPatchSize > 0:
    maxidx1 = imgShape[3] - imgPatchSize + 1
  else:
    maxidx1 =  1
  
  if imgShape[4] - imgPatchSize > 0:
    maxidx2 = imgShape[4] - imgPatchSize + 1
  else:
    maxidx2 = 1
  return (maxidx0, maxidx1, maxidx2)

def getPatchSize(imgShape, imgPatchSize):
  if imgShape[2] - imgPatchSize > 0:
    patchSize0 = imgPatchSize
  else:
    patchSize0 = imgShape[2]
  
  if imgShape[3] - imgPatchSize > 0:
    patchSize1 = imgPatchSize
  else:
    patchSize1 = imgShape[3]
  
  if imgShape[4] - imgPatchSize > 0:
    patchSize2 = imgPatchSize
  else:
    patchSize2 = imgShape[4]  
    
  return (patchSize0, patchSize1, patchSize2)

def deformImage(imgToDef, defFields, device, detach=True):
  zeroDefField = getZeroDefField(imgToDef.shape)
  zeroDefField = zeroDefField.to(device)  
  currDefField = torch.empty(zeroDefField.shape, device=device, requires_grad=False)
  if (detach):
    currDefField[..., 0] = zeroDefField[..., 0] + defFields[:, 0, ].detach()
    currDefField[..., 1] = zeroDefField[..., 1] + defFields[:, 1, ].detach()
    currDefField[..., 2] = zeroDefField[..., 2] + defFields[:, 2, ].detach()
  else:
    currDefField[..., 0] = zeroDefField[..., 0] + defFields[:, 0, ]
    currDefField[..., 1] = zeroDefField[..., 1] + defFields[:, 1, ]
    currDefField[..., 2] = zeroDefField[..., 2] + defFields[:, 2, ]
  deformedTmp = torch.nn.functional.grid_sample(imgToDef, currDefField, mode='bilinear', padding_mode='border')
  return deformedTmp