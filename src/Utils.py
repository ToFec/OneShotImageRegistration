import numpy as np
import SimpleITK as sitk
import torch
from GaussSmoothing import GaussianSmoothing
import Context
import pickle
import Options as useropts

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
def getZeroDefField(imagShape, device):
  if (Context.zeroDefField is None) or (imagShape[2:] != Context.zeroDefField.shape):
    m0=np.linspace(-1, 1, imagShape[2], dtype=np.float32)
    m1=np.linspace(-1, 1, imagShape[3], dtype=np.float32)
    m2=np.linspace(-1, 1, imagShape[4], dtype=np.float32)
    grid0, grid1, grid2 = np.meshgrid(m0,m1,m2,indexing='ij')
    defField = np.stack([grid2, grid1, grid0], axis=3)
    defField = np.expand_dims(defField, axis=0)
    defField = np.tile(defField, (imagShape[0],1,1,1,1))
    defField = torch.from_numpy(defField)
    Context.zeroDefField = defField.to(device)
    
  return Context.zeroDefField

def getZeroIdxField(imagShape, device):
  if (Context.zeroIndices is None) or (imagShape[2:] != Context.zeroIndices[0].shape[2:]):
    zeroIndices = torch.from_numpy( np.indices([imagShape[0],3,imagShape[2],imagShape[3],imagShape[4]],dtype=np.float32))
    zeroIndices[1] -= 3.0
    idxs0 = zeroIndices[0].long().to(device)
    idxs1 = zeroIndices[1].long().to(device)
    idxs2 = zeroIndices[2].to(device)
    idxs3 = zeroIndices[3].to(device)
    idxs4 = zeroIndices[4].to(device)
    if not useropts.useContext:
      return [idxs0, idxs1, idxs2, idxs3, idxs4]
    Context.zeroIndices = [idxs0, idxs1, idxs2, idxs3, idxs4]
  [idxs0, idxs1, idxs2, idxs3, idxs4] = Context.zeroIndices
  return [idxs0.clone(), idxs1.clone(), idxs2.clone(), idxs3.clone(), idxs4.clone()]

def getImgDataDef(imagShape, device):
  if useropts.useContext:
    if (Context.imgDataDef is None) or (imagShape != Context.imgDataDef.shape):
      imgDataDef = torch.empty(imagShape, device=device, requires_grad=False)
      Context.imgDataDef = imgDataDef
    else:
      Context.imgDataDef.detach_()
    return Context.imgDataDef
  else:
    return torch.empty(imagShape, device=device, requires_grad=False)

def getImgDataDef2(imagShape, device):
  if useropts.useContext:
    if (Context.imgDataDef2 is None) or (imagShape != Context.imgDataDef2.shape):
      imgDataDef2 = torch.empty(imagShape, device=device, requires_grad=False)
      Context.imgDataDef2 = imgDataDef2
    else:
      Context.imgDataDef2.detach_()
    return Context.imgDataDef2
  else:
    return torch.empty(imagShape, device=device, requires_grad=False)

def getCycleImgData(defFieldShape, device):
  if useropts.useContext:
    if (Context.cycleImgData is None) or (defFieldShape != Context.cycleImgData.shape):
      Context.cycleImgData = torch.empty(defFieldShape, device=device)
    else:
      Context.cycleImgData.detach_()
    return Context.cycleImgData
  else:
    return torch.empty(defFieldShape, device=device)

def getLoss10(vecFieldShape, device):
  if useropts.useContext:
    if (Context.loss10 is None) or (vecFieldShape != Context.loss10.shape):
      Context.loss10 = torch.zeros(vecFieldShape, device=device, requires_grad=False)
    else:
      Context.loss10.detach_()
    return Context.loss10
  else:
    return torch.zeros(vecFieldShape, device=device, requires_grad=False)

def getLoss11(vecFieldShape, device):
  if useropts.useContext:
    if (Context.loss11 is None) or (vecFieldShape != Context.loss11.shape):
      Context.loss11 = torch.zeros(vecFieldShape, device=device, requires_grad=False)
    else:
      Context.loss11.detach_()  
    return Context.loss11
  else:
    return torch.zeros(vecFieldShape, device=device, requires_grad=False)

def getLoss20(vecFieldShape, device):
  if useropts.useContext:
    if (Context.loss20 is None) or (vecFieldShape != Context.loss20.shape):
      Context.loss20 = torch.zeros(vecFieldShape, device=device, requires_grad=False)
    else:
      Context.loss20.detach_()    
    return Context.loss20
  else:
    return torch.zeros(vecFieldShape, device=device, requires_grad=False)

def getLoss21(vecFieldShape, device):
  if useropts.useContext:
    if (Context.loss21 is None) or (vecFieldShape != Context.loss21.shape):
      Context.loss21 = torch.zeros(vecFieldShape, device=device, requires_grad=False)
    else:
      Context.loss21.detach_()
    return Context.loss21
  else:
    return torch.zeros(vecFieldShape, device=device, requires_grad=False)

def getLoss30(vecFieldShape, device):
  if useropts.useContext:
    if (Context.loss30 is None) or (vecFieldShape != Context.loss30.shape):
      Context.loss30 = torch.zeros(vecFieldShape, device=device, requires_grad=False)
    else:
      Context.loss30.detach_()
    return Context.loss30
  else:
    return torch.zeros(vecFieldShape, device=device, requires_grad=False)

def getLoss31(vecFieldShape, device):
  if useropts.useContext:
    if (Context.loss31 is None) or (vecFieldShape != Context.loss31.shape):
      Context.loss31 = torch.zeros(vecFieldShape, device=device, requires_grad=False)
    else:
      Context.loss31.detach_()
    return Context.loss31
  else:
    return torch.zeros(vecFieldShape, device=device, requires_grad=False)

def smoothArray3D(inputArray, device, nrOfFilters=2, variance = 2, kernelSize = 5):
    smoothing = GaussianSmoothing(1, kernelSize, variance, 3, device)
    padVal = int(kernelSize / 2)
    input = inputArray[None, None, ]
    for i in range(0,nrOfFilters):
      input = torch.nn.functional.pad(input, (padVal,padVal,padVal,padVal,padVal,padVal))
      input = smoothing(input)
    return input[0,0]
  
def getMaxIdxs(imgShape, imgPatchSize):
  if type(imgPatchSize) is list or type(imgPatchSize) is tuple:
    return getMaxIdxsTuple(imgShape, imgPatchSize)
  else:
    return getMaxIdxsScalar(imgShape, imgPatchSize)  
  
def getMaxIdxsScalar(imgShape, imgPatchSize):
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

def getMaxIdxsTuple(imgShape, imgPatchSizes):
  if imgShape[2] - imgPatchSizes[0] > 0:
    maxidx0 = imgShape[2] - imgPatchSizes[0] + 1
  else:
    maxidx0 = 1
  
  if imgShape[3] - imgPatchSizes[1] > 0:
    maxidx1 = imgShape[3] - imgPatchSizes[1] + 1
  else:
    maxidx1 =  1
  
  if imgShape[4] - imgPatchSizes[2] > 0:
    maxidx2 = imgShape[4] - imgPatchSizes[2] + 1
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
    
  return [patchSize0, patchSize1, patchSize2]

def deformImage(imgToDef, defFields, device, detach=True):
  zeroDefField = getZeroDefField(imgToDef.shape, device)
  currDefField = torch.empty(zeroDefField.shape, device=device, requires_grad=False)
  if (detach):
    ## our image are of the form z,y,x; grid_sample takes as input the order x,y,z
    currDefField[..., 0] = zeroDefField[..., 0] + defFields[:, 0, ].detach() / ((imgToDef.shape[4]-1) / 2.0)
    currDefField[..., 1] = zeroDefField[..., 1] + defFields[:, 1, ].detach() / ((imgToDef.shape[3]-1) / 2.0)
    currDefField[..., 2] = zeroDefField[..., 2] + defFields[:, 2, ].detach() / ((imgToDef.shape[2]-1) / 2.0)
  else:
    currDefField[..., 0] = zeroDefField[..., 0] + defFields[:, 0, ] / ((imgToDef.shape[4]-1) / 2.0)
    currDefField[..., 1] = zeroDefField[..., 1] + defFields[:, 1, ] / ((imgToDef.shape[3]-1) / 2.0)
    currDefField[..., 2] = zeroDefField[..., 2] + defFields[:, 2, ] / ((imgToDef.shape[2]-1) / 2.0)
  deformedTmp = torch.nn.functional.grid_sample(imgToDef, currDefField, mode='bilinear', padding_mode='border')
  return deformedTmp

def getReceptiveFieldOffset(nuOfLayers):
  receptiveFieldOffset = np.power(2,nuOfLayers)
  for i in range(nuOfLayers-1,0,-1):
    receptiveFieldOffset += 2*np.power(2,i)
  return receptiveFieldOffset

def loadImage(filename):
  tmp = sitk.ReadImage(str(filename))
  img0 = sitk.GetArrayFromImage(tmp)
  img0 = np.expand_dims(img0, axis=0)
  img0 = np.expand_dims(img0, axis=0)
  img0 = torch.from_numpy(img0)
  return img0

def saveImg(imgData, filename, isVector = False):
    itkImg = sitk.GetImageFromArray(imgData, isVector=isVector)
    sitk.WriteImage(itkImg,filename)
    
def normalizeImg(img):
  imgMean = img.mean()
  imgData = img - imgMean
  imgStd = imgData.std()
  imgData = imgData / imgStd
  return imgData
    
def compareDicts(dict1, dict2):
  shared_items = {k: dict1[k] for k in dict1 if k in dict2 and bool(torch.all(dict1[k] == dict2[k]))}
  return len(shared_items)  

def printHash(obj):
  h=pickle.dumps(obj)
  print hash(h)
  
def numpyRepeat(obj, times):
  objShape = obj.shape
  tmp = obj.view(objShape[0],-1).repeat(1,times).view(objShape[0]*times,objShape[1],objShape[2],objShape[3])
  return tmp
  
def sampleImgData(data, samplingRate):
    imgDataOrig = data['image']
    labelDataOrig = data['label']
    maskDataOrig = data['mask']
    landmarkData = data['landmarks']
    if samplingRate < 1:
      imgData = torch.nn.functional.interpolate(imgDataOrig,scale_factor=samplingRate,mode='trilinear')
      if (maskDataOrig.dim() == imgDataOrig.dim()):
        maskDataOrig = maskDataOrig.float()
        maskData = torch.nn.functional.interpolate(maskDataOrig, scale_factor=samplingRate, mode='nearest')
        maskData = maskData.byte()
      else:
        maskData = maskDataOrig
      if (labelDataOrig.dim() == imgDataOrig.dim()):
        labelDataOrig = labelDataOrig.float()
        labelData = torch.nn.functional.interpolate(labelDataOrig, scale_factor=samplingRate, mode='nearest')
        labelData = labelData.byte()
      else:
        labelData = labelDataOrig
    else:
      imgData = imgDataOrig
      maskData = maskDataOrig
      labelData = labelDataOrig
    
    return (imgData, maskData, labelData, landmarkData)
  
def sampleImg(img, samplingRate):
  if samplingRate != 1.0:
    img = torch.nn.functional.interpolate(img,scale_factor=samplingRate,mode='trilinear')
  return img
  
def getPaddedData(imgData, maskData, labelData, padVals):
  imgData = torch.nn.functional.pad(imgData, padVals, "constant", 0)
  if (maskData.dim() == imgData.dim()):
    maskData = maskData.float()
    maskData = torch.nn.functional.pad(maskData, padVals, "constant", 0)
    maskData = maskData.byte()
  if (labelData.dim() == imgData.dim()):
    labelData = labelData.float()
    labelData = torch.nn.functional.pad(labelData, padVals, "constant", 0)
    labelData = labelData.byte()
    
  return (imgData, maskData, labelData)

def save_grad(name):
  
  def hook(grad):
      print(name, np.float64(torch.sum(grad)))

  return hook
