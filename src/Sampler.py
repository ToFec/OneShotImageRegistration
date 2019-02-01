import torch
from Utils import getMaxIdxs, getPatchSize, normalizeImg
import numpy as np

class Sampler(object):

  def __init__(self, maskData, imgData, labelData, imgPatchSize):
    
    if (maskData.dim() != imgData.dim()):
      maskData = torch.ones(imgData.shape, dtype=torch.int8)

    self.maskChanSum = torch.sum(maskData, 1)
    self.imgData = imgData
    self.labelData = labelData
    self.maxIdxs = getMaxIdxs(imgData.shape, imgPatchSize)
    self.patchSizes = getPatchSize(imgData.shape, imgPatchSize)

  def getRandomSubSamples(self, numberofSamplesPerRun, idxs, currIteration=0, normImgPatch=False):
   
    imgDataNew = torch.empty((numberofSamplesPerRun, self.imgData.shape[1], self.patchSizes[0], self.patchSizes[1], self.patchSizes[2]), requires_grad=False)
    if (self.labelData.dim() == self.imgData.dim()):
      labelDataNew = torch.empty((numberofSamplesPerRun, self.imgData.shape[1], self.patchSizes[0], self.patchSizes[1], self.patchSizes[2]), requires_grad=False)
    
    randSampleIdxs = np.random.randint(0, len(idxs[0]), (numberofSamplesPerRun,))
    for j in range(0, numberofSamplesPerRun):
      idx0 = idxs[0][randSampleIdxs[j]]
      idx2 = idxs[1][randSampleIdxs[j]]
      idx3 = idxs[2][randSampleIdxs[j]]
      idx4 = idxs[3][randSampleIdxs[j]]
      imgPatch = self.imgData[idx0, : , idx2:idx2 + self.patchSizes[0], idx3:idx3 + self.patchSizes[1], idx4:idx4 + self.patchSizes[2]]
      if normImgPatch:
        imgPatch = normalizeImg(imgPatch)
      imgDataNew[j, ] = imgPatch
  #     indexArrayTest[idx2:idx2 + patchSizes[0], idx3:idx3 + patchSizes[1], idx4:idx4 + patchSizes[2]] += 1
      if (self.labelData.dim() == self.imgData.dim()):
        labelDataNew[j, ] = self.labelData[idx0, : , idx2:idx2 + self.patchSizes[0], idx3:idx3 + self.patchSizes[1], idx4:idx4 + self.patchSizes[2]]
    
    imgDataToWork = imgDataNew
    if (self.labelData.dim() == self.imgData.dim()):
      labelDataToWork = labelDataNew
    else:
      labelDataToWork = torch.Tensor();
      
    return (imgDataToWork, labelDataToWork)    
  
  def getSubSample(self, idx, normImgPatch):
      imgPatch = self.imgData[:, :, idx[0]:idx[0] + self.patchSizes[0], idx[1]:idx[1] + self.patchSizes[1], idx[2]:idx[2] + self.patchSizes[2]]
      labelData = torch.empty((1, self.imgData.shape[1], self.patchSizes[0], self.patchSizes[1], self.patchSizes[2]), requires_grad=False)
      if normImgPatch:
        imgPatch = normalizeImg(imgPatch)
      if (self.labelData.dim() == self.imgData.dim()):
        labelData = self.labelData[:, :, idx[0]:idx[0] + self.patchSizes[0], idx[1]:idx[1] + self.patchSizes[1], idx[2]:idx[2] + self.patchSizes[2]]
      return (imgPatch, labelData)
    
  def getUniformlyDistributedSubsamples(self,numberofSamplesPerRun, idxs, currIteration, normImgPatch=False):
    
    startIdx = currIteration % numberofSamplesPerRun
    
    
    imgDataNew = torch.empty((numberofSamplesPerRun, self.imgData.shape[1], self.patchSizes[0], self.patchSizes[1], self.patchSizes[2]), requires_grad=False)
    if (self.labelData.dim() == self.imgData.dim()):
      labelDataNew = torch.empty((numberofSamplesPerRun, self.imgData.shape[1], self.patchSizes[0], self.patchSizes[1], self.patchSizes[2]), requires_grad=False)
    
    iterationRange = np.arange(startIdx,startIdx+numberofSamplesPerRun)
    iterationRange[iterationRange >= len(idxs)] = iterationRange[iterationRange >= len(idxs)] - len(idxs)
    j = 0
    for i in iterationRange:
      idx = idxs[i]
      imgPatch = self.imgData[:, :, idx[0]:idx[0] + self.patchSizes[0], idx[1]:idx[1] + self.patchSizes[1], idx[2]:idx[2] + self.patchSizes[2]]
      if normImgPatch:
        imgPatch = normalizeImg(imgPatch)
      imgDataNew[j, ] = imgPatch
      if (self.labelData.dim() == self.imgData.dim()):
        labelDataNew[j, ] = self.labelData[:, :, idx[0]:idx[0] + self.patchSizes[0], idx[1]:idx[1] + self.patchSizes[1], idx[2]:idx[2] + self.patchSizes[2]]
      j=j+1
      
    imgDataToWork = imgDataNew
    if (self.labelData.dim() == self.imgData.dim()):
      labelDataToWork = labelDataNew
    else:
      labelDataToWork = torch.Tensor();
      
    return (imgDataToWork, labelDataToWork)  
  
  def getIndicesForRandomization(self):
    maskChanSumCrop = self.maskChanSum[:, 0:self.maxIdxs[0], 0:self.maxIdxs[1], 0:self.maxIdxs[2]]
    idxs = np.where(maskChanSumCrop > 0)
  
    return idxs
  
  def getIndicesForOneShotSampling(self, receptiveFieldSize):
    return self.getIndicesForUniformSamplingPathShift(receptiveFieldSize)  
  
  def iterateImg(self, startidx, shift):
    idxs = []
    for patchIdx0 in range(startidx[0], self.maxIdxs[0], shift[0]):
      for patchIdx1 in range(startidx[1], self.maxIdxs[1], shift[1]):
        for patchIdx2 in range(startidx[2], self.maxIdxs[2], shift[2]):
          if (self.maskChanSum[:,patchIdx0:patchIdx0 + self.patchSizes[0], patchIdx1:patchIdx1 + self.patchSizes[1], patchIdx2:patchIdx2 + self.patchSizes[2]].median() > 0):
            idxs.append( (patchIdx0, patchIdx1, patchIdx2) )
    return idxs
            
            
  def getIndicesForUniformSamplingPathShift(self, patchShift):
    imgShape = self.imgData.shape

    idxs = self.iterateImg((0,0,0), patchShift)
       
    leftover0 = (imgShape[2] - self.patchSizes[0]) % patchShift[0]
    startidx0 = imgShape[2] - self.patchSizes[0] if (leftover0 > 0) & (self.maxIdxs[0] > self.patchSizes[0])  else 0
    leftover1 = (imgShape[3] -self.patchSizes[1]) % patchShift[1] 
    startidx1 = imgShape[3] - self.patchSizes[1] if (leftover1 > 0) & (self.maxIdxs[1] > self.patchSizes[1])  else 0
    leftover2 = (imgShape[4] - self.patchSizes[2]) % patchShift[2] 
    startidx2 = imgShape[4] - self.patchSizes[2] if (leftover2 > 0) & (self.maxIdxs[2] > self.patchSizes[2])  else 0
    
    if startidx0 > 0:
      idxs.append( self.iterateImg((startidx0, 0, 0), patchShift))
    if startidx1 > 0:
      idxs.append( self.iterateImg((0, startidx1, 0), patchShift))
    if startidx2 > 0:
      idxs.append( self.iterateImg((0, 0, startidx2), patchShift))
      
    return idxs
    
  def getIndicesForUniformSampling(self):
    return self.getIndicesForUniformSamplingPathShift(self.patchSizes / 2)
    