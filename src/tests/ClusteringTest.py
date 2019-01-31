import sys
import Utils
import Options
import numpy as np
import torch
import Optimize
from Net import UNet
#import kmeans #https://github.com/ilyaraz/pytorch_kmeans

class ClusterTest():
  
  # takes too much time to compute
  def kMeansTest(self):
    imgFileName0 = '/home/fechter/workspace/TorchSandbox/resources/PopiTmp/img0.nrrd'
    maskFileName0 = '/home/fechter/workspace/TorchSandbox/resources/PopiTmp/mask0.nrrd'
    
    img0 = Utils.loadImage(imgFileName0)
    mask0 = Utils.loadImage(maskFileName0)
    
    maskChanSum = torch.sum(mask0, 1)
    idxs = np.where(maskChanSum > 0)
    
    maxNumberOfPixs = Options.patchSize * Options.patchSize * Options.patchSize * img0.shape[1] + 1
    maskPixs = len(idxs[0])
    clusterData = np.transpose(np.asarray(idxs)).astype(np.float32)
    nClusterStart = np.int(maskPixs / maxNumberOfPixs)
    
    
    dataset = torch.from_numpy(clusterData).to(Options.device)
    print('Starting clustering')
    centers, codes = kmeans.cluster(dataset, nClusterStart)
    print(centers)
    print(codes)
    
    return True
  
  def clusterCenterErosionTest(self):
    imgFileName0 = '/home/fechter/workspace/TorchSandbox/resources/PopiTmp/img0.nrrd'
    maskFileName0 = '/home/fechter/workspace/TorchSandbox/resources/PopiTmp/mask0.nrrd'
    
    img0 = Utils.loadImage(imgFileName0)
    mask0 = Utils.loadImage(maskFileName0)
    
    maskChanSum = torch.sum(mask0, 1)
    
    
    patchSizes = Utils.getPatchSize(img0.shape, Options.patchSize)
    maxIdxs = Utils.getMaxIdxs(img0.shape, Options.patchSize)
    idxs = []
    for patchIdx0 in range(0, maxIdxs[0], patchSizes[0]/2):
      for patchIdx1 in range(0, maxIdxs[1], patchSizes[1]/2):
        for patchIdx2 in range(0, maxIdxs[2], patchSizes[2]/2):
          if (maskChanSum[0,patchIdx0:patchIdx0 + patchSizes[0], patchIdx1:patchIdx1 + patchSizes[1], patchIdx2:patchIdx2 + patchSizes[2]].median() > 0):
            idxs.append( (patchIdx0, patchIdx1, patchIdx2) )
    
    leftover0 = img0.shape[2] % patchSizes[0]
    startidx0 = img0.shape[2] - patchSizes[0] if (leftover0 > 0) & (maxIdxs[0] > patchSizes[0])  else 0
    leftover1 = img0.shape[3] % patchSizes[1]
    startidx1 = img0.shape[3] - patchSizes[1] if (leftover1 > 0) & (maxIdxs[1] > patchSizes[1])  else 0
    leftover2 = img0.shape[4] % patchSizes[2]
    startidx2 = img0.shape[4] - patchSizes[2] if (leftover2 > 0) & (maxIdxs[2] > patchSizes[2])  else 0
    
    if (startidx2 + startidx1 + startidx0 > 0) :               
      for patchIdx0 in range(startidx0, maxIdxs[0], patchSizes[0]/2):
        for patchIdx1 in range(startidx1, maxIdxs[1], patchSizes[1]/2):
          for patchIdx2 in range(startidx2, maxIdxs[2], patchSizes[2]/2):
            if (maskChanSum[0,patchIdx0:patchIdx0 + patchSizes[0], patchIdx1:patchIdx1 + patchSizes[1], patchIdx2:patchIdx2 + patchSizes[2]].median() > 0):
              idxs.append( (patchIdx0, patchIdx1, patchIdx2) )        
    
    net = UNet()
    optimizer = Optimize.Optimize(net, Options)
    otherIdxs = optimizer.getIndicesForUniformSampling(mask0,img0,Options.patchSize)
    
    if len(otherIdxs) == len(idxs):
      return True
    else:
      return False
#     Utils.saveImg(resultImg, '/home/fechter/workspace/TorchSandbox/resources/PopiTmp/clusterCentersMedian.nrrd', False)

def main(argv):
  ctest = ClusterTest()
  
  result = ctest.clusterCenterErosionTest()
  if (result):
    print('tests passed')
  else:
    print('tests failed')

if __name__ == '__main__':
    main(sys.argv[1:]) 