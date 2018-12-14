from medpy.io import load, save
from Utils import deform
import numpy as np
import LossFunctions as lf
import torch


imgpath = '/home/fechter/workspace/TorchSandbox/resources/Data00/img0.nii.gz'
defFieldpath = '/home/fechter/workspace/TorchSandbox/resources/Data00/def0.nii.gz'
refPath = '/home/fechter/workspace/TorchSandbox/resources/Data00/img1.nii.gz'

torch.manual_seed(0)
np.random.seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

imgNii, imgHeader = load(imgpath)
refImgNii, refImgHeader = load(refPath)
defFieldNii, defFieldHeader = load(defFieldpath)
defFieldNii = np.moveaxis(defFieldNii, 3, 0)
defFieldNii = np.expand_dims(defFieldNii, axis=0)


m0=np.linspace(-1, 1, imgNii.shape[0], dtype=np.float32)

m1=np.linspace(-1, 1, imgNii.shape[1], dtype=np.float32)

m2=np.linspace(-1, 1, imgNii.shape[2], dtype=np.float32)

grid0, grid1, grid2 = np.meshgrid(m0,m1,m2,indexing='ij')


imgNii = np.expand_dims(imgNii, axis=0)
imgNii = np.expand_dims(imgNii, axis=0)


defField = np.stack([grid2, grid1, grid0], axis=3)
defField = np.expand_dims(defField, axis=0)
defField = torch.from_numpy(defField)

refImgNii = np.expand_dims(refImgNii, axis=0)
refImgNii = np.expand_dims(refImgNii, axis=0)

imgNii = torch.from_numpy(imgNii)
refImgNii = torch.from_numpy(refImgNii)
defFieldNii = torch.from_numpy(defFieldNii)

#https://pytorch.org/docs/stable/nn.html?highlight=grid

deformedTmp = torch.nn.functional.grid_sample(imgNii, defField, mode='bilinear', padding_mode='border')

diffImg = imgNii.sub(deformedTmp)
print(diffImg.max())
print(diffImg.min())

cc0 = lf.normCrossCorr(imgNii, imgNii)
cc1 = lf.normCrossCorr(imgNii, deformedTmp)

svf = lf.smoothnessVecField(defFieldNii)

deformed0 = deform(imgNii, defFieldNii[0,:,:,:],defFieldNii[1,:,:,:],defFieldNii[2,:,:,:])
np.sum(deformed0-refImgNii)
save(deformed0, 'img0Def.nii.gz', imgHeader)
