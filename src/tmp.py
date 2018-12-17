from medpy.io import load, save
from Utils import deform, getDefField
import numpy as np
import LossFunctions as lf
import torch
from HeadAndNeckDataset import saveData
import SimpleITK as sitk

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

print(defFieldNii.mean())

defFieldNii[:,:,:,0] = defFieldNii[:,:,:,0] / (defFieldNii.shape[0]/2)
defFieldNii[:,:,:,1] = defFieldNii[:,:,:,1] / (defFieldNii.shape[1]/2)
defFieldNii[:,:,:,2] = defFieldNii[:,:,:,2] / (defFieldNii.shape[2]/2)
defFieldNiiTmp = defFieldNii
defFieldNii = np.moveaxis(defFieldNii, 3, 0)
defFieldNii = np.expand_dims(defFieldNii, axis=0)


m0=np.linspace(-1, 1, imgNii.shape[0], dtype=np.float32)

m1=np.linspace(-1, 1, imgNii.shape[1], dtype=np.float32)

m2=np.linspace(-1, 1, imgNii.shape[2], dtype=np.float32)

grid0, grid1, grid2 = np.meshgrid(m0,m1,m2,indexing='ij')


imgNii = np.expand_dims(imgNii, axis=0)
imgNii = np.expand_dims(imgNii, axis=0)


defField0 = np.stack([grid2, grid1, grid0], axis=3)

defField = defField0 + defFieldNiiTmp

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

# svf = lf.smoothnessVecField(defFieldNii)

deformed0 = deform(imgNii[0,0,:], defFieldNii[0,0,:,:,:],defFieldNii[0,1,:,:,:],defFieldNii[0,2,:,:,:])
#np.sum(deformed0-refImgNii)
save(deformedTmp[0,0,:].numpy(), 'img0Def.nii.gz', imgHeader)

defX = defFieldNii[0, 0 * 3,].detach() * (defFieldNii.shape[2]/2)
defY = defFieldNii[0, 0 * 3 + 1,].detach() * (defFieldNii.shape[3]/2)
defZ = defFieldNii[0, 0 * 3 + 2,].detach() * (defFieldNii.shape[4]/2)
imgToDeform = sitk.GetImageFromArray(getDefField(defX, defY, defZ),isVector=True)
sitk.WriteImage(imgToDeform, 'test123.nrrd')
# saveData(getDefField(defX, defY, defZ), 'deformdationFieldDataset' + str(0) + 'image' + str(0)+ 'channel' + str(0) + '.nii.gz')
