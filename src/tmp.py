from medpy.io import load, save
from Utils import deform
import numpy as np
import LossFunctions as lf
import torch


imgpath = '/home/fechter/workspace/TorchSandbox/resources/Data00/img0.nii.gz'
defFieldpath = '/home/fechter/workspace/TorchSandbox/resources/Data00/def0.nii.gz'
refPath = '/home/fechter/workspace/TorchSandbox/resources/Data00/img1.nii.gz'

imgNii, imgHeader = load(imgpath)
refImgNii, refImgHeader = load(refPath)
defFieldNii, defFieldHeader = load(defFieldpath)
defFieldNii = np.moveaxis(defFieldNii, 3, 0)
defFieldNii = np.expand_dims(defFieldNii, axis=0)

imgNii = np.expand_dims(imgNii, axis=0)
imgNii = np.expand_dims(imgNii, axis=0)

refImgNii = np.expand_dims(refImgNii, axis=0)
refImgNii = np.expand_dims(refImgNii, axis=0)

imgNii = torch.from_numpy(imgNii)
refImgNii = torch.from_numpy(refImgNii)
defFieldNii = torch.from_numpy(defFieldNii)

cc0 = lf.normCrossCorr(imgNii, imgNii)
cc1 = lf.normCrossCorr(imgNii, refImgNii)

svf = lf.smoothnessVecField(defFieldNii)

deformed0 = deform(imgNii, defFieldNii[0,:,:,:],defFieldNii[1,:,:,:],defFieldNii[2,:,:,:])
np.sum(deformed0-refImgNii)
save(deformed0, 'img0Def.nii.gz', imgHeader)
