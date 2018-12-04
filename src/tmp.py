from medpy.io import load, save
from Utils import deform
import numpy as np


imgpath = '/home/fechter/workspace/TorchSandbox/resources/Data00/img0.nii.gz'
defFieldpath = '/home/fechter/workspace/TorchSandbox/resources/Data00/def0.nii.gz'
refPath = '/home/fechter/workspace/TorchSandbox/resources/Data00/img1.nii.gz'

imgNii, imgHeader = load(imgpath)
refImgNii, refImgHeader = load(refPath)
defFieldNii, defFieldHeader = load(defFieldpath)

deformed0 = deform(imgNii, defFieldNii[:,:,:,0],defFieldNii[:,:,:,1],defFieldNii[:,:,:,2])
np.sum(deformed0-refImgNii)
save(deformed0, 'img0Def.nii.gz', imgHeader)
