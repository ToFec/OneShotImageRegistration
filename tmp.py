from medpy.io import load
from Utils import deform
import numpy as np


imgpath = '/home/fechter/Bilder/4DRegistration/Datasets/Data00/img0.nii.gz'
defFieldpath = '/home/fechter/Bilder/4DRegistration/Datasets/Data00/def0.nii.gz'
refPath = '/home/fechter/Bilder/4DRegistration/Datasets/Data00/img1.nii.gz'

imgNii, imgHeader = load(imgpath)
refImgNii, refImgHeader = load(refPath)
defFieldNii, defFieldHeader = load(defFieldpath)

deformed0 = deform(imgNii, defFieldNii[:,:,:,0],defFieldNii[:,:,:,1],defFieldNii[:,:,:,2])
deformed1 = deform(refImgNii, defFieldNii[:,:,:,0],defFieldNii[:,:,:,1],defFieldNii[:,:,:,2])
np.sum(deformed1-imgNii)