# OneShotImageRegistration
One Shot Deformable Medical Image Registration

a description of this work can be found here: https://arxiv.org/abs/1907.04641

**to start a registration call:**
python src/OnePatchShot.py --trainingFiles=PATH_TO_INPUT_CSV_FILE  --outputPath=OUTPUT_PATH

the input csv file contains a list of folders; each folder represents a separate registration task; the image files in the folders must follow the following naming scheme: 

first image: img0.FILE_FORMAT
second image: img1.FILE_FORMAT
...

the currently supported FILE_FORMATs are .nrrd and .nii.gz

an example csv file and image folder can be found in the resources folder

the input folders can also contain segmentations or landmarks. segmentations and landmarks will be deformed with the deformation fields calculated for the appropriate images. the naming convention is as follows: 

img0.nrrd mask0.nrrd 00.pts
img1.nrrd mask1.nrrd 10.pts

the .pts format is described here: https://www.creatis.insa-lyon.fr/rio/dir_validation_data



a more detailed description and manual will follow as soon as possible; in case of any questions feel free to contact me
