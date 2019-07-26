# OneShotImageRegistration
One Shot Deformable Medical Image Registration

a description of this work can be found here: https://arxiv.org/abs/1907.04641

**to start a registration call:**
python src/OnePatchShot.py --trainingFiles=PATH_TO_INPUT_CSV_FILE  --outputPath=OUTPUT_PATH

the input csv file contains a list of folders; each folder represents a separate registration task; the image files in the folders must follow the following naming scheme: 

* first image: img0.FILE_FORMAT
* second image: img1.FILE_FORMAT
* ...

the currently supported FILE_FORMATs are .nrrd and .nii.gz

an example csv file and image folder can be found in the resources folder

the input folders can also contain segmentations and/or masks. landmarks will be deformed with the deformation fields calculated for the appropriate images. the masks are used to define the region of interest and to reduce memory consumption. the naming convention is as follows: 

* img0.nrrd mask0.nrrd 00.pts
* img1.nrrd mask1.nrrd 10.pts

the .pts format is described here: https://www.creatis.insa-lyon.fr/rio/dir_validation_data

pytorch v0.4.1 was used for development and testing

### Loss Visualisation

in the optimisation process a loss file is constantly updated with the current value of the loss function. The file is named lossLog.csv and saved in the output folder. With the following command the loss progress can be visualised:  

python src/Visualize.py --csvFile=OUTPUT_FOLDER/lossLog.csv


a more detailed description and manual will follow as soon as possible; in case of any questions feel free to contact me
