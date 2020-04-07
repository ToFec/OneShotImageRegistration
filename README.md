# OneShotImageRegistration
One Shot Deformable Medical Image Registration

a description of this work can be found here: https://arxiv.org/abs/1907.04641 and https://ieeexplore.ieee.org/document/8989991

if you use this work please cite:
T. Fechter and D. Baltas, "One Shot Learning for Deformable Medical Image Registration and Periodic Motion Tracking," in IEEE Transactions on Medical Imaging.

**to start a registration call:**
python src/OnePatchShot.py --trainingFiles=PATH_TO_INPUT_CSV_FILE  --outputPath=OUTPUT_PATH

_the refactored version of the code allows the user to use the net not only in a one-shot fashion but also to train and test the network in a conventional way, to fine-tune existing models (for a few-shot approach) and to enable diffeomorphic registration. a detailed description how to use those features will follow soon._

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

python 2.7 and pytorch v0.4.1 were used for development and testing

### Loss Visualisation

in the optimisation process a loss file is constantly updated with the current value of the loss function. The file is named lossLog.csv and saved in the output folder. With the following command the loss progress can be visualised:  

python src/Visualize.py --csvFile=OUTPUT_FOLDER/lossLog.csv


a more detailed description and manual will follow as soon as possible; in case of any questions feel free to contact me
