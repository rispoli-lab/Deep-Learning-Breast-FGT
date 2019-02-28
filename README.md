# Deep-Learning-Breast-FGT
The u-net deep learning python code is used to segment the breast FGT and fat tissue. The FGT % is calculated automatically from a test image of a 2D input breast MRI scan image
--------------
1. creating mask
--------------
Matlab scripts developed in our lab are in the Mask_Generator_hand_draw folder. 
For any 256 by 256 dicom images, the user can apply threshold and recursive handraw and removing ROI strategies to obtain a ground truth for FGT mask, and the whole breast mask

--------------
2. Traing model
--------------
TrainingLoop.py can be used to tain model with training data.
User only need to enter the directories path in the begining lines of the script
To use differenct depth of U-net model, just change line 7 of TrainingLoop.py. For example, "from unet_d3 import UNet" is to use the Depth 3 U-net model. unet_d4.py is depth 4 U-net model, unet_d5.py is depth 5 U-net model.

--------------
2. Validate model
--------------
The directory path of the validation dataset can be entered by the user at the biginning part of TrainingLoop.py. The model at every epoch is validate, the model with the best loss from validation is return to the testing model.  

--------------
4. Testing model
--------------
TrainingLoop.py automatically test the model with test dataset. The directory of the test dataset path is entered by the user at the begining part of the script.
