# Deep-Learning-Breast-FGT
The u-net deep learning python code is used to segment the breast FGT and fat tissue. The FGT % is calculated automatically from a test image of a 2D input breast MRI scan image
--------------
1. creating mask
--------------
Matlab scripts developed in our lab are in the Mask_Generator_hand_draw folder. 
The program works for any dicom images. The user can first apply threshold to show the ROI. Next the user can use the cursor to draw the outline of the non-ROI noise and remove from the ROI images. Finally, the user can save the mask of the ROI. User can use this program to generate FGT mask, and the whole breast mask.

Steps 1: read dicom images
|Dicom breast MRI image|
|:-:|
|<img src = "https://github.com/rispoli-lab/Deep-Learning-Breast-FGT/Pictures/mask1.png" >| 



--------------
2. Traing model
--------------
TrainingLoop.py can be used to tain model with training data.
User only need to enter the directories path in the begining lines of the script
To use differenct depth of U-net model, just change line 7 of TrainingLoop.py. For example, "from unet_d3 import UNet" is to use the Depth 3 U-net model. unet_d4.py is depth 4 U-net model, unet_d5.py is depth 5 U-net model.

--------------
3. Validate model
--------------
The directory path of the validation dataset can be entered by the user at the biginning part of TrainingLoop.py. The model at every epoch is validate, the model with the best loss from validation is return to the testing model.  

--------------
4. Testing model
--------------
TrainingLoop.py automatically test the model with test dataset. The directory of the test dataset path is entered by the user at the begining part of the script. The Fibroglandular tissue percentage (FGT%) in breast and dice coefficient of the predicted versus labels are automatically calculated and saved in the file name of the output images.
