# Deep-Learning-Breast-FGT
The u-net deep learning python code is used to segment the breast FGT and whole breast. The FGT % is calculated automatically from 2D breast MRI scan images.
--------------
## 1. Creating mask
--------------
Matlab scripts developed in our lab are in the Mask_Generator_hand_draw folder. 
The program works for any dicom images. The user can first apply threshold to show the ROI. Next the user can use the cursor to draw the outline of the non-ROI noise and remove from the ROI images. Finally, the user can save the mask of the ROI. User can use this program to generate FGT mask, and the whole breast mask.

### Steps 1: Read dicom image:
<img src = "https://github.com/rispoli-lab/Deep-Learning-Breast-FGT/blob/master/Pictures/mask1.png" >

### Steps 2: Generating mask for whole breast:
<img src = "https://github.com/rispoli-lab/Deep-Learning-Breast-FGT/blob/master/Pictures/mask2.png" >

### Steps 3: Finished mask for whole breast:
<img src = "https://github.com/rispoli-lab/Deep-Learning-Breast-FGT/blob/master/Pictures/mask3.png" >

### Steps 4: Gnerating mask for Fibroglandular tissue(FGT):
<img src = "https://github.com/rispoli-lab/Deep-Learning-Breast-FGT/blob/master/Pictures/mask4.png" >

### Steps 5: Remove skin, muscle and image artifacts from the the FGT maks:
<img src = "https://github.com/rispoli-lab/Deep-Learning-Breast-FGT/blob/master/Pictures/mask4_5.png" >

<img src = "https://github.com/rispoli-lab/Deep-Learning-Breast-FGT/blob/master/Pictures/mask5-6.png" >

Keep outlining and remove skin, muscle and image artifacts from the the FGT mask...
### Steps 6: Finished mask for breast FGT:
<img src = "https://github.com/rispoli-lab/Deep-Learning-Breast-FGT/blob/master/Pictures/mask7.png" >


--------------
## 2. Training model
--------------
TrainingLoop.py can be used to tain model with training data.
### Load training data, validating data and testing data
User only need to enter the directories path in the begining lines of the script.\
"TrainImgDir" is the folder storing the MRI images for training.\
"Train_class1Dir" is the folder storing the binary mask images for class 1. Here they are the FGT mask images.\
"Train_class2Dir" is the folder storing the binary mask images for class 2. Here they are the whole breast mask images.\
"EvalImgDir" is the folder storing the MRI images for validating.\
"Eval_class1Dir" is the folder storing the binary mask images for class 1 for validating. \
"Eval_class2Dir" is the folder storing the binary mask images for class 2 for validating.\
"TestImgDir" is the folder storing the testing MRI images. \
"Test_class1Dir" is the folder storing the binary mask images for class 1 for testing.\
"Test_calss2Dir" is the folder storing the binary mask images for class 2 for testing.
### Enter folder to save predicted images
"Pred_Test_class1Dir" is the folder where the prediciton of class 1 mask images will be saved.\
"Pred_Test_class2Dir" is the folder where the prediction of class 2 mask images will be saved.\
when the testing process is finished, user can compare the predicted mask images in "Pred_Test_class1Dir" and "Pred_Test_class2Dir" with the groud truth maks images in 
"Test_class1Dir" and "Test_calss2Dir".

### Select U-net model
To use differenct depth of U-net model, just change line 7 of TrainingLoop.py. For example, "from unet_d3 import UNet" is to use the Depth 3 U-net model. unet_d4.py is depth 4 U-net model, unet_d5.py is depth 5 U-net model.

--------------
## 3. Validating model
--------------
The directories path of the validation dataset can be entered by the user at the biginning part of TrainingLoop.py. The model at every epoch is validated, the model with the best loss from the validation is returned to for testing model.  

--------------
## 4. Testing model
--------------
TrainingLoop.py automatically test the model with test dataset. The directory of the test dataset path is entered by the user at the begining part of the script. The Fibroglandular tissue percentage (FGT%) in breast and dice coefficient of the predicted versus labels are automatically calculated and saved in the file name of the output images.
