# Deep-Learning-Breast-FGT
The u-net deep learning python code is used to segment the breast FGT and fat tissue. The FGT % is calculated automatically from a test image of a 2D input breast MRI scan image
--------------
1. creating mask
--------------
Matlab scripts developed in our lab are in the Mask_Generator_hand_draw folder. 
For any 256 by 256 dicom images, the user can apply threshold and recursive handraw and removing ROI strategies to obtain a ground truth for FGT mask, and the whole breast mask
