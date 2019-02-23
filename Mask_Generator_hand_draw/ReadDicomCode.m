clc
close all
clear all 
 %% image
       addpath('E:\Deep Learning Seg\Data Repository\Breast Seg Cancer Imaging Archive')
       filename = sprintf('000095.dcm');
       X = dicomread(filename);
       X = uint8(X);
       figure(1)
       imagesc(X);
       title('before threshold');
       axis tight; axis equal;colormap('gray');
 %%      
        currentFolder = pwd;
        Flo = find(currentFolder =='\');
        currentFolder(currentFolder== '\') = '_'; 
        currentFolder = currentFolder(Flo(5)+1:end);
        savelocation = 'E:\Deep Learning Seg\Training Image\Train_Images\';
        outputFileName = [savelocation,currentFolder,'_', filename,'.tif'];
        imwrite(X,outputFileName)
 %% Threshold breast Fat   

 %% Label breast fat
       [cropedMask] = handraw_whole(X);
       figure(12);
       imagesc(cropedMask)
       title('All breast tissue');
       axis tight; axis equal
 %% Save Breast Fat
        savelocation = 'E:\Deep Learning Seg\Training Image\Class_2_Breast_Whole\';
        outputFileName = [savelocation,currentFolder,'_', filename,'Fat.tif'];
        imwrite(cropedMask,outputFileName)
        disp('Breast Fat image saved')
 %% Treshold breast FGT   
       Thr = 55;
       mask = X;
       mask(X>=Thr ) = intmax('uint8');
       mask(X<Thr ) = intmin('uint8');
       mask = uint8(mask);
       figure(2)
       imagesc(mask)
       title('after threshod');
       axis tight; axis equal

  
 %% Modify FGT until satisfied
       close(figure(2))
       [~,mask] = handraw(mask,intmax('uint8'));
       figure(13);
       imagesc(mask)
       title('Breast FGT after mask editing');
       axis tight; axis equal
     %% Save FGT label
        savelocation = 'E:\Deep Learning Seg\Training Image\Class_1_FGT\';
        outputFileName = [savelocation,currentFolder,'_', filename,'FGT.tif'];
        imwrite(mask,outputFileName)
        disp('Breast FGT image saved')
       
       