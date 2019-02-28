import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as img
import torch.utils.data as data_utils
from unet_d3 import UNet  # depth of U-net change here
from torchsummary import summary
from loss import dice_loss, dice_coef, FGT_coe
from collections import defaultdict
import torch.optim as optim
from torch.optim import lr_scheduler
import os
import time
import copy
from skimage import color

TrainImgDir = '/home/jack/Documents/trainingdata_2classes_2_25/Train_Images/'
Train_class1Dir = '/home/jack/Documents/trainingdata_2classes_2_25/Class_1_FGT/'
Train_class2Dir = '/home/jack/Documents/trainingdata_2classes_2_25/Class_2_Breast_Whole/'

EvalImgDir = '/home/jack/Documents/trainingdata_2classes_2_25/Eval_Images/'
Eval_class1Dir = '/home/jack/Documents/trainingdata_2classes_2_25/Eval_Class_1/'
Eval_class2Dir = '/home/jack/Documents/trainingdata_2classes_2_25/Eval_Class_2/'
Eval_datasize = 10

Pred_Eval_class1Dir = '/home/jack/Documents/trainingdata_2classes_2_25/Prediction_Eval_Class_1/'
Pred_Eval_class2Dir = '/home/jack/Documents/trainingdata_2classes_2_25/Prediction_Eval_Class_2/'

TestImgDir = '/home/jack/Documents/trainingdata_2classes_2_25/Test_Images/'
Test_class1Dir = '/home/jack/Documents/trainingdata_2classes_2_25/Test_Class_1/'
Test_class2Dir = '/home/jack/Documents/trainingdata_2classes_2_25/Test_Class_2/'
Test_datasize = 30

Pred_Test_class1Dir = '/home/jack/Documents/trainingdata_2classes_2_25/Prediction_Test_Class_1/'
Pred_Test_class2Dir =  '/home/jack/Documents/trainingdata_2classes_2_25/Prediction_Test_Class_2/'

def importImg(TrainImgDir,Train_class1Dir,Train_class2Dir):
    Train_dataset_list = []
    Train_Class1_list = []
    Train_Class2_list = []

    for datafilename in sorted(os.listdir(TrainImgDir)):
        Train_dataset_list.append(datafilename)

    for labelfilename in sorted(os.listdir(Train_class1Dir)):
        Train_Class1_list.append(labelfilename)

    for labelfilename in sorted(os.listdir(Train_class2Dir)):
        Train_Class2_list.append(labelfilename)


    Train_dataset = torch.zeros(len(Train_dataset_list), 1, 256, 256)
    Train_Class1 = torch.zeros(len(Train_dataset_list), 256, 256)
    Train_Class2 = torch.zeros(len(Train_dataset_list), 256, 256)

    for file_index in range(len(Train_dataset_list)):
        Train_dataset[file_index, 0, :, :] = torch.tensor(img.imread(TrainImgDir + Train_dataset_list[file_index]))

        Train_label = img.imread(Train_class1Dir + Train_Class1_list[file_index])
        if Train_label.max() > 0:
            Train_label = Train_label/Train_label.max()
        Train_Class1[file_index, :, :] = torch.tensor(Train_label)

        Train_label = img.imread(Train_class2Dir + Train_Class2_list[file_index])[:, :, 0]
        if Train_label.max() > 0:
            Train_label = Train_label/Train_label.max()
        Train_Class2[file_index, :, :] = torch.tensor(Train_label)

    Train_labels = torch.empty(size=(len(Train_dataset_list), 2, 256, 256))
    Train_labels[:, 0, :, :] = Train_Class1
    Train_labels[:, 1, :, :] = Train_Class2
    return Train_dataset, Train_labels

def save_imgs(best_img,colormask_GT,Test_dataset,Pred_classDir_1,figIndex,tresh = 0.5):
    img.imsave(os.path.join(Pred_classDir_1 + 'best_' + str(figIndex) + '.tif'), best_img)
    best_img[best_img >= tresh] = 1
    best_img[best_img < tresh] = 0
    colormask_pred = best_img
    colormasks = np.zeros((256, 256, 3))
    mask_red = (colormask_pred == 1) & (colormask_GT == 0)
    mask_blue = (colormask_pred == 0) & (colormask_GT == 1)
    mask_yellow = (colormask_pred == 1) & (colormask_GT == 1)
    colormasks[mask_red] = [1,0,0]
    colormasks[mask_blue] = [0,0,1]
    colormasks[mask_yellow] = [1,1,0]
    BG_norm = Test_dataset[figIndex, 0, :, :] / Test_dataset.max()
    img_color = np.dstack((BG_norm, BG_norm, BG_norm))
    img_hsv = color.rgb2hsv(img_color)
    color_mask_hsv = color.rgb2hsv(colormasks)
    img_hsv[..., 0] = color_mask_hsv[..., 0]
    img_hsv[..., 1] = color_mask_hsv[..., 1] * 0.6  # alpha = 0.6
    img_masked = color.hsv2rgb(img_hsv)
    dice_coe = dice_coef(colormask_pred,colormask_GT)
    img.imsave(os.path.join(Pred_classDir_1 + 'best_overlay' + str(figIndex) +'_dice_'+ str("{:4f}".format(dice_coe))+'_.tif'), img_masked)

    return dice_coe


def calc_loss(pred, target, metrics, bce_weight=0.5):
    bce = F.binary_cross_entropy_with_logits(pred, target)
    pred = torch.sigmoid(pred)
    dice = dice_loss(pred, target)

    loss = bce * bce_weight + dice * (1 - bce_weight)

    metrics['bce'] += bce.data.cpu().numpy() * target.size(0)
    metrics['dice'] += dice.data.cpu().numpy() * target.size(0)
    metrics['loss'] += loss.data.cpu().numpy() * target.size(0)

    return loss

def print_metrics(metrics, epoch_samples, phase):
    outputs = []
    for k in metrics.keys():
        outputs.append("{}: {:4f}".format(k, metrics[k] / epoch_samples))

    print("{}: {}".format(phase, ", ".join(outputs)))

def train_model(model, optimizer, scheduler, num_epochs=25):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1e10

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        since = time.time()

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                for param_group in optimizer.param_groups:
                    print("LR", param_group['lr'])

                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            metrics = defaultdict(float)
            epoch_samples = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.cuda()
                labels = labels.cuda()

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = calc_loss(outputs, labels, metrics)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                epoch_samples += inputs.size(0)

            if phase == 'val':
                pred_test = torch.sigmoid(outputs)
                pred_test = pred_test.data.cpu().numpy()
                for figIndex in range(6):
                    img.imsave(os.path.join(Pred_Eval_class1Dir +'epoch_'+str(epoch)+'_'+str(figIndex)+'.tif'), pred_test[figIndex,0,:,:])
                    img.imsave(os.path.join(Pred_Eval_class2Dir + 'epoch_' + str(epoch) + '_' + str(figIndex) + '.tif'), pred_test[figIndex, 1, :, :])

            print_metrics(metrics, epoch_samples, phase)
            epoch_loss = metrics['loss'] / epoch_samples

            # deep copy the model
            if phase == 'val' and epoch_loss < best_loss:
                print("saving best model")
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                best_img = pred_test

        time_elapsed = time.time() - since
        print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    print('Best val loss: {:4f}'.format(best_loss))
    dice_coe_class1 = 0
    dice_coe_class2 = 0
    for figIndex in range(Eval_datasize):
        dice_coe = save_imgs(best_img[figIndex, 0, :, :], Eval_labels[figIndex, 0, :, :].data.cpu().numpy(), Eval_dataset, Pred_Eval_class1Dir, figIndex)
        dice_coe_class1 += dice_coe
        dice_coe = save_imgs(best_img[figIndex, 1, :, :], Eval_labels[figIndex, 1, :, :].data.cpu().numpy(), Eval_dataset, Pred_Eval_class2Dir, figIndex)
        dice_coe_class2 += dice_coe
        print('Eval results: Predicted FGT% of figure {} : {:4f}'.format(figIndex,FGT_coe(best_img[figIndex, 0, :, :], best_img[figIndex,1,:,:])))
        print('Eval results: Real FGT% of figure {} : {:4f}'.format(figIndex,FGT_coe(Eval_labels[figIndex, 0, :, :].data.cpu().numpy(), Eval_labels[figIndex, 1, :, :].data.cpu().numpy())))
    print('Eval results: Best val class 1 dice coef: {:4f}, class 2 dice coef: {:4f}'.format(dice_coe_class1/Eval_datasize, dice_coe_class2/Eval_datasize))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

# import training data, eval data, and test data from folders
Train_dataset, Train_labels = importImg(TrainImgDir,Train_class1Dir,Train_class2Dir)
Eval_dataset, Eval_labels = importImg(EvalImgDir,Eval_class1Dir,Eval_class2Dir)
Test_dataset, Test_labels = importImg(TestImgDir,Test_class1Dir,Test_class2Dir)

# load data into dataloader
train = data_utils.TensorDataset(Train_dataset, Train_labels)
Eval = data_utils.TensorDataset(Eval_dataset, Eval_labels)
dataloaders = { 'train' : data_utils.DataLoader(train, batch_size = 15, shuffle = True),
                'val'   : data_utils.DataLoader(Eval, batch_size = Eval_datasize, shuffle = False)
                }

# train loop for training data
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # only for CPU or one PGU
model = UNet(n_class=2).cuda() # for 2 GPUs
gpus = [0,1]
model = torch.nn.DataParallel(model, device_ids=gpus)
summary(model, input_size=(1, 256, 256))
optimizer_ft = optim.Adam(model.parameters(),lr=1e-4)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=30, gamma=0.1)
inputs, masks = next(iter(dataloaders['train']))
print(inputs.shape, masks.shape)
model = train_model(model, optimizer_ft, exp_lr_scheduler, num_epochs=100)

# test model
Test = data_utils.TensorDataset(Test_dataset, Test_labels)
Test_dataloader =  data_utils.DataLoader(Test, batch_size = Test_datasize, shuffle = False)
inputs, labels = next(iter(Test_dataloader))
pred = model(inputs)
pred_test = torch.sigmoid(pred)
pred_test = pred_test.data.cpu().numpy()
dice_coe_class1 = np.zeros(Test_datasize)
dice_coe_class2 = np.zeros(Test_datasize)
print('--- Test ---\n')
for figIndex in range(Test_datasize):
    dice_coe = save_imgs(pred_test[figIndex, 0, :, :], Test_labels[figIndex, 0, :, :].data.cpu().numpy(), Test_dataset, Pred_Test_class1Dir, figIndex)
    dice_coe_class1[figIndex]= dice_coe
    dice_coe = save_imgs(pred_test[figIndex, 1, :, :], Test_labels[figIndex, 1, :, :].data.cpu().numpy(), Test_dataset, Pred_Test_class2Dir, figIndex)
    dice_coe_class2[figIndex]= dice_coe
    print('Test results: Predicted FGT% of figure {} : {:4f}'.format(figIndex, FGT_coe(pred_test[figIndex, 0, :, :], pred_test[figIndex, 1, :, :])))
    print('Test results: Real FGT% of figure {} : {:4f}'.format(figIndex, FGT_coe(Test_labels[figIndex, 0, :, :].data.cpu().numpy(), Test_labels[figIndex, 1, :, :].data.cpu().numpy())))
print('Test results subject 1: mean class 1 dice coef: {:4f}, class 2 dice coef: {:4f} '.format(dice_coe_class1[0:9].sum()/10, dice_coe_class2[0:9].sum()/10))
print('Test results subject 2: mean class 1 dice coef: {:4f}, class 2 dice coef: {:4f} '.format(dice_coe_class1[10:19].sum()/10, dice_coe_class2[10:19].sum()/10))
print('Test results subject 3: mean class 1 dice coef: {:4f}, class 2 dice coef: {:4f} '.format(dice_coe_class1[20:29].sum()/10, dice_coe_class2[20:29].sum()/10))
