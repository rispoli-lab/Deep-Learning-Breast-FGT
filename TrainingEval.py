import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as img
import torch.utils.data as data_utils
from unet_d4 import UNet
from torchsummary import summary
from PIL import Image
from loss import dice_loss
import glob
from collections import defaultdict
import torch.optim as optim
from torch.optim import lr_scheduler
import os
import time
import copy

TrainImgDir = '/home/jack/Documents/Train_Test_2_23_2019/Train_Images/'
Train_class1Dir = '/home/jack/Documents/Train_Test_2_23_2019/Train_Class_1_FGT/'
Train_class2Dir = '/home/jack/Documents/Train_Test_2_23_2019/Train_Class_2_Breast_Whole/'

TestImgDir = '/home/jack/Documents/Train_Test_2_23_2019/Test_Images/'
Test_class1Dir = '/home/jack/Documents/Train_Test_2_23_2019/Test_Class_1/'
Test_class2Dir = '/home/jack/Documents/Train_Test_2_23_2019/Test_Class_2/'

Pred_class1Dir = '/home/jack/Documents/Train_Test_2_23_2019/Prediction_Class_1/'
Pred_class2Dir = '/home/jack/Documents/Train_Test_2_23_2019/Prediction_Class_2/'

# import training data from folders
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

#   import testing data from folders
Test_dataset_list = []
Test_Class1_list = []
Test_Class2_list = []

for datafilename in sorted(os.listdir(TestImgDir)):
    Test_dataset_list.append(datafilename)

for labelfilename in sorted(os.listdir(Test_class1Dir)):
    Test_Class1_list.append(labelfilename)

for labelfilename in sorted(os.listdir(Test_class2Dir)):
    Test_Class2_list.append(labelfilename)


Test_dataset = torch.zeros(len(Test_dataset_list), 1, 256, 256)
Test_Class1 = torch.zeros(len(Test_dataset_list), 256, 256)
Test_Class2 = torch.zeros(len(Test_dataset_list), 256, 256)

for file_index in range(len(Test_dataset_list)):
    Test_dataset[file_index, :, :] = torch.tensor(img.imread(TestImgDir + Test_dataset_list[file_index]))

    Test_label = img.imread(Test_class1Dir+Test_Class1_list[file_index])
    if Test_label.max() > 0:
        Test_label = Test_label/Test_label.max()
    Test_Class1[file_index, :, :] = torch.tensor(Test_label)

    Test_label = img.imread(Test_class2Dir+Test_Class2_list[file_index])[:, :, 0]
    if Test_label.max() > 0:
        Test_label = Test_label/Test_label.max()
    Test_Class2[file_index, :, :] = torch.tensor(Test_label)

Train_labels = torch.empty(size=(len(Train_dataset_list), 2, 256, 256))
Train_labels[:,0,:,:] = Train_Class1
Train_labels[:,1,:,:] = Train_Class2

Test_labels = torch.empty(size=(len(Test_dataset_list), 2, 256, 256))
Test_labels[:,0,:,:] = Test_Class1
Test_labels[:,1,:,:] = Test_Class2

# load data into dataloader
train = data_utils.TensorDataset(Train_dataset, Train_labels)
test = data_utils.TensorDataset(Test_dataset, Test_labels)
dataloaders = { 'train' : data_utils.DataLoader(train, batch_size = 5, shuffle = True),
                'val'   : data_utils.DataLoader(test, batch_size = 6, shuffle = False)
                }


def calc_loss(pred, target, metrics, bce_weight=0.5):
    #pred = pred.sum(dim=1)
    #print('target.shape')
    #print(target.shape)
    #print('pred.shape')
    #print(pred.shape)
    bce = F.binary_cross_entropy_with_logits(pred, target)
    pred = F.sigmoid(pred)
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
                #print(outputs.shape)
                pred_test = F.sigmoid(outputs)
                pred_test = pred_test.data.cpu().numpy()
                for figIndex in range(6):
                    img.imsave(os.path.join(Pred_class1Dir +'epoch_'+str(epoch)+'_'+str(figIndex)+'.tif'), pred_test[figIndex,0,:,:])
                    img.imsave(os.path.join(Pred_class2Dir + 'epoch_' + str(epoch) + '_' + str(figIndex) + '.tif'), pred_test[figIndex, 1, :, :])

            print_metrics(metrics, epoch_samples, phase)
            epoch_loss = metrics['loss'] / epoch_samples

            # deep copy the model
            if phase == 'val' and epoch_loss < best_loss:
                print("saving best model")
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())

        time_elapsed = time.time() - since
        print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    print('Best val loss: {:4f}'.format(best_loss))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model




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
model = train_model(model, optimizer_ft, exp_lr_scheduler, num_epochs=60)
