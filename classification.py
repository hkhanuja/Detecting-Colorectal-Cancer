#!/usr/bin/env python
# coding: utf-8




import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision.models as models
import torchvision.datasets as datasets
import torchvision
from torch.utils.data import Dataset, DataLoader
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms
import sys
import time
import math
from matplotlib.patches import Circle
import tqdm
from torch.optim.lr_scheduler import MultiStepLR,StepLR, ReduceLROnPlateau
from scipy.ndimage.morphology import grey_dilation
from statistics import mean, median, mode
from torch.autograd import Variable
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, classification_report, accuracy_score, confusion_matrix
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
import torchvision.models as models
import random
from sklearn.model_selection import train_test_split
from torch.utils.data.sampler import SubsetRandomSampler
import GPUtil

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"




c_dict = {0:'cancer',1:'non_cancer'}

# Check if GPU is available
if torch.cuda.is_available():
    print("The code will run on GPU.")
else:
    print("The code will run on CPU.")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


pos_wsis = sorted(os.listdir('/scratch/Preprocessed_DigestPath/patches/tissue_train_pos-v1/'))


pos_wsis = [ele for ele in pos_wsis if '.csv' not in ele]

pos_patients = [ele.split('-lv1-')[0] for ele in pos_wsis]

pos_patients = sorted(list(set(pos_patients)))

validation_patients = pos_patients[-25:-6]
test_patients = pos_patients[-6:]
train_patients = []
for ele in pos_patients:
    if ele not in validation_patients and ele not in test_patients:
        train_patients.append(ele)

print("No. of patients in test set:",len(test_patients))
print("No. of patients in validation set:",len(validation_patients))
print("No. of patients in training set:",len(train_patients))

test_files = [ele for ele in pos_wsis if ele.split('-lv1-')[0] in test_patients]

validation_files = [ele for ele in pos_wsis if ele.split('-lv1-')[0] in validation_patients]

train_files = [ele for ele in pos_wsis if ele.split('-lv1-')[0] in train_patients]



#Getting the path to patches for training and validation data
X_train = []
y_train = []
# cpatches = 0
ncpatches = 0
for file in tqdm.tqdm(train_files):
    cancer_patches = '/scratch/Preprocessed_DigestPath/patches/tissue_train_pos-v1/' + file + '/cancerous_patches/'
#     cpatches += len(os.listdir(cancer_patches))
    for patch in sorted(os.listdir(cancer_patches)):
        if 'mask' not in patch:
            img_path = '/scratch/Preprocessed_DigestPath/patches/tissue_train_pos-v1/' + file + '/cancerous_patches/' + patch
            X_train.append(img_path)
            y_train.append(0)
    
    
    noncancer_patches = '/scratch/Preprocessed_DigestPath/patches/tissue_train_pos-v1/' + file + '/non_cancerous_patches/'
    for patch in sorted(os.listdir(noncancer_patches)):        
        img_path = '/scratch/Preprocessed_DigestPath/patches/tissue_train_pos-v1/' + file + '/non_cancerous_patches/' + patch
        if ncpatches>=25000:
            break
        X_train.append(img_path)
        y_train.append(1)
        ncpatches +=1
print(ncpatches)


X_val = []
y_val = []
# cpatches = 0
ncpatches = 0
for file in tqdm.tqdm(validation_files):
    cancer_patches = '/scratch/Preprocessed_DigestPath/patches/tissue_train_pos-v1/' + file + '/cancerous_patches/'
#     cpatches += len(os.listdir(cancer_patches))
    for patch in os.listdir(cancer_patches):
        if 'mask' not in patch:
            img_path = '/scratch/Preprocessed_DigestPath/patches/tissue_train_pos-v1/' + file + '/cancerous_patches/' + patch
            X_val.append(img_path)
            y_val.append(0)
    
    
    noncancer_patches = '/scratch/Preprocessed_DigestPath/patches/tissue_train_pos-v1/' + file + '/non_cancerous_patches/'
    for patch in sorted(os.listdir(noncancer_patches)):        
        img_path = '/scratch/Preprocessed_DigestPath/patches/tissue_train_pos-v1/' + file + '/non_cancerous_patches/' + patch
        if ncpatches>=6000:
            break
        X_val.append(img_path)
        y_val.append(1)
        ncpatches += 1 

print(ncpatches)

ncpatches = 0
X_test = []
y_test = []
for file in tqdm.tqdm(test_files):
    cancer_patches = '/scratch/Preprocessed_DigestPath/patches/tissue_train_pos-v1/' + file + '/cancerous_patches/'
#     cpatches += len(os.listdir(cancer_patches))
    for patch in os.listdir(cancer_patches):
        if 'mask' not in patch:
            img_path = '/scratch/Preprocessed_DigestPath/patches/tissue_train_pos-v1/' + file + '/cancerous_patches/' + patch
            X_test.append(img_path)
            y_test.append(0)

    noncancer_patches = '/scratch/Preprocessed_DigestPath/patches/tissue_train_pos-v1/' + file + '/non_cancerous_patches/'
    for patch in sorted(os.listdir(noncancer_patches)):        
        img_path = '/scratch/Preprocessed_DigestPath/patches/tissue_train_pos-v1/' + file + '/non_cancerous_patches/' + patch
        if ncpatches>=1000:
            break
        X_test.append(img_path)
        y_test.append(1)
        ncpatches += 1 

        
print("No. of patches in test set:",len(X_test))
print("No. of patches in validation set:",len(X_val))
print("No. of patches in training set:",len(X_train))

def common(a,b): 
    c = [value for value in a if value in b] 
    return c
d=common(train_patients,validation_patients)
print('train and validation:',d)

d=common(train_patients,test_patients)
print('train and test:',d)

d=common(validation_patients,test_patients)
print('validation and test:',d)




image_shape =(224,224)
batch_size = 16



train_transform=transforms.Compose([
         transforms.RandomHorizontalFlip(),
         transforms.ToTensor(),
         transforms.RandomResizedCrop(size=224,scale=(0.8,1.0)),   
         #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

valid_transform=transforms.Compose([
         transforms.ToTensor(),
         #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

test_transform=transforms.Compose([
         transforms.ToTensor(),
         #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])




class DatasetLoader(Dataset):
    def __init__(self,X,Y,transforms):
        self.X = X
        self.Y = Y
        self.transform = transforms
     

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self,idx):
        img_path = self.X[idx]
        label = self.Y[idx]
        image = cv2.imread(img_path)
        if self.transform:
            transformed_image = self.transform((torchvision.transforms.functional.to_pil_image(image)))
        del image
        return (transformed_image),label





train_classification_dataset = DatasetLoader(X_train,y_train,train_transform)
train_dataloader = DataLoader(train_classification_dataset,batch_size=batch_size,shuffle=True)
val_classification_dataset = DatasetLoader(X_val,y_val,valid_transform)
valid_dataloader = DataLoader(val_classification_dataset,batch_size=batch_size)

test_classification_dataset = DatasetLoader(X_test,y_test,test_transform)
test_dataloader = DataLoader(test_classification_dataset,batch_size=1)





patches,label = next(iter(train_dataloader))
print(torch.max(patches))
print(type(label))
print(label.shape)
print(patches.shape)





class resnet50(nn.Module):
    def __init__(self):
        super(resnet50,self).__init__()
        self.model = models.resnet50(pretrained=True)    
        self.sigmoid = nn.Sigmoid()
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs,1)
    def forward(self, x):
        x = self.model(x)
        x = self.sigmoid(x)
        return x
freeze_layers = True





model = resnet50()





if freeze_layers:
    for name,param in model.named_parameters():
        if 'layer4' in name or 'fc' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
        





def train_model(model, criterion, optimizer,lr_scheduler, train_loader,valid_loader,train_len,val_len,batch_size, num_epochs=20):
    
    since = time.time()
    val_acc_history = []
    train_loss_history = []
    train_acc_history = []
    val_loss_history = []
    print('training') 
    print(GPUtil.showUtilization())   
    model = model.cuda()

    for epoch in range(num_epochs):
        model.train()
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            k=0
            if phase == 'train':
                model.train()  # Set model to training mode
                dataloader=train_loader
                number_of_images = train_len
            else:
                model.eval()   # Set model to evaluate mode
                dataloader=valid_loader
                number_of_images = val_len


            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in iter(dataloader):
                
                k += inputs.shape[0]
                if k%10000==0:
                    print(k,'/',number_of_images)
                 
                inputs = inputs.to(device,dtype= torch.float32)
                labels = torch.unsqueeze(labels,1).to(device,dtype= torch.float32)
                
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                
                if phase == 'train':
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                else:
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)


                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                out = (outputs>0.5).float()
                running_corrects += torch.sum(out == labels.data)
                


            epoch_loss = running_loss / (number_of_images)
            epoch_acc = (running_corrects.double() / (number_of_images))* 100

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
           
        
            #Saving the train,validation and test logs
            if phase == 'train':
                train_loss_history.append(epoch_loss)
                train_acc_history.append(epoch_acc)
                print('train_loss:',epoch_loss,'train_accuracy:',epoch_acc.detach().item())
                
            
            else:
                val_acc_history.append(epoch_acc)
                val_loss_history.append(epoch_loss)
                print('val_loss:',epoch_loss,'val_accuracy:',epoch_acc.detach().item())

            
            
            #Early-Stopping
#             if phase == 'val' and epoch_loss < min_val_loss:
#                 epochs_no_improve = 0
#                 min_val_loss = epoch_loss
#             elif phase == 'val':
#                 epochs_no_improve += 1
                
            #To exit the phase loop
#             if phase == 'val' and epoch > 5 and epochs_no_improve == n_epochs_stop:
#                 print('-----------Early stopping!--------------' )
#                 early_stop = True
#                 break 
#             else:
#                 continue


        
        #To exit the epoch loop
#         if early_stop:
#             break


    if lr_scheduler is not None:
        lr_scheduler.step()


    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    #print('Best val Acc: {:4f}'.format(best_acc))
    
    
    history = {"val_acc_history": val_acc_history, "val_loss_history": val_loss_history, "train_loss_history": train_loss_history, "train_acc_history": train_acc_history }
    return model, history





opt = "sgd" #optimizer
lr = 0.001  #the initial learning rate
momentum = 0.9  #the momentum for sgd
weight_decay = 1e-5 #the weight decay
lr_scheduler = "step"        #['step', 'exp', 'stepLR', 'fix', 'cos'] the learning rate schedule
gamma = 0.1 #learning rate scheduler parameter for step and exp
steps = "10"  #the learning rate decay for step and stepLR





if opt == 'sgd':
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=lr,
                               momentum=momentum, weight_decay=weight_decay)
elif opt == 'adam':
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr,
                                weight_decay=weight_decay)
else:
    raise Exception("optimizer not implement")





if lr_scheduler == 'step':
    steps_list = [int(step) for step in steps.split(',')]
    lr_scheduler_fn = optim.lr_scheduler.MultiStepLR(optimizer, steps_list, gamma=gamma)
elif lr_scheduler == 'exp':
    lr_scheduler_fn = optim.lr_scheduler.ExponentialLR(optimizer, gamma)
elif lr_scheduler == 'stepLR':
    steps = int(steps)
    lr_scheduler_fn = optim.lr_scheduler.StepLR(optimizer, steps, gamma)
elif lr_scheduler == 'cos':
    steps = int(steps)
    lr_scheduler_fn = optim.lr_scheduler.CosineAnnealingLR(optimizer, steps, 0)
elif lr_scheduler == 'fix':
    lr_scheduler_fn = None
else:
    raise Exception("lr schedule not implement")





criterion = nn.BCELoss()





model,history = train_model(model, criterion, optimizer, lr_scheduler_fn, train_dataloader,valid_dataloader,len(train_classification_dataset),len(val_classification_dataset),batch_size, num_epochs=25)


 # Disable grad
with torch.no_grad():

    running_corrects = 0
    model.eval()
    
    for inputs, labels in iter(test_dataloader):
        inputs = inputs.to(device,dtype= torch.float32)
        labels = torch.unsqueeze(labels,1).to(device,dtype= torch.float32)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        out = (outputs>0.5).float()
        running_corrects += torch.sum(out == labels.data)
        print(f'Prediction: {c_dict[int(out)]} - Actual target: {c_dict[int(labels)]}')
    print('Test accuracy:',running_corrects/len(test_classification_dataset).detach().item())
# In[ ]:


dest_model_weights = "/home/swathiguptha/colon_cancer_dataset/resnet_model_weights/patient_separated/"


# In[ ]:


with open(dest_model_weights+'/train_loss_ep50', 'w') as f:
    f.write(' '.join(map(str,history['train_loss_history'])))
with open(dest_model_weights+'/valid_loss_50ep', 'w') as f:
    f.write(' '.join(map(str,history['val_loss_history'])))
with open(dest_model_weights+'/train_acc_50ep', 'w') as f:
    f.write(' '.join(map(str,history['train_acc_history'])))
with open(dest_model_weights+'/valid_acc_50ep', 'w') as f:
    f.write(' '.join(map(str,history['val_acc_history'])))


# In[ ]:


torch.save(model,dest_model_weights+'/resnet_50epochs.h5')

