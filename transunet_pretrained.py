#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import glob
import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, models, transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
import torch.optim as optim
from torch.autograd import Variable
from time import time
from torch import Tensor
from torch.utils.data.sampler import SubsetRandomSampler
from torch.optim import lr_scheduler
import tqdm
import gc
import time
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split
import random
from collections import OrderedDict




m1 = timm.create_model('vit_base_patch16_384',pretrained='True')




pos_wsis = sorted(os.listdir('/scratch/Preprocessed_DigestPath/patches/tissue_train_pos-v1/'))




print(len(pos_wsis))





pos_wsis = [ele for ele in pos_wsis if '.csv' not in ele]
print(len(pos_wsis))




pos_patients = [ele.split('-lv1-')[0] for ele in pos_wsis]
print(len(pos_patients))






pos_patients = set(pos_patients)
print(len(pos_patients))




validation_patients = random.sample(pos_patients, 19)
train_patients = [ele for ele in pos_patients if ele not in validation_patients]




print("No. of patients in validation set:",len(validation_patients))
print("No. of patients in training set:",len(train_patients))




validation_files = [ele for ele in pos_wsis if ele.split('-lv1-')[0] in validation_patients]




train_files = [ele for ele in pos_wsis if ele.split('-lv1-')[0] in train_patients]




print("No. of WSIs in validation:",len(validation_files))
print("No. of WSIs in training:",len(train_files))




def common(a,b): 
    c = [value for value in a if value in b] 
    return c
d=common(train_patients,validation_patients)
print(d)




image_shape =(224,224)
batch_size = 6


# In[19]:


X_train = []
y_train = []


# In[20]:


# cpatches = 0
# ncpatches = 0
for file in tqdm.tqdm(train_files):
    cancer_patches = '/scratch/Preprocessed_DigestPath/patches/tissue_train_pos-v1/' + file + '/cancerous_patches/'
#     cpatches += len(os.listdir(cancer_patches))
    for patch in os.listdir(cancer_patches):
        if 'mask' not in patch:
            img_path = '/scratch/Preprocessed_DigestPath/patches/tissue_train_pos-v1/' + file + '/cancerous_patches/' + patch
            mask_patch = patch.split('.')[0] + '_mask.png'
            mask_path = '/scratch/Preprocessed_DigestPath/patches/tissue_train_pos-v1/' + file + '/cancerous_patches/' + mask_patch
            X_train.append(img_path)
            y_train.append(mask_path)
    
    
    noncancer_patches = '/scratch/Preprocessed_DigestPath/patches/tissue_train_pos-v1/' + file + '/non_cancerous_patches/'
#     ncpatches += len(os.listdir(noncancer_patches))
    for patch in os.listdir(noncancer_patches):        
        img_path = '/scratch/Preprocessed_DigestPath/patches/tissue_train_pos-v1/' + file + '/non_cancerous_patches/' + patch
        mask_path = None
        X_train.append(img_path)
        y_train.append(mask_path)


# In[21]:


X_test = []
y_test = []


# In[22]:


for file in tqdm.tqdm(validation_files):
    cancer_patches = '/scratch/Preprocessed_DigestPath/patches/tissue_train_pos-v1/' + file + '/cancerous_patches/'
    for patch in os.listdir(cancer_patches):
        if 'mask' not in patch:
            img_path = '/scratch/Preprocessed_DigestPath/patches/tissue_train_pos-v1/' + file + '/cancerous_patches/' + patch
            mask_patch = patch.split('.')[0] + '_mask.png'
            mask_path = '/scratch/Preprocessed_DigestPath/patches/tissue_train_pos-v1/' + file + '/cancerous_patches/' + mask_patch
            X_test.append(img_path)
            y_test.append(mask_path)
    
    
    noncancer_patches = '/scratch/Preprocessed_DigestPath/patches/tissue_train_pos-v1/' + file + '/non_cancerous_patches/'
    for patch in os.listdir(noncancer_patches):        
        img_path = '/scratch/Preprocessed_DigestPath/patches/tissue_train_pos-v1/' + file + '/non_cancerous_patches/' + patch
        mask_path = None
        X_test.append(img_path)
        y_test.append(mask_path)


# In[23]:


print("Total no. of patches in Training data:",len(y_train))
print("Total no. of patches in Testing data:",len(y_test))


# In[24]:


class DatasetLoader(Dataset):
    def __init__(self,X,Y,transforms):
        self.X = X
        self.Y = Y
        self.transform = transforms
     

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self,idx):
        img_path = self.X[idx]
        mask_path = self.Y[idx]
        label = 0 
        if mask_path is None:
            mask = np.zeros((224,224))
            label = np.array([1,-1]) #Non-Cancerous
        else:
            mask = cv2.imread(mask_path,cv2.IMREAD_GRAYSCALE)
            label = np.array([0,1]) #Cancerous
        image = cv2.imread(img_path)
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            transformed_image = transformed['image']
            transformed_mask = transformed['mask']
        del image,mask
        return (transformed_image/255.),(transformed_mask/255.),label


# In[25]:


train_transform=A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomResizedCrop(width=224, height=224,scale=(0.8, 1.2),p=0.5),
        ToTensorV2(),
         #transforms.RandomHorizontalFlip(),
         # transforms.ToTensor(),
         #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

val_transform=A.Compose([
        ToTensorV2(),
         #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


# In[26]:


train_classification_dataset = DatasetLoader(X_train,y_train,train_transform)
train_dataloader = DataLoader(train_classification_dataset,batch_size=batch_size,shuffle=True)
test_classification_dataset = DatasetLoader(X_test,y_test,val_transform)
valid_dataloader = DataLoader(test_classification_dataset,batch_size=batch_size)


# In[27]:


patches,masks, labels = next(iter(train_dataloader))
print(labels)
print(torch.max(patches))
print((torch.unique(masks[0])))
print((torch.unique(masks[1])))
print((torch.unique(masks[2])))
print((torch.unique(masks[3])))


:




# In[29]:


# Check if GPU is available
if torch.cuda.is_available():
    print("The code will run on GPU.")
else:
    print("The code will run on CPU.")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


# In[30]:


class Attention(nn.Module):
    def __init__(self,dim,n_heads=12,qkv_bias=False,attn_p=0.,proj_p = 0.):
        super(Attention,self).__init__()
        self.dim = dim                            #dimension of each patch of image
        self.n_heads = n_heads                    #number of heads in MHA
        self.head_dim = dim//n_heads              #the lenght of q,k,v for each patch
        
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(dim,dim*3,bias = qkv_bias)
        self.attn_drop = nn.Dropout(attn_p)
        self.proj = nn.Linear(dim,dim)
        self.proj_drop = nn.Dropout(proj_p)
        
    def forward(self,x):
        n_samples,n_tokens,dim = x.shape
        
        qkv = self.qkv(x)
        qkv = qkv.reshape(n_samples,n_tokens,3,self.n_heads,self.head_dim)
        qkv = qkv.permute(2,0,3,1,4)
        
        q,k,v = qkv[0],qkv[1],qkv[2]
        
        k_t = k.transpose(-2,-1)
        
        dp = (q@k_t)*self.scale
        
        attn = dp.softmax(dim=-1)
        attn = self.attn_drop(attn)
        weighted_avg = attn@v
        weighted_avg = weighted_avg.transpose(1,2)
        weighted_avg = weighted_avg.flatten(2)
        x = self.proj(weighted_avg)
        x = self.proj_drop(x)
        
        return x
        


# In[31]:


class MLP(nn.Module):
    def __init__(self,in_features,hidden_features,out_features,p=0.):
        super(MLP,self).__init__()
        self.fc1 = nn.Linear(in_features,hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features,out_features)
        self.drop = nn.Dropout(p)
        
    def forward(self,x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        
        return x 


# In[32]:


class Block(nn.Module):
    def __init__(self,dim,n_heads,mlp_ratio=4,qkv_bias=True,p=0.,attn_p=0.):
        super(Block,self).__init__()
        self.norm1 = nn.LayerNorm(dim,eps=1e-6)
        self.attn = Attention(dim=dim,n_heads = n_heads,qkv_bias = qkv_bias,attn_p = attn_p,proj_p = p)
        self.norm2 = nn.LayerNorm(dim,eps=1e-6)
        hidden_features = int(dim*mlp_ratio) #3072
        self.mlp = MLP(in_features = dim,hidden_features = hidden_features,out_features=dim)
    
    def forward(self,x):
        
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        
        return x


# In[33]:


def conv_trans_block(in_channels,out_channels):
    conv_trans_block = nn.Sequential(
    nn.Conv2d(in_channels,out_channels,kernel_size=3,padding=1),
    nn.ReLU(inplace=True),
    nn.BatchNorm2d(out_channels)
    )
    return conv_trans_block


# In[34]:




class StdConv2d(nn.Conv2d):

    def forward(self, x):
        w = self.weight
        v, m = torch.var_mean(w, dim=[1, 2, 3], keepdim=True, unbiased=False)
        w = (w - m) / torch.sqrt(v + 1e-5)
        return F.conv2d(x, w, self.bias, self.stride, self.padding,
                        self.dilation, self.groups)


def conv3x3(cin, cout, stride=1, groups=1, bias=False):
    return StdConv2d(cin, cout, kernel_size=3, stride=stride,
                     padding=1, bias=bias, groups=groups)


def conv1x1(cin, cout, stride=1, bias=False):
    return StdConv2d(cin, cout, kernel_size=1, stride=stride,
                     padding=0, bias=bias)


class PreActBottleneck(nn.Module):

    def __init__(self, cin, cout=None, cmid=None, stride=1):
        super().__init__()
        cout = cout or cin
        cmid = cmid or cout//4

        self.gn1 = nn.GroupNorm(32, cmid, eps=1e-6)
        self.conv1 = conv1x1(cin, cmid, bias=False)
        self.gn2 = nn.GroupNorm(32, cmid, eps=1e-6)
        self.conv2 = conv3x3(cmid, cmid, stride, bias=False)  # Original code has it on conv1!!
        self.gn3 = nn.GroupNorm(32, cout, eps=1e-6)
        self.conv3 = conv1x1(cmid, cout, bias=False)
        self.relu = nn.ReLU(inplace=True)

        if (stride != 1 or cin != cout):
            # Projection also with pre-activation according to paper.
            self.downsample = conv1x1(cin, cout, stride, bias=False)
            self.gn_proj = nn.GroupNorm(cout, cout)

    def forward(self, x):

        # Residual branch
        residual = x
        if hasattr(self, 'downsample'):
            residual = self.downsample(x)
            residual = self.gn_proj(residual)

        # Unit's branch
        y = self.relu(self.gn1(self.conv1(x)))
        y = self.relu(self.gn2(self.conv2(y)))
        y = self.gn3(self.conv3(y))

        y = self.relu(residual + y)
        return y

    

class ResNetV2(nn.Module):

    def __init__(self, block_units, width_factor):
        super().__init__()
        width = int(64 * width_factor)
        self.width = width

        self.root = nn.Sequential(OrderedDict([
            ('conv', StdConv2d(3, width, kernel_size=7, stride=2, bias=False, padding=3)),
            ('gn', nn.GroupNorm(32, width, eps=1e-6)),
            ('relu', nn.ReLU(inplace=True)),
            # ('pool', nn.MaxPool2d(kernel_size=3, stride=2, padding=0))
        ]))

        self.body = nn.Sequential(OrderedDict([
            ('block1', nn.Sequential(OrderedDict(
                [('unit1', PreActBottleneck(cin=width, cout=width*4, cmid=width))] +
                [(f'unit{i:d}', PreActBottleneck(cin=width*4, cout=width*4, cmid=width)) for i in range(2, block_units[0] + 1)],
                ))),
            ('block2', nn.Sequential(OrderedDict(
                [('unit1', PreActBottleneck(cin=width*4, cout=width*8, cmid=width*2, stride=2))] +
                [(f'unit{i:d}', PreActBottleneck(cin=width*8, cout=width*8, cmid=width*2)) for i in range(2, block_units[1] + 1)],
                ))),
            ('block3', nn.Sequential(OrderedDict(
                [('unit1', PreActBottleneck(cin=width*8, cout=width*16, cmid=width*4, stride=2))] +
                [(f'unit{i:d}', PreActBottleneck(cin=width*16, cout=width*16, cmid=width*4)) for i in range(2, block_units[2] + 1)],
                ))),
        ]))

    def forward(self, x):
        features = []
        b, c, in_size, _ = x.size()
        x = self.root(x)
        features.append(x)
        x = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)(x)
        for i in range(len(self.body)-1):
            x = self.body[i](x)
            right_size = int(in_size / 4 / (i+1))
            if x.size()[2] != right_size:
                pad = right_size - x.size()[2]
                assert pad < 3 and pad > 0, "x {} should {}".format(x.size(), right_size)
                feat = torch.zeros((b, x.size()[1], right_size, right_size), device=x.device)
                feat[:, :, 0:x.size()[2], 0:x.size()[3]] = x[:]
            else:
                feat = x
            features.append(feat)
        x = self.body[-1](x)
        return x, features[::-1]


# In[35]:


class Embeddings(nn.Module):

    def __init__(self,embed_dim = 768,n_patches=196,p=0.,in_channels=3):
        super(Embeddings, self).__init__()
        self.hybrid_model = ResNetV2(block_units=(3, 4, 9), width_factor=1)
        in_channels = self.hybrid_model.width * 16
        self.patch_embeddings = nn.Conv2d(in_channels=in_channels,
                                       out_channels=embed_dim,
                                       kernel_size=1,
                                       stride=1)
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches, embed_dim))
        self.dropout = nn.Dropout(p=p)


    def forward(self, x):
        x, features = self.hybrid_model(x)
        x = self.patch_embeddings(x)  # (B, hidden. n_patches^(1/2), n_patches^(1/2))
        x = x.flatten(2)
        x = x.transpose(-1, -2)  # (B, n_patches, hidden)
        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings, features


# In[36]:


class transUnet(nn.Module):
    def __init__(self,img_size = 224,patch_size=16,in_channels = 3,n_classes=1,embed_dim = 768,depth=12,n_heads=12,mlp_ratio=4.,qkv_bias=True,p=0.,attn_p=0.):
        super(transUnet,self).__init__()
        
        self.embed_dim = embed_dim
        self.img_size = img_size
        

        
        self.n_patches = (img_size//patch_size) ** 2
        self.embeddings = Embeddings(embed_dim,self.n_patches,p)

        
        self.blocks = nn.ModuleList(
                        [
                            Block(dim = embed_dim,
                                  n_heads=n_heads,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias = qkv_bias,
                                 p=p,
                                 attn_p=attn_p,)
                            
                            for _ in range(depth)
                        ])

        
        #decoder part
        self.deconv1 = conv_trans_block(embed_dim,512)
        
        self.deconv2_1 = conv_trans_block(1024,256)
        self.deconv2_2 = conv_trans_block(256,256)
        
        self.deconv3_1 = conv_trans_block(512,128)
        self.deconv3_2 = conv_trans_block(128,128)
        
        
        self.deconv4_1 = conv_trans_block(192,64)
        self.deconv4_2 = conv_trans_block(64,64)
        
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)
        
        self.prefinal_1 = conv_trans_block(64,16)
        self.prefinal_2 = conv_trans_block(16,16)
        
        self.out = nn.Conv2d(16,1,kernel_size=1)
        
                        
        
    def forward(self,x):
        
        n_samples = x.shape[0]
        


        x = self.embeddings(x)
        projections = x[0]
        features = x[1]

        for block in self.blocks:
            projections = block(projections)
        
        projections = projections.transpose(1,2)
        projections = projections.reshape(n_samples,self.embed_dim,int(self.img_size/16),int(self.img_size/16))
        
        x1 = projections
        
        x1 = self.deconv1(x1)     #(n,512,224/16,224/16)

        x1 = self.upsample(x1)    #(n,512,224/8,224/8)
        x1 = self.deconv2_1(torch.cat([x1,features[0]],1))
        x1 = self.deconv2_2(x1)
        
        
        x1 = self.upsample(x1)    #(n,256,224/4,224/4)

        x1 = self.deconv3_1(torch.cat([x1,features[1]],1))
        x1 = self.deconv3_2(x1)
        
        x1 = self.upsample(x1)    #(n,128,224/2,224/2)
        
        x1 = self.deconv4_1(torch.cat([x1,features[2]],1))
        x1 = self.deconv4_2(x1)
        
        x1 = self.upsample(x1)   #(n,64,224,224)
        x1 = self.prefinal_1(x1)
        x1 = self.prefinal_2(x1)
        
        x1 = self.out(x1)
        x1 = torch.sigmoid(x1)
        return x1


# In[37]:


def dice_metric(inputs, target,labels):
    num = target.size(0)
    
    target1 = target.reshape(num,-1)
    inputs1 = inputs.reshape(num,-1)
    for j in range(num):
        target1 = torch.add(labels[j][0],target1[j],alpha=labels[j][1])
        inputs1 = torch.add(labels[j][0],inputs1[j],alpha=labels[j][1])
    
    intersection = (target1 * inputs1)
    dice = (2. * intersection.sum(1) ) / (target1.sum(1) + inputs1.sum(1) + 1e-8)
    
    dice = dice.sum() / num
    return dice


# In[38]:


def dice_loss(inputs, target,labels):
    diceloss = 1 - dice_metric(inputs, target,labels)
    return diceloss


# In[39]:


def bce_dice_loss(inputs, target,labels):
    dicescore = dice_loss(inputs, target,labels)
    bcescore = nn.BCELoss()
    bceloss = bcescore(inputs, target)

    return bceloss + dicescore


# In[40]:


import time
def train1(model, train_dl, valid_dl,train_len,val_len, optimizer, scheduler,BATCH_SIZE, epochs=1):
    start = time.time()
    model.to(device)

    train_loss, valid_loss,train_dice,valid_dice = [], [] , [], []
    best_acc = 0.0

    for epoch in range(epochs):
        print('Epoch {}/{}'.format(epoch, epochs - 1))
        print('-' * 10)

        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()  # Set trainind mode = true
                dataloader = train_dl
                number_of_images = train_len
                
            else:
                model.eval()  # Set model to evaluate mode
                dataloader = valid_dl
                number_of_images = val_len
                
            print("Phase:",phase)
            running_loss = 0.0
            running_dice_score = 0.0

            step = 0

            # iterate over data
            for x,y,labels in tqdm.tqdm(dataloader):
                #print(x.shape)
                #print(y.shape)
                x = (x).to(device,dtype= torch.float32)
                y = (y).to(device, dtype = torch.float32)
                labels = labels.to(device)
                step += 1
                #print(x.size(),y.size())
                # forward pass
                if phase == 'train':
                    # zero the gradients
                    optimizer.zero_grad()
                    outputs = model(x)
                    
                    loss = bce_dice_loss(outputs.squeeze(1), y.squeeze(1),labels)
                    
     

                    loss.backward()
                    optimizer.step()
                    # scheduler.step()
                    dice_score = dice_metric(outputs.squeeze(1), y.squeeze(1),labels)

                else:
                    with torch.no_grad():
                        outputs = model(x)
                        loss = bce_dice_loss(outputs.squeeze(1), y.squeeze(1),labels)
                        dice_score = dice_metric(outputs.squeeze(1), y.squeeze(1),labels)
                        



                running_loss += loss*x.shape[0] 
                running_dice_score += dice_score*x.shape[0]
                

                gc.collect()

                
            epoch_loss = running_loss / number_of_images
            epoch_dice_score = running_dice_score/number_of_images
            
            if phase=='train':
                train_loss.append(epoch_loss.detach().item())
                train_dice.append(epoch_dice_score.detach().item())
            else:
                valid_loss.append(epoch_loss.detach().item())
                valid_dice.append(epoch_dice_score.detach().item())
            
            if phase == 'valid':
                scheduler.step(epoch_dice_score)

            print('{} Loss: {:.4f} dice_score: {}'.format(phase, epoch_loss, epoch_dice_score))
            
            if epoch == epochs-1 and phase == 'valid':
                destination = '/home/swathiguptha/colon_cancer_dataset/model_weights/pretrain/'
                torch.save({'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    }, destination+'transunet_pretrain_v04.h5')
                
            torch.cuda.empty_cache()

            
            
    time_elapsed = time.time() - start
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))    
    
    return train_loss, valid_loss,train_dice,valid_dice,model


# In[41]:


transunet_model = transUnet(p=0.2,attn_p=0.2)
transunet_model_dict = transunet_model.state_dict()
pretrained_dict = {k: v for k, v in m1.state_dict().items() if k in transunet_model_dict}
transunet_model_dict.update(pretrained_dict)
transunet_model.load_state_dict(transunet_model_dict)


# In[42]:


transunet_model.to(device)
opt = optim.SGD(transunet_model.parameters(), lr = 0.01, momentum = 0.9,weight_decay=0.0001)  
scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, 'min', patience=2)  


# In[43]:


train_loss, valid_loss,train_dice,valid_dice,out_model = train1(transunet_model,train_dataloader,valid_dataloader,len(train_classification_dataset),len(test_classification_dataset),opt,scheduler,batch_size,epochs = 15)


# In[ ]:


with open('/home/swathiguptha/colon_cancer_dataset/model_metrics/pretrain/transunet_pretrain_v04_train_loss.txt', 'w') as f:
    f.write(' '.join(map(str,train_loss)))
with open('/home/swathiguptha/colon_cancer_dataset/model_metrics/pretrain/transunet_pretrain_v04_valid_loss.txt', 'w') as f:
    f.write(' '.join(map(str,valid_loss)))
with open('/home/swathiguptha/colon_cancer_dataset/model_metrics/pretrain/transunet_pretrain_v04_train_dice.txt', 'w') as f:
    f.write(' '.join(map(str,train_dice)))
with open('/home/swathiguptha/colon_cancer_dataset/model_metrics/pretrain/transunet_pretrain_v04_valid_dice.txt', 'w') as f:
    f.write(' '.join(map(str,valid_dice)))






