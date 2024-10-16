#!/usr/bin/env python
# coding: utf-8




import os
import glob
import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, models, transforms
from torch.utils.data.sampler import SubsetRandomSampler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import tqdm
import gc
import time
import timm





m1 = timm.create_model('vit_base_patch16_384',pretrained='True')





#import wandb
#from wandb.keras import WandbCallback
#run = wandb.init(project = 'Colon_Cancer_transunet_v3', name = 'transUnet_v3')





data_dir = "/scratch/colon_cancer_dataset"





patches = np.load(data_dir+"/patches.npy")
masks = np.load(data_dir+"/masks.npy")





print(patches.shape)
print(masks.shape)





image_shape =(224,224)
val_split = .2
batch_size = 8
#positive_tissue_dir=".../input/digestpath-patches-numpy/numpy_folders_digestpath"





from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(patches, masks,test_size=0.2, random_state = 42)





class DatasetLoader(Dataset):
    def __init__(self,X,Y, batch_size,transforms):
        self.X = X
        self.batch_size = batch_size
        self.Y = Y
        self.transform = transforms
    def __len__(self):
        return len(self.X)
    def __getitem__(self,idx):
        x = self.X[idx]
        y = self.Y[idx]
        
        if self.transform:
            x = self.transform(x)
            y = self.transform(y)
        
        return x,y





transform=transforms.Compose([
         #transforms.RandomHorizontalFlip(),
         transforms.ToTensor(),
         #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    





train_classification_dataset = DatasetLoader(X_train,y_train, batch_size,transform)
train_dataloader = DataLoader(train_classification_dataset,batch_size=batch_size,shuffle=True)
test_classification_dataset = DatasetLoader(X_test,y_test, batch_size,transform)
valid_dataloader = DataLoader(test_classification_dataset,batch_size=batch_size)


# In[ ]:


def show_thumbnail(image,thumbnail):
    f, axes = plt.subplots(1, 2, figsize=(10, 10));
    ax = axes.ravel();
    ax[0].imshow(image, cmap='gray');
    ax[0].set_title('Original');
    ax[1].imshow(thumbnail,cmap = 'gray')
    ax[1].set_title('thumbnail')
    #return f





#  Get a batch of training data
patches, masks = next(iter(train_dataloader))
print(patches.shape,masks.shape)


# In[ ]:


for i in range(0,4):
    patch = patches[i]
    mask = masks [i]
    inp = patch.numpy().transpose((1, 2, 0))
    mask = mask.numpy().transpose((1, 2, 0))
    print(np.unique(mask*255))
    #g = show_thumbnail(inp,mask*255)
    #wandb.log({"img": [wandb.Image(g, caption="test")]})


# In[ ]:


## Define network





import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from torchsummary import summary
import torch.optim as optim
from torch.autograd import Variable
from time import time
from torch import Tensor





# Check if GPU is available
if torch.cuda.is_available():
    print("The code will run on GPU.")
else:
    print("The code will run on CPU.")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)





'''
""" Parts of the U-Net model """


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)
    
    
class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

'''





'''
class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        logits = torch.sigmoid(logits)
        return logits
'''





'''
def dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon=1e-6):
    # Average of Dice coefficient for all batches, or for a single mask
    assert input.size() == target.size()
    if input.dim() == 2 and reduce_batch_first:
        raise ValueError(f'Dice: asked to reduce batch but got tensor without batch dimension (shape {input.shape})')

    if input.dim() == 2 or reduce_batch_first:
        inter = torch.dot(input.reshape(-1), target.reshape(-1))
        sets_sum = torch.sum(input) + torch.sum(target)
        if sets_sum.item() == 0:
            sets_sum = 2 * inter

        return (2 * inter + epsilon) / (sets_sum + epsilon)
    else:
        # compute and average metric for each batch element
        dice = 0
        for i in range(input.shape[0]):
            dice += dice_coeff(input[i, ...], target[i, ...])
        return dice / input.shape[0]


def multiclass_dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon=1e-6):
    # Average of Dice coefficient for all classes
    assert input.size() == target.size()
    dice = 0
    for channel in range(input.shape[1]):
        dice += dice_coeff(input[:, channel, ...], target[:, channel, ...], reduce_batch_first, epsilon)

    return dice / input.shape[1]


def dice_loss(input: Tensor, target: Tensor, multiclass: bool = False):
    # Dice loss (objective to minimize) between 0 and 1
    assert input.size() == target.size()
    fn = multiclass_dice_coeff if multiclass else dice_coeff
    return 1 - fn(input, target, reduce_batch_first=True)
'''





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





def conv_trans_block(in_channels,out_channels):
    conv_trans_block = nn.Sequential(
    nn.Conv2d(in_channels,out_channels,kernel_size=3,padding=1),
    nn.ReLU(inplace=True),
    nn.BatchNorm2d(out_channels)
    )
    return conv_trans_block





from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F


def np2th(weights, conv=False):
    """Possibly convert HWIO to OIHW."""
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)


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
    """Pre-activation (v2) bottleneck block.
    """

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
    """Implementation of Pre-activation (v2) ResNet mode."""

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





class Embeddings(nn.Module):
    """Construct the embeddings from patch, position embeddings.
    """
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





def dice_metric(inputs, target):
    num = target.size(0)
    inputs = inputs.reshape(num, -1)
    target = target.reshape(num, -1)
    smooth = 1.0
    intersection = (inputs * target)
    dice = (2. * intersection.sum(1) + smooth) / (inputs.sum(1) + target.sum(1) + smooth)
    dice = dice.sum() / num
    return dice

def dice_loss(inputs, target):
    num = target.size(0)
    inputs = inputs.reshape(num, -1)
    target = target.reshape(num, -1)
    smooth = 1.0
    intersection = (inputs * target)
    dice = (2. * intersection.sum(1) + smooth) / (inputs.sum(1) + target.sum(1) + smooth)
    dice = 1 - dice.sum() / num
    return dice

def bce_dice_loss(inputs, target):
    dicescore = dice_loss(inputs, target)
    bcescore = nn.BCELoss()
    bceloss = bcescore(inputs, target)

    return bceloss + dicescore





'''
def evaluate(net, dataloader, device,phase):
    net.eval()
    num_val_batches = len(dataloader)
    dice_score = 0
    # iterate over the validation set
    print(phase+" dataset evaluation")
    for batch in tqdm.tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
        #print(batch[0].shape,batch[1].shape)
        image, mask_true = batch[0], batch[1]
        # move images and labels to correct device and type
        image = image.to(device=device, dtype=torch.float32)
        mask_true = mask_true.to(device=device, dtype=torch.long)
        mask_true = F.one_hot(mask_true.squeeze(1), net.n_classes).permute(0, 3, 1, 2).float()

        with torch.no_grad():
            # predict the mask
            mask_pred = net(image)

            # convert to one-hot format
            if net.n_classes == 1:
                mask_pred = (F.sigmoid(mask_pred) > 0.5).float()
                # compute the Dice score
                dice_score += dice_coeff(mask_pred, mask_true.squeeze(1), reduce_batch_first=False)
            else:
                mask_pred = F.one_hot(mask_pred.squeeze(1).argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float()
                # compute the Dice score, ignoring background
                dice_score += multiclass_dice_coeff(mask_pred[:, 1:, ...], mask_true[:, 1:, ...], reduce_batch_first=False)
    
           

    net.train()

    # Fixes a potential division by zero error
    if num_val_batches == 0:
        return dice_score
    #print("dice_score:", dice_score/ num_val_batches)
    return dice_score / num_val_batches
'''





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
            for x,y in tqdm.tqdm(dataloader):
                #print(x.shape)
                #print(y.shape)
                x = x.to(device,dtype= torch.float32)
                y = y.to(device, dtype = torch.float32)
                step += 1
                #print(x.size(),y.size())
                # forward pass
                if phase == 'train':
                    # zero the gradients
                    optimizer.zero_grad()
                    outputs = model(x)
                    dice_score = dice_metric(outputs.squeeze(1), y.squeeze(1))
                    loss = bce_dice_loss(outputs.squeeze(1), y.squeeze(1))
                    #loss = loss_fn(outputs.squeeze(1), y.squeeze(1)) +  dice_loss(F.softmax(outputs, dim=1).float(),
                                       #F.one_hot(y.squeeze(1), model.n_classes).permute(0, 3, 1, 2).float(),
                                       #multiclass=True)

                    # the backward pass frees the graph memory, so there is no 
                    # need for torch.no_grad in this training pass
                    loss.backward()
                    optimizer.step()
                    # scheduler.step()

                else:
                    with torch.no_grad():
                        outputs = model(x)
                        loss = bce_dice_loss(outputs.squeeze(1), y.squeeze(1))
                        dice_score = dice_metric(outputs.squeeze(1), y.squeeze(1))
                        #loss = loss_fn(outputs.squeeze(1), y.squeeze(1)) + dice_loss(F.softmax(outputs, dim=1).float(),
                                       #F.one_hot(y.squeeze(1), model.n_classes).permute(0, 3, 1, 2).float(),
                                       #multiclass=False)




                # stats - whatever is the phase
                #acc = acc_fn(outputs, y)

                #running_dice_score  += evaluate(model, dataloader , device)
                #print("running_dice_score:",running_dice_score)
                running_loss += loss*x.shape[0] 
                running_dice_score += dice_score*x.shape[0]
                

                #collect residual garbage
                gc.collect()

                #if step % 10 == 0:
                    # clear_output(wait=True)
                    #print('Current step: {}  Loss: {}  Acc: {}  AllocMem (Mb): {}'.format(step, loss, acc, torch.cuda.memory_allocated()/1024/1024))
                    # print(torch.cuda.memory_summary())
            
            epoch_loss = running_loss / number_of_images
            #epoch_dice_score = evaluate(model, dataloader , device,phase)
            epoch_dice_score = running_dice_score/number_of_images
            
            if phase=='train':
                #wandb.log({'train_loss': epoch_loss, 'epoch': epoch})
                #wandb.log({'train_dice_score': epoch_dice_score, 'epoch': epoch})
                train_loss.append(epoch_loss.detach().item())
                train_dice.append(epoch_dice_score.detach().item())
            else:
                #wandb.log({'val_loss': epoch_loss, 'epoch': epoch})
                #wandb.log({'val_dice_score': epoch_dice_score, 'epoch': epoch})
                valid_loss.append(epoch_loss.detach().item())
                valid_dice.append(epoch_dice_score.detach().item())
            
            if phase == 'valid':
                scheduler.step(epoch_dice_score)

            print('{} Loss: {:.4f} dice_score: {}'.format(phase, epoch_loss, epoch_dice_score))
            
            if epoch == epochs-1 and phase == 'valid':
                destination = '/home/harneet.khanuja/colon_cancer/model_weights/pretrain/'
                torch.save({'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    }, destination+'transunet_pretrain_v01.h5')
            
            
    time_elapsed = time.time() - start
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))    
    
    return train_loss, valid_loss,train_dice,valid_dice,model    





transunet_model = transUnet(p=0.2,attn_p=0.2)
transunet_model_dict = transunet_model.state_dict()
pretrained_dict = {k: v for k, v in m1.state_dict().items() if k in transunet_model_dict}
transunet_model_dict.update(pretrained_dict)
transunet_model.load_state_dict(transunet_model_dict)





# n_channels=3 for RGB images
# n_classes is the number of probabilities you want to get per pixel

#UNetmodel1 = UNet(n_channels=3, n_classes=1)
#checkpoint = torch.load('/home/harneet.khanuja/colon_cancer/model_weights/transunet_test_v0302.h5')

#transunet_model = transUnet()
transunet_model.to(device)
opt = optim.SGD(transunet_model.parameters(), lr = 0.01, momentum = 0.9,weight_decay=0.000001)  
scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, 'max', patience=2)  # goal: maximize Dice score
#opt.load_state_dict(checkpoint['optimizer_state_dict'])
#transunet_model.load_state_dict(checkpoint['state_dict'])
#scheduler.load_state_dict(checkpoint['scheduler'])





train_loss, valid_loss,train_dice,valid_dice,out_model = train1(transunet_model,train_dataloader,valid_dataloader,len(train_classification_dataset),len(test_classification_dataset),opt,scheduler,batch_size,epochs = 1)





with open('/home/harneet.khanuja/colon_cancer/model_metrics/pretrain/transunet_pretrain_v01_train_loss.txt', 'w') as f:
    f.write(' '.join(map(str,train_loss)))
with open('/home/harneet.khanuja/colon_cancer/model_metrics/pretrain/transunet_pretrain_v01_valid_loss.txt', 'w') as f:
    f.write(' '.join(map(str,valid_loss)))
with open('/home/harneet.khanuja/colon_cancer/model_metrics/pretrain/transunet_pretrain_v01_train_dice.txt', 'w') as f:
    f.write(' '.join(map(str,train_dice)))
with open('/home/harneet.khanuja/colon_cancer/model_metrics/pretrain/transunet_pretrain_v01_valid_dice.txt', 'w') as f:
    f.write(' '.join(map(str,valid_dice)))




#checkpoint = torch.load(path)




device='cpu'
img_id = 1 
transunet_model.eval()
x,y = next(iter(valid_dataloader))
x = x.to(device,dtype= torch.float32)
y = y.to(device, dtype = torch.float32)
transunet_model = transunet_model.to(device)
outputs = transunet_model(x)
f, axes = plt.subplots(6, 3, figsize=(12, 12))
for k in range(x.shape[0]):
    gt = y[k].detach().numpy().transpose((1, 2, 0))
    pred = outputs[k].detach().numpy().transpose((1, 2, 0))
    img = x[k].detach().numpy().transpose((1, 2, 0))
    pred[pred>=0.5] = 1
    pred[pred<0.5] = 0
    axes[k][0].imshow(img)
    axes[0][0].set_title('Original')
    axes[k][0].xaxis.set_visible(False)
    axes[k][0].yaxis.set_visible(False)
    
    axes[k][1].imshow(gt,cmap='gray')
    axes[0][1].set_title('Ground Truth')
    axes[k][1].xaxis.set_visible(False)
    axes[k][1].yaxis.set_visible(False)
    
    axes[k][2].imshow(pred,cmap='gray')
    axes[0][2].set_title('Predicted')
    axes[k][2].xaxis.set_visible(False)
    axes[k][2].yaxis.set_visible(False)
#     print(img_id)
    img_id+=1





