import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"
import torch
import numpy as np
import torch.nn as nn
import pandas as pd
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader, Dataset, random_split
from skimage.io import imread
from data.dataset import ArtGenreDataset
from torchvision.transforms import v2



class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride = 1, downsample = None) -> None:
        super().__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride = 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()    
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride = stride, padding = 1),
            nn.BatchNorm2d(out_channels),   
            nn.ReLU() 
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels*4, kernel_size=1, stride = stride),
            nn.BatchNorm2d(out_channels*4)    
        )
        self.relu = nn.ReLU()
        self.downsample = downsample
        self.out_channels = out_channels
    
    def forward(self,x):
        identity = x
        out = self.conv1(x) 
        out = self.conv2(out)
        out = self.conv3(out)
        if self.downsample:
            identity =self.downsample(identity)
        
        out+=identity

        return self.relu(out)

class ResNet(nn.Module):
    def __init__(self,block, layers, num_classes, in_channels=None, out = None):
        super().__init__()
        
        self.in_channels = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels = 3, out_channels = self.in_channels , kernel_size=3, stride = 1, padding = 1),
            nn.BatchNorm2d(self.in_channels),  
            nn.ReLU(inplace = True)  
        )
        self.conv2to5 = nn.Sequential(
            nn.Conv2d(in_channels = self.in_channels, out_channels = self.in_channels , kernel_size=3, stride = 1, padding = 1),
            nn.BatchNorm2d(self.in_channels),  
            nn.ReLU(inplace = True)  
        )
        self.avgpool = nn.AvgPool2d(3, stride=1)
        self.layer0 = self._make_layer(block, 64, layers[0])
        self.layer1 = self._make_layer(block, 128, layers[1])
        self.layer2 = self._make_layer(block, 256, layers[2])
        self.layer3 = self._make_layer(block, 512, layers[3])
        self.dropout = nn.Dropout(.1)
        self.fc = nn.Linear(2048,num_classes)
        
    def _make_layer(self, block, out_channels, blocks, stride = 1):
        downsample = None
        if stride > 1 or self.in_channels != out_channels*4:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels=self.in_channels, out_channels=out_channels*4, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels*4)
            )
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels*4
        for i in range(1,blocks):
            layers.append(block(self.in_channels, out_channels, downsample = None))
        return nn.Sequential(*layers)
    def forward(self, x):
        x = self.conv1(x)
        for i in range(4):
            x = self.conv2to5(x)
        x = self.avgpool(x)
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.dropout(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = nn.functional.adaptive_avg_pool2d(x, (1,1))
        x = torch.flatten(x,1)
        x = self.fc(x)
        return x
        




        







