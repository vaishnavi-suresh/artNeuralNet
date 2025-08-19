import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
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

T = v2.Compose([
    v2.CenterCrop([224,224]),
    v2.ColorJitter(brightness= [0.4,0.6], contrast = [0.4,0.6], saturation = [0.4,0.,6])
])

art_dataset = ArtGenreDataset(
    images_dir='../../data/files/resized/resized',
    csv_path='../../data/files/artist_images.csv',
    transform=T
)

train_size = int(0.8 * len(art_dataset))
test_size = len(art_dataset) - train_size

generator = torch.Generator().manual_seed(42)
train_dataset, test_dataset = random_split(art_dataset, [train_size, test_size], generator=generator)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=4)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride = 1, downsample = None) -> None:
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride = 1),
            nn.BatchNorm2d(out_channels)    
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride = 3, padding = 1),
            nn.BatchNorm2d(out_channels)    
        )
        self.relu = nn.ReLU()
        self.downsample = downsample
        self.out_channels = out_channels
    
    def forward(self,x):
        identity = x
        x = self.conv1(x) 
        x = self.conv2(x)
        if self.downsample:
            identity =self.downsample(x)
        
        x+=identity

        return self.relu(x)

class ResNet(nn.Module):
    def __init__(self,block, layers, num_classes, in_channels=None):
        super.init()
        self.in_channels = in_channels if in_channels else 64
        self.conv2to5 = nn.Sequential(
            nn.Conv2d(self.in_channels, out_channels = self.in_channels, kernel_size=3, stride = 3, padding = 1),
            nn.BatchNorm2d(self.in_channels)    
        )
        self.avgpool = nn.AvgPool2d(3, stride=1)
        self.layer0 = self._make_layer(block, 64, layers[0])
        self.layer1 = self._make_layer(block, 128, layers[0])
        self.layer2 = self._make_layer(block, 256, layers[0])
        self.layer3 = self._make_layer(block, 512, layers[0])
        self.dropout = nn.Dropout(.1)
        self.fc = nn.Linear(512,num_classes)
        
    def _make_layer(self, block, out_channels, blocks, stride = 1):
        downsample = None
        if stride >1 or self.in_channels != self.out_channels:
            downsample = nn.Sequential([
                nn.Conv2d(in_channels = self.in_channels, out_channels=out_channels, kernel_size = 1, stride = stride, bias = false)
                nn.BatchNorm2d(out_channels)
            ])
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for i in range(blocks):
            layers.append(block(self.in_channels, out_channels))
        return nn.Sequential(*layers)
    def forward(self, x):
        x = self.conv2to5(x)
        for i in range(4):
            x = self.conv2to5(x)
        x = self.avgpool(x)
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.dropout(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = torch.flatten(x,1)
        x = self.fc(x)
        return x
        




        







