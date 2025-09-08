import pandas as pd
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"
import torch
from torch.amp import autocast, GradScaler
from artneuralnet.resnet import ResNet, ResidualBlock
import numpy as np
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader, Dataset, random_split
from skimage.io import imread
from data.dataset import ArtGenreDataset
from torchvision.transforms import v2
import gc
import psutil
import logging
from training.hyperparameters import test_loader


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
scaler = GradScaler(enabled=(device.type == 'cuda'))

def test(num_classes = None, num_epochs= None, batch_size = None, learning_rate = None, layers = None):
    df =pd.read_csv('../../data/files/artist_images.csv')
    splitcol = df['genre'].str.split(', ')
    exp = splitcol.explode()
    exp = exp.str.strip()
    num_classes = num_classes if num_classes else exp.nunique()
    num_epochs = num_epochs if num_epochs else 10
    batch_size = batch_size if batch_size else 2
    learning_rate = learning_rate if learning_rate else 0.01
    layers = layers if layers else [3,4,18,4]

    print(device)
    model = ResNet(ResidualBlock, layers, num_classes).to(device)
    model.load_state_dict(torch.load('../training/resnet_art.pth')) #CHANGE PATH
    criterion = nn.BCEWithLogitsLoss()

    model.eval()
    val_loss = 0

    with torch.no_grad():
        for i, (img, label_vector) in enumerate(test_loader):
            images = img.to(device)
            label = label_vector.to(device)
            if device.type == 'cuda':
                with autocast(device_type='cuda', dtype=torch.float16):
                    out = model(images)
                    loss = criterion(out, label)
                    val_loss += loss.item()
                    _, predicted = torch.max(out.data, 1)
                    
            else:
                out = model(images)
                loss = criterion(out, label)
                val_loss += loss.item()
                _, predicted = torch.max(out.data, 1)
            correct += (predicted == label).sum().item()
            total += label.size(0)
            precision += (predicted & label).sum().item()
            recall += (predicted & label).sum().item()
            
            
            


            if i % 10 == 0:
                print('Test Step [{}/{}], Loss: {:.4f}'.format(i, len(test_loader), loss.item()))
        
        
    accuracy = 100 * correct / total
    f1 = 2 * (precision * recall) / (precision + recall)
    with open('../../tests/test_results.txt', 'w') as f:
        f.write(f'Test Accuracy: {accuracy:.2f}%\n')
        f.write(f'Test F1 Score: {f1:.4f}\n')
        f.write(f'Final Test Loss: {loss.item():.4f}\n')
    print('Test Accuracy: {:.2f}%'.format(accuracy))
    print('Test F1 Score: {:.4f}'.format(f1))
    print('Final Test Loss: {:.4f}'.format(loss.item()))

