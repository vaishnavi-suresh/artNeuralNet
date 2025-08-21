import pandas as pd
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
import torch
from src.artneuralnet.resnet import ResNet, ResidualBlock
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

def log_memory():
    process = psutil.Process()
    mem_mb = process.memory_info().rss / (1024 * 1024)
    logging.info(f"Memory usage: {mem_mb:.2f} MB")

def log_gpu_memory():
    allocated = torch.cuda.memory_allocated() / (1024 * 1024)
    reserved = torch.cuda.memory_reserved() / (1024 * 1024)
    print(f"[GPU] Allocated: {allocated:.2f} MB, Reserved: {reserved:.2f} MB")


T = v2.Compose([
    v2.ToImage(),                    # robust to input; or ToPILImage if you prefer PIL
    v2.ToDtype(torch.uint8),         # ensure jitter gets proper dtype
    v2.RandomResizedCrop(224, antialias=True),
    v2.ColorJitter(brightness=(0.4, 0.6), contrast=(0.4, 0.6), saturation=(0.4, 0.6)),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    
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

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

def train(num_classes = None, num_epochs= None, batch_size = None, learning_rate = None, layers = None):
    df =pd.read_csv('../../data/files/artist_images.csv')
    splitcol = df['genre'].str.split(', ')
    exp = splitcol.explode()
    exp = exp.str.strip()
    num_classes = num_classes if num_classes else exp.nunique()
    num_epochs = num_epochs if num_epochs else 10
    batch_size = batch_size if batch_size else 4
    learning_rate = learning_rate if learning_rate else 0.01
    layers = layers if layers else [3,4,18,4]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = ResNet(ResidualBlock, layers, num_classes).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(params = model.parameters(), lr = learning_rate, weight_decay=.001, momentum = .9)



    for epoch in range (num_epochs):
        for i, (img, label_vector) in enumerate(train_loader):

            images = img.to(device)
            label = label_vector.to(device)
            out = model(images)
            loss = criterion(out,label)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gc.collect()
            optimizer.step()
            log_memory()
            log_gpu_memory()


        print ('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))

def main():

    train()

if __name__ == "__main__":
    main()









