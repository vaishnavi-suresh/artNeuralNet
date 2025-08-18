import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
import torch
import numpy as np
import torch.nn as nn
import pandas as pd
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader, Dataset
from skimage.io import imread
from data.dataset import ArtGenreDataset

art_dataset = ArtGenreDataset(
    images_dir='../../data/files/resized/resized',
    csv_path='../../data/files/artist_images.csv'
)

print(art_dataset[0])







