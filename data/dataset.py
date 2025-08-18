import torch
import numpy as np
import torch.nn as nn
import pandas as pd
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader, Dataset
from skimage.io import imread

artists_csv = pd.read_csv('data/files/artists.csv')

class ArtGenreDataset(Dataset):
    def __init__(self, images_dir, csv_path, transform=None):
        
        self.images_dir = images_dir
        self.transform = transform
        self.data = pd.read_csv(csv_path)
        self.genre_to_index = self._build_genre_index()
        self.n_classes = len(self.genre_to_index)

    def _build_genre_index(self):
        all_genres = set()
        for genres in self.data[1]: 
            genre_list = genres.split(', ')
            all_genres.update(genre_list)
        return {genre: idx for idx, genre in enumerate(sorted(all_genres))}

    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        image_name = os.path.join(self.images_dir, self.data.iloc[idx, 0])
        img = imread(image_name)
        
        if self.transform:
            image = self.transform(image)
        genre_str = self.data.iloc[idx, 1]
        genre_list = genre_str.split(', ')
        label_vector = torch.zeros(self.n_classes, dtype=torch.float32)
        for genre in genre_list:
            genre = genre.strip()
            if genre in self.genre_to_index:
                label_vector[self.genre_to_index[genre]] = 1.0
        return image, label_vector







