import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import torch
from PIL import Image
import pandas as pd

def transform_image(image):
    custom_transformer = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: torch.cat([x, x, x], 0)),
    ])
    image_tr = custom_transformer(image)
    return image_tr

def convert_label(label):
    if label == 'Wake':
        label = 0
    elif label == 'N1':
        label = 1
    elif label == 'N2':
        label = 2
    elif label == 'N3':
        label = 3
    elif label == 'REM':
        label = 4
    else:
        label = None
    return label

class TrainDataset(Dataset):
    def __init__(self, X, y, root='../DATA/'):
        self.root = root
        self.X = X
        self.y = y
        self.data = {}

        X.reset_index(drop=True, inplace=True)
        lst_img = [os.path.join(self.root, X[0][i], X[1][i]) for i in range(X.shape[0])]
        lst_label = [convert_label(label) for label in y.tolist()]
        self.data['image'] = lst_img
        self.data['label'] = lst_label

    def __getitem__(self, index):
        path = self.data['image'][index]
        img = Image.open(path)
        img = transform_image(img)
        label = self.data['label'][index]
        return img, label

    def __len__(self):
        return len(self.data['image'])


class TestDataset(Dataset):
    def __init__(self, root='../DATA/'):
        self.root = root
        self.data = {}
        test_path = os.path.join(self.root, 'testset-for_user.csv')
        test_data = pd.read_csv(test_path,header=None)

        lst_img = [os.path.join(self.root, test_data[0][i], test_data[1][i]) for i in range(test_data.shape[0])]
        self.data['image'] = lst_img

    def __getitem__(self, index):
        path = self.data['image'][index]
        img = Image.open(path)
        img = transform_image(img)
        return img

    def __len__(self):
        return len(self.data['image'])

def data_loader(phase, dataset, batch_size, num_workers=1):
    if phase == 'train':
        dataloader = DataLoader(dataset=dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    else: # valid, test
        dataloader = DataLoader(dataset=dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    return dataloader