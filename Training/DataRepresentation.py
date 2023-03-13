""" This document is responsible for creating the data-set and then the dataloaders for the training and validation of the model.
This will also be the file for splitting the data, into train and test sets.
"""

import os
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import random
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset, DataLoader
from PIL import Image

# Load the DataSet.csv file
data_set = pd.read_csv('/Users/madsandersen/PycharmProjects/BscProjektData/BachelorProject/Data/VitusData/DataSet.csv')

# Split the data into train and test sets with sklearn
train_set, test_set = train_test_split(data_set, test_size=0.4, random_state=42)

# Create a class for the data set
class SolarELData(Dataset):
    def __init__(self, DataFrame, transform=None):
        self.data_set = DataFrame
        self.transform = transform

    def __len__(self):
        return len(self.data_set)

    def __getitem__(self, idx):
        img_name = self.data_set.iloc[idx, 0]
        image = Image.open(img_name)
        image = ToTensor()(image)

        # Resize the image to 224x224
        image = transforms.Resize((224, 224))(image)
        # Make three channels
        image = torch.cat((image, image, image), 0)

        label = self.data_set.iloc[idx, 1]
        label = self.one_hot_encode(label)

        if self.transform is not None:
            image = self.transform(image)
        return image, label

    def one_hot_encode(self, label):
        # One hot encoding
        if label == 'Crack A':
            label = torch.tensor([1, 0, 0, 0, 0], dtype=torch.float32)
        elif label == 'Crack B':
            label = torch.tensor([0, 1, 0, 0, 0], dtype=torch.float32)
        elif label == 'Crack C':
            label = torch.tensor([0, 0, 1, 0, 0], dtype=torch.float32)
        elif label == 'Finger Failure':
            label = torch.tensor([0, 0, 0, 1, 0], dtype=torch.float32)
        else:
            label = torch.tensor([0, 0, 0, 0, 1], dtype=torch.float32)
        return label

#Transforms
transform_train = transforms.Compose(
    [
     transforms.RandomHorizontalFlip(),
     transforms.RandomVerticalFlip(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

test_transform = transforms.Compose(
    [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


# Create the data sets
train_data = SolarELData(train_set, transform=transform_train)
test_data = SolarELData(test_set, transform=test_transform)

# Create the data loaders
#train_loader = DataLoader(train_data, batch_size=4, shuffle=True, num_workers=2)
#test_loader = DataLoader(test_data, batch_size=4, shuffle=True, num_workers=2)

if __name__ == '__main__':
    # Test the data set
    print('Length of train set:', len(train_data))
    print('Length of test set:', len(test_data))
