import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import ast
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models
from torchmetrics import F1Score,ConfusionMatrix,ROC,PrecisionRecallCurve
from sklearn.metrics import ConfusionMatrixDisplay
import copy
import wandb
from torchvision.utils import make_grid

#Import huggin face transformer model for image classification
from transformers import ViTFeatureExtractor, ViTForImageClassification

# Hyperparameters
n_epochs = 120
batch_size_train = 128
batch_size_test = 128
learning_rate = 1e-5
momentum = 0.5
decay_gamma = 0.0
label_smoothing = 0.01
weight_decay = 0.2
architecture = 'resnet152'
optimizer = 'Adam'#'Adam'#'SGD'
loss = 'FocalLoss'#'CrossEntropy'#'FocalLoss'
sigma = 1.0 # For gaussian noise added.

#For focal loss
gamma = 1.5
alpha = 0.5

#Use synthetic data?
syn_type = 'Poisson'#'Gaussian' #Poisson
synthetic_data = True

loss_weights = torch.tensor([1, 4])

hp_dict = {
    'n_epochs': n_epochs,
    'batch_size_train': batch_size_train,
    'batch_size_test': batch_size_test,
    'learning_rate': learning_rate,
    'momentum': momentum,
    'decay_gamma': decay_gamma,
    'label_smoothing': label_smoothing,
    'weight_decay': weight_decay,
    'architecture': architecture,
    'optimizer': optimizer,
    'loss': loss,
    'gamma': gamma,
    'alpha': alpha,
    'syn_type': syn_type,
    'synthetic_data': synthetic_data,
    'loss_weights': loss_weights
}

converters = {
    'Label': lambda x: ast.literal_eval(x),
    'MaskDir': lambda x: ast.literal_eval(x) if str(x) != 'nan' else x
}
root_dir = '/Users/madsandersen/PycharmProjects/BscProjektData/'
# Load the DataSet.csv file
train_set = pd.read_csv(f'{root_dir}BachelorProject/Data/VitusData/Train.csv', converters=converters)
val_set = pd.read_csv(f'{root_dir}BachelorProject/Data/VitusData/Val.csv', converters=converters)
#synthetic_set = pd.read_csv(f'/work3/s204137/Poisson/Data/Synthetic/SyntheticData.csv')


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the dataset class
class SegmentationSet(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        if 'Unnamed: 0' in self.data.columns:
            self.data = self.data.drop(columns=['Unnamed: 0'])

        self.transform = transform
        self.base_dir = '/Users/madsandersen/PycharmProjects/BscProjektData/BachelorProject/Data/'

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_dir = self.data.iloc[idx, 0]
        label = self.data.iloc[idx, 1]
        mask_dir = self.data.iloc[idx, 2]
        image = Image.open(self.base_dir + image_dir)
        if mask_dir != 'nan':
            mask = [np.array(Image.open(self.base_dir + mask_dir[i])) for i in range(len(label))]
            #Convert list of np arrays to single np array
            mask = np.stack(mask, axis=0)
            mask = torch.tensor(mask)
        else:
            im_shape = np.array(image).shape
            mask = torch.zeros((len(label), im_shape[0], im_shape[1]))

        if self.transform:
            image = self.transform(image)


        return image, mask, label


# Define the model
data = SegmentationSet(train_set, transform=None)
data[21]

if __name__ == '__main__':
    print('test')

