import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.transforms import functional as F, InterpolationMode, transforms as T
import numpy as np
from torch import nn, Tensor
from torchvision import ops
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import pandas as pd

#Since pytorch does not support labelsmooth for bce loss
class LabelSmoothBCELoss(nn.Module):
    def __init__(self, smoothing=0.1):
        super(LabelSmoothBCELoss, self).__init__()
        self.smoothing = smoothing
        self.bce_loss = nn.BCELoss(reduction='mean')
    def forward(self, output, target):
        target = target * (1 - self.smoothing) + 0.5 * self.smoothing
        loss = self.bce_loss(output, target)
        return loss

class GaussianNoise(nn.Module):
    def __init__(self, mean=0, std=0.1):
        self.mean = mean
        self.std = std

    def forward(self, image):
        noise = torch.randn_like(image) * self.std + self.mean
        noisy_image = image + noise
        noisy_image = torch.clamp(noisy_image, 0, 1)  # Clip values between 0 and 1
        return noisy_image

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

    def __call__(self, image):
        return self.forward(image)


# Randomly gamma correct image
class RandomGammaCorrection(nn.Module):
    """
    Apply Gamma Correction to the images
    """
    def __init__(self, gamma_range=(0.5, 2.0)):
        self.gamma_range = gamma_range

    def __call__(self, img):
        #sample log uniformly in range [log(gamma_range[0]), log(gamma_range[1])]
        gamma = np.random.uniform(self.gamma_range[0], self.gamma_range[1])
        return F.adjust_gamma(img, gamma)

    def forward(self, img):
        return self.__call__(img)

    def __repr__(self):
        return self.__class__.__name__ + '(gamma_range={0})'.format(self.gamma_range)
        
class ZeroPad(nn.Module):
    def __init__(self, size):
        self.size = size

    def __call__(self, image):
        max_widt, max_height = self.size[0],self.size[1]
        width, height = image.shape[2], image.shape[1]
        pad_widt, pad_height = max_widt - width, max_height - height
        image = transforms.Pad((0,0, pad_widt, pad_height), padding_mode='constant')(image)
        return image
        
    def forward(self, img):
        return self.__call__(img)
    
    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.size)
        
def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


#PVLEAD dataset
class PVLEAD(Dataset):

    def __init__(self, transform=None):
        self.data_set = pd.read_csv('/work3/s204137/PVLEAD/PVLEAD_dataset.csv')
        self.transform = transform
        self.root = '/work3/s204137/PVLEAD/JPEGImages'

    def __len__(self):
        return len(self.data_set)

    def __getitem__(self, idx):
        img_name = self.data_set['file_name'].iloc[idx]
        img_name = f'{self.root}/{img_name}'
        image = Image.open(img_name)
        image = transforms.ToTensor()(image)*255
        
        image = self.pad_image(image)

        # Make three channels
        #image = torch.cat((image, image, image), 0)

        label = self.data_set['defect'].iloc[idx]
        label = self.one_hot_encode(label)

        if self.transform is not None:
            image = self.transform(image)

        sample = {'image': image, 'label': label}
        return sample

    def one_hot_encode(self, label):
        # One hot encoding
        place_holder = torch.tensor([0, 0], dtype=torch.float32)

        if 'Negative' in label:
            place_holder[0] = 1
        else:
            place_holder[1] = 1

        return place_holder

    def pad_image(self, image):
        max_widt, max_height = 430, 430
        width, height = image.shape[2], image.shape[1]
        pad_widt, pad_height = max_widt - width, max_height - height
        image = transforms.Pad((0, 0, pad_widt, pad_height), padding_mode='constant')(image)
        return image


class SolarELDataSynTest(Dataset):

    def __init__(self,synthetic_dataframe,syn_type, transform=None):
        self.synthetic_data_set = synthetic_dataframe
        self.syn_type = syn_type

        # Create variable in both indicating if the image is synthetic or not
        self.synthetic_data_set['Synthetic'] = True

        # Set the roots
        if self.syn_type == 'Mixed':
            self.syn_root_gaussian = f'/work3/s204137/SyntheticTestSet/Gaussian'
            self.syn_root_poisson = f'/work3/s204137/SyntheticTestSet/Poisson'

        self.syn_root = f'/work3/s204137/SyntheticTestSet/{self.syn_type}'  # '/Users/madsandersen/PycharmProjects/BscProjektData/BachelorProject'

        # Append the synthetic data set to the original data set
        self.data_set = self.synthetic_data_set
        self.transform = transform

        # Synthetic data is stored in a different folder so create a variable in the dataframes containing the root
        self.data_set['root'] = self.syn_root
        self.classes = ['Positive', 'Negative']

    def __len__(self):
        return len(self.data_set)

    def __getitem__(self, idx):

        is_synthetic = self.data_set['Synthetic'].iloc[idx]

        img_name = self.data_set['ImageDir'].iloc[idx][23:]

        # Use the different roots
        if self.syn_type == 'Mixed':
           img_name_G = f'{self.syn_root_gaussian}/{img_name}'
           img_name_P = f'{self.syn_root_poisson}/{img_name}'

        else:
           img_name = f'{self.syn_root}{img_name}'

        # Open the image
        if self.syn_type == 'Mixed' and is_synthetic:
            try:
                image = Image.open(img_name_G)
            except:
                image = Image.open(img_name_P)
        else:
            image = Image.open(img_name)

        image = transforms.ToTensor()(image) * 255


        image = self.pad_image(image)

        # Make three channels
        if image.shape[0] == 1:
            image = torch.cat((image, image, image), 0)

        label = self.data_set['Label'].iloc[idx]
        label = self.one_hot_encode(label if not is_synthetic else [label])

        if self.transform is not None:
            image = self.transform(image)

        sample = {'image': image, 'label': label}
        return sample

    def one_hot_encode(self, label):
        # One hot encoding
        place_holder = torch.tensor([0, 0], dtype=torch.float32)

        if 'Negative' == label:
            place_holder[0] = 1
        else:
            place_holder[1] = 1

        return place_holder

    def pad_image(self, image):
        max_widt, max_height = 430, 430
        width, height = image.shape[2], image.shape[1]
        pad_widt, pad_height = max_widt - width, max_height - height
        image = transforms.Pad((0, 0, pad_widt, pad_height), padding_mode='constant')(image)
        return image
