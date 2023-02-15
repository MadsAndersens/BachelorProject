import torch
import os
import cv2
import numpy as np
from scipy.io import loadmat


# Write a coustum dataset for the data from Claire
class DTU_Dataset(torch.utils.data.Dataset):
    def __init__(self, img_root,MaskGT, transform=None):
        self.img_root = img_root
        self.MaskGT = MaskGT
        self.transform = transform
        self.imgs = list(sorted(os.listdir(os.path.join(img_root))))
        self.labels = list(sorted(os.listdir(os.path.join(MaskGT))))

    def __getitem__(self, idx):
        # load images ad masks
        img_path = os.path.join(self.img_root, self.imgs[idx])
        labels_path = os.path.join(self.MaskGT, 'GT_Serie_1_Image_-' + self.imgs[idx][17:-4]+'.mat')
        #print(img_path, labels_path)


        #Read the image with CV2
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        #Load the labels from the .mat file
        try:
            gt = loadmat(labels_path)
            print(len(gt['GTLabel']))
             # Set GTlabel
            category = gt['GTLabel'][0]
             # Set the mask
            mask = gt['GTMask']
        except FileNotFoundError:
            category = 'No Defect'
            mask = np.zeros((img.shape[0],img.shape[1]))

        # apply transforms
        if self.transform is not None:
            img = self.transform(img)

        return img, category,mask

    def __len__(self):
        return len(self.imgs)

# Write a coustum class which allows us to only get the images which have defects ie. a mask
class DTU_Dataset_Defects(torch.utils.data.Dataset):
    def __init__(self, img_root,MaskGT, transform=None):
        self.img_root = img_root
        self.MaskGT = MaskGT
        self.transform = transform
        self.imgs = list(sorted(os.listdir(os.path.join(img_root))))
        self.labels = list(sorted(os.listdir(os.path.join(MaskGT))))

    def __getitem__(self, idx):
        # load images which are found in the MaskGT folder
        lbl = self.labels[idx]
        img_name = f'Serie_1_ImagesGS_{lbl[18:-4]}.png'
        img_path = os.path.join(self.img_root, img_name)
        labels_path = os.path.join(self.MaskGT, lbl)

        #Read the image with CV2
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        #Load the labels from the .mat file

        gt = loadmat(labels_path)
        # Set GTlabel
        category = gt['GTLabel'][0]
        # Set the mask
        mask = gt['GTMask']

    def __len__(self):
        return len(self.labels)

