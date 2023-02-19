from PIL import Image
import numpy as np
import pandas as pd
from scipy.io import loadmat
import os
import random
from utils import get_image_name, trim_image, base_dir

def create_data_set():
    """"
    This is a function for creating a data set from the .mat files in the MaskGT folder. It returns the two dictionaries fault_set and cropped_set.
    The fault_set dictionary contains the image name as key and a list of tuples as value. The tuples contain the fault name and the mask for that fault.
    The cropped_set dictionary contains the fault name as key and a list of cropped images as value.
    """
    # Get the labels from the .mat files
    MaskGT = f'/Users/madsandersen/PycharmProjects/BscProjektData/BachelorProject/Data/Serie1_raw_14Feb/MaskGT'


    labels = list(sorted(os.listdir(os.path.join(MaskGT))))

    fault_set = {'Crack A': {},
                 'Crack B': {},
                 'Crack C': {},
                 'Finger Failure': {}
                 }
    cropped_set = {
                    'Crack A':[],
                    'Crack B':[],
                    'Crack C':[],
                    'Finger Failure':[]
                    }
    #base_dir = '/'
    lab = []
    # loop through each of the .mat files
    for fault in labels:
        try:
            f = loadmat(f'{base_dir}/BachelorProject/Data/Serie1_raw_14Feb/MaskGT/{fault}')
        except:
            print(f'file{fault} failed loading')
            continue

        label,mask = f['GTLabel'].ravel(), f['GTMask']
        lab.append(label)
        img_name = get_image_name(fault)
        img = Image.open(f'{base_dir}/BachelorProject/Data/Serie1_raw_14Feb/CellsGS/Serie_1_ImageGS_{img_name}.png').convert('L')
        # Split into correct places
        for idx,i in enumerate(label):
            #print(idx)
            if i[0] == 'Crack A':
                # Add the crop to the cropped data_base:
                crop = trim_image(img,mask[:,:,idx] if len(mask.shape)>2 else mask)
                cropped_set[i[0]].append(crop)
                if img_name not in fault_set['Crack A'].keys():
                    fault_set['Crack A'][img_name] = [(i[0],mask[:,:,idx] if len(mask.shape) > 2 else mask)]
                else:
                    fault_set['Crack A'][img_name].append((i[0],mask[:,:,idx] if len(mask.shape) > 2 else mask))
            elif i[0] == 'Crack B':
                # Add the crop to the cropped data_base:
                crop = trim_image(img,mask[:,:,idx] if len(mask.shape)>2 else mask)
                cropped_set[i[0]].append(crop)
                if img_name not in fault_set['Crack B'].keys():
                    fault_set['Crack B'][img_name] = [(i[0],mask[:,:,idx] if len(mask.shape)>2 else mask)]
                else:
                    fault_set['Crack B'][img_name].append((i[0],mask[:,:,idx] if len(mask.shape)>2 else mask))
            elif i[0] == 'Crack C':
                # Add the crop to the cropped data_base:
                crop = trim_image(img,mask[:,:,idx] if len(mask.shape)>2 else mask)
                cropped_set[i[0]].append(crop)
                if img_name not in fault_set['Crack C'].keys():
                    fault_set['Crack C'][img_name] = [(i[0],mask[:,:,idx] if len(mask.shape)>2 else mask)]
                else:
                    fault_set['Crack C'][img_name].append((i[0],mask[:,:,idx] if len(mask.shape)>2 else mask))
            elif i[0] == 'Finger Failure':
                # Add the crop to the cropped data_base:
                crop = trim_image(img,mask[:,:,idx] if len(mask.shape)>2 else mask)
                cropped_set[i[0]].append(crop)
                if img_name not in fault_set['Finger Failure'].keys():
                    fault_set['Finger Failure'][img_name] = [(i[0],mask[:,:,idx] if len(mask.shape)>2 else mask)]
                else:
                    fault_set['Finger Failure'][img_name].append((i[0],mask[:,:,idx] if len(mask.shape)>2 else mask))

    return fault_set,cropped_set

if __name__ == '__main__':
    fault_set,cropped_set = create_data_set()

    #print(fault_set['CrackA']['-10_4081_Cell_Row4_Col_2'][0][0])
