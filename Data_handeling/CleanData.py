import numpy as np
from PIL import Image
import os
import pandas as pd
from scipy.io import loadmat

# Define the path to the data
path = f'/BachelorProject/Data/Serie1_raw_14Feb/MaskGT'
dir = os.listdir(path)

# Function for loading the mat files
def load_GT(dir):
    gt = {}
    for file in dir:
        try:
            g = loadmat(f'{path}/{file}')
            gt[file] = [g['GTLabel'].ravel(),g['GTMask']]
        except:
            pass
    return gt

# Function for checkin which image has a mask positive above a certain threshold
def check_mask(gt, threshold):
    mask = []
    for key, value in gt.items():
        if np.sum(value[1]) > threshold:
            mask.append(key)
    return mask

def write_csv(gt):
    df = pd.DataFrame.from_dict(gt,orient='index')
    df.columns = ['GTLabel','GTMask']
    df.to_csv('/Users/madsandersen/PycharmProjects/BscProjektData/BachelorProject/Data/Mask.csv', index=True, header=True)


if __name__ == '__main__':
    #gt = load_GT(dir)
    #mask = check_mask(gt, 100)
    #write_csv(gt)
