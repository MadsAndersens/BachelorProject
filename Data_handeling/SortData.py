import os
from PIL import Image
from utils import base_dir, get_image_name
from scipy.io import loadmat
import pandas as pd
from tqdm import tqdm
import numpy as np

def sort_data(mask_threshold=150):
    """ Sort the data into the correct folders. """
    # Get the names of the images
    folders = os.listdir(f'{base_dir}/BachelorProject/Data/VitusData/Serier')
    folders = sorted(folders)
    data_set = pd.DataFrame(columns=['ImageDir', 'Label','MaskDir'])

    # Loop through the folders
    for series in tqdm(folders):
        # Skip the .DS_Store file
        if series == '.DS_Store':
            continue

        # Get the names of the images and the ground truth
        images = os.listdir(f'{base_dir}/BachelorProject/Data/VitusData/Serier/{series}/CellsCorr')
        gt_dirs = os.listdir(f'{base_dir}/BachelorProject/Data/VitusData/Serier/{series}/MaskGT')

        # Loop through the images
        for image in images:
            # Get the name of the image
            image_name = get_image_name(image)
            gt_name = f'{gt_dirs[0][:17]}{image_name}.mat' # Get the name of the ground truth file
            if gt_name in gt_dirs:
                # load the matlab file
                f = loadmat(f'{base_dir}/BachelorProject/Data/VitusData/Serier/{series}/MaskGT/{gt_name}')

                # Get the mask and the label
                mask = f['GTMask']*255
                label = f['GTLabel'].ravel()
            else:
                mask = None
                label = 'Negative'

            if mask is not None:
                # Loop through the labels and save the mask
                for idx, lab in enumerate(label):
                    if np.sum(mask[:,:,idx] if len(mask.shape) > 2 else mask) >= mask_threshold:
                        # Get the name of the label
                        lab_name = ''.join(lab[0].split(' '))
                        mask_dir = f'{base_dir}/BachelorProject/Data/VitusData/Masks/{image_name}_{lab_name}_{idx}.png'

                        # Save the mask
                        Image.fromarray(mask[:,:,idx] if len(mask.shape) > 2 else mask).save(mask_dir)

                        # Add the data to the data set
                        temp = pd.DataFrame({'ImageDir': [
                            f'{base_dir}/BachelorProject/Data/VitusData/Serier/{series}/CellsCorr/{image}'],
                                             'Label': [lab_name],
                                             'MaskDir': [mask_dir]})
                        data_set = pd.concat([data_set, temp])

            else:
                # Add the data to the data set
                temp = pd.DataFrame({'ImageDir': [f'{base_dir}/BachelorProject/Data/VitusData/Serier/{series}/CellsCorr/{image}'],
                                    'Label': ['Negative'],
                                    'MaskDir': [None]})
                data_set = pd.concat([data_set,temp])

    # Save the data set as csv file
    data_set.to_csv(f'{base_dir}/BachelorProject/Data/VitusData/DataSet.csv', index=False)


if __name__ == '__main__':
    sort_data(mask_threshold=150)