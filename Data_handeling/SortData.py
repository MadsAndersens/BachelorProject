import os
from PIL import Image
from utils import base_dir, get_image_name
from scipy.io import loadmat
import pandas as pd

def sort_data():
    """ Sort the data into the correct folders. """
    # Get the names of the images
    folders = os.listdir(f'{base_dir}/BachelorProject/Data/VitusData/Serier')
    data_set = pd.DataFrame(columns=['ImageDir', 'Label','MaskDir'])

    for series in folders:
        images = os.listdir(f'{base_dir}/BachelorProject/Data/VitusData/Serier/{series}/{series}/CellsCorr')
        gt_dirs = os.listdir(f'{base_dir}/BachelorProject/Data/VitusData/Serier/{series}/{series}/MaskGT')
        for image in images:
            # Get the name of the image
            image_name = get_image_name(image)
            gt_name = f'{gt_dirs[0][:17]}{image_name}.mat'
            if gt_name in gt_dirs:
                # load the matlab file
                f = loadmat(f'{base_dir}/BachelorProject/Data/VitusData/Serier/{series}/{series}/MaskGT/{gt_name}')

                # Get the mask and the label
                mask = f['GTMask']*255
                label = f['GTLabel'].ravel()
            else:
                mask = None
                label = 'Negative'

            if mask is not None:
                # Loop through the labels and save the mask
                for idx, lab in enumerate(label):

                    # Get the name of the label
                    lab_name = lab[0]
                    mask_dir = f'{base_dir}/BachelorProject/Data/VitusData/Masks/{image_name}_{lab_name}_{idx}.png'

                    # Save the mask
                    Image.fromarray(mask[:,:,idx] if len(mask.shape) > 2 else mask).save(mask_dir)

                    # Add the data to the data set
                    temp = pd.DataFrame({'ImageDir': [
                        f'{base_dir}/BachelorProject/Data/VitusData/Serier/{series}/{series}/CellsCorr/{image}'],
                                         'Label': [lab_name],
                                         'MaskDir': [mask_dir]})
                    data_set = pd.concat([data_set, temp])

            else:
                # Add the data to the data set
                temp = pd.DataFrame({'ImageDir': [f'{base_dir}/BachelorProject/Data/VitusData/Serier/{series}/{series}/CellsCorr/{image}'],
                                    'Label': ['Negative'],
                                    'MaskDir': [None]})
                data_set = pd.concat([data_set,temp])

    # Save the data set as csv file
    data_set.to_csv(f'{base_dir}/BachelorProject/Data/VitusData/DataSet.csv', index=False)


if __name__ == '__main__':
    sort_data()