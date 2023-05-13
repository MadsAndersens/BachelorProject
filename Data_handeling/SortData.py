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

    filtered_data = pd.read_csv('/Users/madsandersen/PycharmProjects/BscProjektData/BachelorProject/Data/VitusData/EL_data_filtered.csv')

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
            #Check if the image is corrected for lighting if not pass on to next iter
            if not filtered_data['name'].str.contains(image[:-4]).any():
                #print('True')
                continue

            # Get the name of the image
            image_name = get_image_name(image)
            gt_name = f'{gt_dirs[0][:17]}{image_name}.mat' # Get the name of the ground truth file
            if gt_name in gt_dirs:
                # load the matlab file
                f = loadmat(f'{base_dir}/BachelorProject/Data/VitusData/Serier/{series}/MaskGT/{gt_name}')

                # Get the mask and the label
                mask = f['GTMask']
                label = f['GTLabel'].ravel()
            else:
                mask = None
                label = 'Negative'

            if mask is not None:
                # Loop through the labels and save the mask
                labs_t,masks_t = [],[]
                for idx, lab in enumerate(label):
                    if np.sum(mask[:,:,idx] if len(mask.shape) > 2 else mask) >= mask_threshold:
                        mask = mask
                        # Get the name of the label
                        lab_name = ''.join(lab[0].split(' '))
                        mask_dir = f'{base_dir}/BachelorProject/Data/VitusData/VitusStatsMask/{image_name}_{lab_name}_{idx}.png'

                        # Save the mask
                        Image.fromarray(mask[:,:,idx]*255 if len(mask.shape) > 2 else mask*255).save(mask_dir)

                        #Make a list of mask dirs and lab names to the temporary lists
                        labs_t.append(lab_name)
                        masks_t.append(mask_dir)


                # Add the data to the data set
                labs_t = '__'.join(labs_t)
                masks_t = '__'.join(masks_t)
                # This takes care of when the mask is not "big enough" then the result will be ''
                if labs_t != '':
                    temp = pd.DataFrame({'ImageDir': [
                        f'{base_dir}/BachelorProject/Data/VitusData/Serier/{series}/CellsCorr/{image}'],
                                         'Label': labs_t,
                                         'MaskDir': masks_t})
                    data_set = pd.concat([data_set, temp])

                else:
                    temp = pd.DataFrame({'ImageDir': [
                        f'{base_dir}/BachelorProject/Data/VitusData/Serier/{series}/CellsCorr/{image}'],
                                         'Label': 'Negative',
                                         'MaskDir': None})
                    data_set = pd.concat([data_set, temp])



            else:
                # Add the data to the data set
                temp = pd.DataFrame({'ImageDir': [f'{base_dir}/BachelorProject/Data/VitusData/Serier/{series}/CellsCorr/{image}'],
                                    'Label': ['Negative'],
                                    'MaskDir': [None]})
                data_set = pd.concat([data_set,temp])

    # Save the data set as csv file and split the cols into lists before saving.
    data_set['Label'] = data_set.apply(lambda x: str(x['Label']).split('__') if x['Label'] is not None else None, axis = 1)
    data_set['is_fault'] = data_set['Label'].apply(lambda x: 0 if 'Negative' in x else 1)
    data_set['MaskDir'] = data_set.apply(lambda x: str(x['MaskDir']).split('__') if x['MaskDir'] is not None else None, axis = 1)
    print(len(data_set))
    print(len(data_set['is_fault'][data_set['is_fault'] == 1] ))
    data_set.to_csv(f'{base_dir}/BachelorProject/Data/VitusData/DataSetVitusStats.csv', index=False)


if __name__ == '__main__':
    sort_data(mask_threshold = 410)
    from focal_loss.focal_loss import FocalLoss
