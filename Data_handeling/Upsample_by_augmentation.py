import pandas as pd
from BachelorProject.Data_handeling.DataAugmentations.Augmentations import GaussianCopyPaste,PoisonCopyPaste
import numpy as np
from PIL import Image
from utils import base_dir
from tqdm import tqdm
from csv import DictWriter


class Upsampler:
    """
    Class for upsampling images given an augmentation. Will save the images into the folder specified by "path"
    data_set: pandas dataframe with columns: ImgDir,label,mask
    """
    def __init__(self,augmentation,save_path,data_set):
        self.augmentation = augmentation
        self.path = save_path
        self.data = data_set
        self.no_faults_data = data_set[data_set['Label'] == 'Negative']

        # Define failure data-sets:
        self.crack_a = data_set[data_set['Label'] == 'CrackA']
        self.crack_b = data_set[data_set['Label'] == 'CrackB']
        self.crack_c = data_set[data_set['Label'] == 'CrackC']
        self.finger_failure = data_set[data_set['Label'] == 'FingerFailure']

        # Intialize the failure data-sets:
        self.csv_name = 'SyntheticData.csv'
        self.init_csv(self.csv_name,['ImageDir','Label','fault'])

    def run_upsample(self,n_upsamples):
        """ Run the augmentation on the images in the database.
            n_upsamples: dict specifying the amount of each class being upsampled.
        """
        for category, n in n_upsamples.items():
            self.upsample(category,n)

    def upsample(self,category,n):
         """ Upsample a given category by n times. Creates a random augmentation for every.
             category: string specifying the category to upsample
             n: int specifying the amount of upsamples.
         """
         #Get the images to augment
         images = self.get_images(n)
         # Create a random augmentation for each image
         for i in tqdm(range(n)):
            # Create a random augmentation
            image = self.load_image(images[i])
            aug_image,img_fail_path = self.augmentation.augment_image(image,category)
            # Save the image
            self.save_image(aug_image,
                            category,
                            i,
                            images[i],
                            img_fail_path)
    def load_image(self,image_path):
        """ Load the image from the path, and trim it."""
        return Image.open(f'/Users/madsandersen/PycharmProjects/BscProjektData/BachelorProject/Data/{image_path}')
    def get_images(self,n):
        """ Get the none faulty images to augment selects them randomly following a uniform distribution, without replacement"""
        images = self.no_faults_data['ImageDir'].values
        return np.random.choice(images,n,replace = True)

    def save_image(self,image,category,i,id,img_fail_path):
        """ Save the image to the folder, and write the csv file."""
        # Naming stuff
        no_fault_name = id.split('/')[-1][:-4] # This crops down the path to the image name
        fault_name = img_fail_path.split('/')[-1][:-4]
        base_dir = '/Users/madsandersen/PycharmProjects/BscProjektData'
        category = ''.join(category.split(' ')) # Remove spaces in the category name
        save_path = f'{base_dir}/BachelorProject/Data/Synthetic/{category}/{no_fault_name}>{fault_name}.png'

        #Save the image
        image.save(save_path)
        csv_row = {'ImageDir': f'Data/Synthetic/{category}/{no_fault_name}>{fault_name}.png',
                   'label': category,
                   'fault': fault_name}
        self.write_csv(csv_row)

    def write_csv(self,csv_row):
        """ Write the csv file, given a dict where keys are the column names and values are the values."""
        csv_path = f'/Users/madsandersen/PycharmProjects/BscProjektData/BachelorProject/Data/Synthetic/{self.csv_name}'
        with open(csv_path,'a') as f:
            dict_writer = DictWriter(f, fieldnames=['ImageDir','label','fault'])
            dict_writer.writerow(csv_row)

    def init_csv(self,name, columns):
        base_dir = '/Users/madsandersen/PycharmProjects/BscProjektData'
        with open(f'{base_dir}/BachelorProject/Data/Synthetic/{name}', 'w') as f:
            f.write(','.join(columns))
            f.write("\n")



if __name__ == '__main__':
    # Create upsample object
    data_set = pd.read_csv('/Users/madsandersen/PycharmProjects/BscProjektData/BachelorProject/Data/VitusData/Train_expanded.csv')

    augmentation = GaussianCopyPaste()#PoisonCopyPaste() #PoisonCopyPaste() #GaussianCopyPaste()
    upsampler = Upsampler(augmentation,save_path = f'{base_dir}/BachelorProject/Data/Synthetic',data_set = data_set)

    # Run upsample
    n_upsamples = {'Crack A': 35000,
                   'Crack B': 35000,
                   'Crack C': 35000,
                   'Finger Failure': 35000}
    upsampler.run_upsample(n_upsamples)
