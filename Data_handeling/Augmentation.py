from PIL import Image,ImageFilter
import os
import pandas as pd
import numpy as np
import random
from matplotlib import pyplot as plt
from data_formatting import create_data_set
from utils import get_image_name, trim_image, base_dir

class CopyPasteAugmentation(object):
    def __init__(self,fault_database,base_dir,blur,enable_plot = False):
        self.fault_database = fault_database
        self.enable_plot = enable_plot
        self.base_dir = base_dir
        self.blur = blur

    def get_augmentet_image(self,image,category):
        """ Return a random copy-paste augmentation given the image and the category of the fault
         you want to augment into the other image
         """

        if type(image) == 'numpy.ndarray':
            image = Image.fromarray(image)
        org_image = image.copy()
        id = random.choice(list(self.fault_database[category].keys())) # Choose a random
        print(id)
        mask = self.fault_database[category][id][0][1]
        image_with_fail = self.retrieve_image(id)
        augmented_image = self.insert_crop(image,image_with_fail,mask).copy()

        #Plot the results of aumentation
        if self.enable_plot:
            self.plot_augmentations(image_with_fail,augmented_image,org_image,mask)

        return augmented_image.copy()

    def insert_crop(self,image,image_with_fail,mask):

        # Get the mask and convert to PIL image, multiplying with 255 to get into scale.
        mask = Image.fromarray(mask*255)
        mask = mask.filter(ImageFilter.GaussianBlur(self.blur)) # Blur the edges

        # Get a random placement in the image
        x,y = self.get_random_placement(image) # get a valid placement in the image for the crop

        # Get random rotation
        rotation = self.get_random_rotation()
        image_with_fail = image_with_fail.rotate(rotation,expand = True)
        mask = mask.rotate(rotation,expand = True)

        #Paste the image into the image
        image.paste(im = image_with_fail, mask = mask)
        return image.copy()

    # Return the PIL image with the id and from the folder.
    def retrieve_image(self,id):
        return Image.open(f'{self.base_dir}/BachelorProject/Data/Serie1_raw_14Feb/CellsGS/Serie_1_ImageGS_{id}.png')

    def get_random_placement(self,image):
        dim = np.array(image).shape
        y = random.randint(0,dim[1])
        x = random.randint(0,dim[0])
        print(x,y)
        #x = dim[1]//2
        #y = dim[0]//2
        return x,y

    def plot_augmentations(self,image_with_fail,augmented_image,org_image,mask):
         # Bare for at test
        fig, ax = plt.subplots(1,4, figsize=(15,20))
        ax[0].imshow(mask,cmap = 'gray')
        ax[0].set_title("Mask")
        ax[1].imshow(np.array(image_with_fail),cmap = 'gray')
        ax[1].set_title("Image with fault")
        ax[2].imshow(np.array(augmented_image),cmap = 'gray')
        ax[2].set_title("Augmented Image")
        ax[3].imshow(np.array(org_image),cmap = 'gray')
        ax[3].set_title("Original Image")
        plt.show()

    def get_random_rotation(self):
        rotations = [0,180]
        return random.choice(rotations)



if __name__ == '__main__':

    # Create the data set
    fault_set,_ = create_data_set()
    cpa = CopyPasteAugmentation(fault_database=fault_set,
                                base_dir=base_dir,
                                blur=3,
                                enable_plot=True)
    image = Image.open(f'{base_dir}/BachelorProject/Data/Serie1_raw_14Feb/CellsGS/Serie_1_ImageGS_-5_4083_Cell_Row6_Col_3.png')
    im = cpa.get_augmentet_image(image=image, category='Finger Failure')
    #Display the image
    plt.imshow(np.array(im),cmap = 'gray')
    plt.show()





