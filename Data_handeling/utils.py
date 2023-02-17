import numpy as np
import pandas as pd
import os
import random
from PIL import Image

base_dir = '/Users/madsandersen/PycharmProjects/BscProjektData'

def get_image_name(file_name):
    return file_name[17:-4]

def trim_image(image,mask):
    """ Returns a trimmed down image of only the mask so it is ready for copy-pasting into the other image"""
    if type(image) != 'numpy.ndarray':
        image = np.array(image)

    image = image*mask #Multiply the binary mask with each index, which will give give zero everywhere else but the fault
    image = image[~np.all(image == 0, axis=1)] #remove zero rows

    #Remove zero colloumns
    idx = np.argwhere(np.all(image[..., :] == 0, axis=0))
    image = np.delete(image, idx, axis=1)

    return image

