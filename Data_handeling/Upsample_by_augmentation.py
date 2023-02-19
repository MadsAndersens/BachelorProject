from Augmentation import CopyPasteAugmentation
import numpy as np
from PIL import Image, ImageFilter
from data_formatting import create_data_set
from utils import get_image_name, trim_image, base_dir, get_none_faulty_images
import random
from tqdm import tqdm

def run_augmentation(fault_database,base_dir,n_upsamples = 10):
    """ Run the augmentation on the images in the database. """
    augmentet_images = {category:[] for category in fault_database.keys()}
    none_faults = get_none_faulty_images()

    aug = CopyPasteAugmentation(fault_database,base_dir,False)
    for category in fault_database.keys():
        for i in tqdm(range(n_upsamples)):
            # Get a random none faulty image
            image_name = random.choice(none_faults)
            image = Image.open(f'{base_dir}/BachelorProject/Data/Serie1_raw_14Feb/CellsGS/{image_name}')
            ref_image = image.copy()
            image = aug.get_augmentet_image(image,category).copy()

            #Save the image to the folder
            image.save(f'{base_dir}/BachelorProject/Data/AugmentedImages/{category}/{image_name[:-4]}_aug{i}.png')





if __name__ == '__main__':
    fault_database,_ = create_data_set()
    run_augmentation(fault_database,base_dir,n_upsamples=2)
