from PIL import Image, ImageFilter
import numpy as np
import pandas as pd
import random
from matplotlib import pyplot as plt
from cv2 import seamlessClone
import cv2

class BaseAugmentation:
    """ Base class for all coustoum augmentations that will be used in the project. """
    def __init__(self):
        self.data_set_csv_path = '/Users/madsandersen/PycharmProjects/BscProjektData/BachelorProject/Data/VitusData/DataSet.csv'
        self.data_set_csv = pd.read_csv(self.data_set_csv_path).set_index('ImageDir')
        self.crack_a = self.data_set_csv[self.data_set_csv['Label'] == 'Crack A']
        self.crack_b = self.data_set_csv[self.data_set_csv['Label'] == 'Crack B']
        self.crack_c = self.data_set_csv[self.data_set_csv['Label'] == 'Crack C']
        self.finger_failure = self.data_set_csv[self.data_set_csv['Label'] == 'Finger Failure']
        self.no_faults = self.data_set_csv[self.data_set_csv['Label'] == 'Negative']
        self.fault_database = {'Crack A': self.crack_a,
                               'Crack B': self.crack_b,
                               'Crack C': self.crack_c,
                               'Finger Failure': self.finger_failure}

    def augment_image(self,image,category):
        pass

    def load_image(self,image_path):
        return Image.open(image_path)

    def get_random_placement(self, image):
        """" Get at random valid placement in the image for the crop, so that the crop is not outside the image. """
        dim = np.array(image).shape
        x,y = random.randint(0, dim[0]), random.randint(0, dim[1])
        return x, y

    def get_random_rotation(self,angels=[0,180]):
        return random.choice(angels)

    def load_mask(self,image_with_fail_path):
        if str(type(self.data_set_csv.loc[image_with_fail_path, 'MaskDir'])) == "<class 'pandas.core.series.Series'>":
            mask = self.load_image(self.data_set_csv.loc[image_with_fail_path, 'MaskDir'][0])
        else:
            mask = self.load_image(self.data_set_csv.loc[image_with_fail_path, 'MaskDir'])
        return mask

    def plot_image(self,mask,image_with_fail,augmented_image,org_image):
        # Bare for at test
        fig, ax = plt.subplots(1, 4, figsize=(15, 20))
        ax[0].imshow(mask, cmap='gray')
        ax[0].set_title("Mask")
        ax[1].imshow(np.array(image_with_fail), cmap='gray')
        ax[1].set_title("Image with fault")
        ax[2].imshow(np.array(augmented_image), cmap='gray')
        ax[2].set_title("Augmented Image")
        ax[3].imshow(np.array(org_image), cmap='gray')
        ax[3].set_title("Original Image")
        plt.show()

class GaussianCopyPaste(BaseAugmentation):

    def __init__(self,blur=5):
        super().__init__()
        self.blur = blur

    def augment_image(self,image,category,plot_image=False):
        """ Augment the image with a random image from the category."""
        org_image = image.copy()
        # Get a random image from the category
        image_with_fail_path = random.choice(list(self.fault_database[category].index))
        image_with_fail = self.load_image(image_with_fail_path)
        image_with_fail_mask = self.load_mask(image_with_fail_path)
        image_with_fail_mask = image_with_fail_mask.filter(ImageFilter.GaussianBlur(self.blur))

        # Get a random placement in the image
        #x,y = self.get_random_placement(image) # get a valid placement in the image for the crop

        # Get random rotation
        rotation = self.get_random_rotation()
        image_with_fail = image_with_fail.rotate(rotation,expand = True)
        image_with_fail_mask = image_with_fail_mask.rotate(rotation,expand = True)

        # Paste the image into the image
        image.paste(im=image_with_fail, mask=image_with_fail_mask)

        if plot_image:
            self.plot_image(image_with_fail_mask,image_with_fail,image,org_image)

        return image.copy(),image_with_fail_path


class PoisonCopyPaste(BaseAugmentation):

    def __init__(self):
        super().__init__()

    def augment_image(self,image,category,plot_image=False):
        """ Augment the image with a random image from the category."""
        org_image = image.copy()
        # Get a random image from the category
        image_with_fail_path = random.choice(list(self.fault_database[category].index))
        image_with_fail = self.load_image(image_with_fail_path)
        image_with_fail_mask = self.load_mask(image_with_fail_path)

        # Get a random placement in the image
        x,y = self.get_random_placement(image) # get a valid placement in the image for the crop

        # Get random rotation
        rotation = self.get_random_rotation()
        image_with_fail = image_with_fail.rotate(rotation,expand = True)
        image_with_fail_mask = image_with_fail_mask.rotate(rotation,expand = True)

        # Paste the image into the image
        image = self.poisson_blend_images(image,image_with_fail,image_with_fail_mask,(x,y))


        if plot_image:
            self.plot_image(image_with_fail_mask,image_with_fail,image,org_image)

        return image.copy()

    def poisson_blend_images(self,image,image_with_fail,image_with_fail_mask,center):
        """
        This implementation utilieses the openCV poisson blending function (seamlessClone) to blend a region of one image
        onto the other. The mask is used to define the region of the image to be blended.
        """

        # Convert images to float32
        image = np.array(image, dtype=np.uint8)
        image_with_fail = np.array(image_with_fail, dtype=np.uint8)
        image_with_fail_mask = np.array(image_with_fail_mask, dtype=np.uint8)
        image_with_fail_mask = image_with_fail_mask*np.uint8(255)

        image = cv2.resize(image, (image_with_fail.shape[1], image_with_fail.shape[0]))
        image = cv2.seamlessClone(image_with_fail, image, image_with_fail_mask, (143,143), cv2.NORMAL_CLONE)

        #Convert back to PIL image
        image = Image.fromarray(image.astype(np.uint8))
        return image.copy()

    def pad_images2_size(self,img1,img2,mask):
        """ This function pads two images into the same size"""
        img1 = np.array(img1)
        img2 = np.array(img2)
        mask = np.array(mask)

        # Get the size of the images
        width1, height1 = img1.shape
        width2, height2 = img2.shape
        width_mask, height_mask = mask.shape

        # Get the max size
        max_width = max(width1,width2)
        max_height = max(height1,height2)

        # Create the new images
        new_img1 = np.zeros((max_width,max_height))
        new_img2 = np.zeros((max_width,max_height))
        new_mask = np.zeros((max_width,max_height))

        # Get the difference in size
        width_diff1 = max_width - width1
        height_diff1 = max_height - height1
        width_diff2 = max_width - width2
        height_diff2 = max_height - height2



        # Pad the images
        new_img1[width_diff1:,height_diff1:] = img1
        new_img2[width_diff2:,height_diff2:] = img2
        new_mask[width_diff2:,height_diff2:] = mask

        return new_img1,new_img2,new_mask



if __name__ == '__main__':
    random.seed(2)
    GausCP = PoisonCopyPaste()#GaussianCopyPaste(blur=5)
    #Poisson_CP = PoisonCopyPaste()
    image_dir = GausCP.no_faults.index[0]
    image = GausCP.load_image(image_dir)
    image = GausCP.augment_image(image,'Finger Failure',plot_image=True)
    #image.show()




