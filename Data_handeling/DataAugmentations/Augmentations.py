from PIL import Image, ImageFilter
import numpy as np
import pandas as pd
import random
from matplotlib import pyplot as plt
#from PoisonBlend import blend
import cv2
from BachelorProject.image_poisson_blending import load_image, blend_image, preprocess


class BaseAugmentation:
    """ Base class for all custom augmentations that will be used in the project. """
    def __init__(self):
        self.data_set_csv_path = '/Users/madsandersen/PycharmProjects/BscProjektData/BachelorProject/Data/VitusData/Train_expanded.csv'
        self.data_set_csv = pd.read_csv(self.data_set_csv_path).set_index('ImageDir')
        self.crack_a = self.data_set_csv[self.data_set_csv['Label'] == 'CrackA']
        self.crack_b = self.data_set_csv[self.data_set_csv['Label'] == 'CrackB']
        self.crack_c = self.data_set_csv[self.data_set_csv['Label'] == 'CrackC']
        self.finger_failure = self.data_set_csv[self.data_set_csv['Label'] == 'FingerFailure']
        self.no_faults = self.data_set_csv[self.data_set_csv['Label'] == 'Negative']
        self.fault_database = {'Crack A': self.crack_a,
                               'Crack B': self.crack_b,
                               'Crack C': self.crack_c,
                               'Finger Failure': self.finger_failure}
        self.base_dir = '/Users/madsandersen/PycharmProjects/BscProjektData/BachelorProject/Data/'

    def augment_image(self,image,category):
        pass

    def load_image(self,image_path):
        return Image.open(image_path)

    def get_random_placement(self, image):
        """" Get at random valid placement in the image for the crop, so that the crop is not outside the image. """
        dim = np.array(image).shape
        x,y = random.randint(0+120, dim[0]-120), random.randint(0+120, dim[1]-120)
        print((x,y))
        return x, y

    def get_random_rotation(self,angels=[0,180]):
        return random.choice(angels)

    def load_mask(self,image_with_fail_path,category):
        category = category.replace(' ', '')
        if str(type(self.data_set_csv.loc[image_with_fail_path[72:], 'MaskDir'])) == "<class 'pandas.core.series.Series'>":
            ser = self.data_set_csv.loc[image_with_fail_path[72:]]
            dirs = ser[ser['Label'] == category]['MaskDir']
            #full_directory = f'{self.base_dir}{dirs}'
            mask = self.combine_masks(dirs)
            #mask = self.load_image(full_directory)
        else:
            dir = self.data_set_csv.loc[image_with_fail_path[72:], 'MaskDir']
            full_directory = f'{self.base_dir}{dir}'
            mask = self.load_image(full_directory)
        return mask

    def combine_masks(self,dirs):
        """" If an image has multiple masks, this function combines them into one mask, if the faults are the same """
        masks = [self.load_image(f'{self.base_dir}{dir}') for dir in dirs]
        mask = masks[0]
        for i in range(1,len(masks)):
            mask = Image.composite(masks[i],mask,masks[i])
        return mask

    def plot_image(self,mask,image_with_fail,augmented_image,org_image):
        # Bare for at test
        fs = 20
        fig, ax = plt.subplots(1, 4, figsize=(20, 5))
        #fig.suptitle(f'{self}', fontsize=16)
        ax[0].imshow(mask, cmap='gray')
        ax[0].set_title("Mask",fontsize=fs)
        ax[0].axis('off')

        ax[1].imshow(np.array(image_with_fail), cmap='gray')
        ax[1].set_title("Source Image", fontsize=fs)
        ax[1].axis('off')

        ax[2].imshow(np.array(augmented_image), cmap='gray')
        ax[2].set_title("Augmented Image", fontsize=fs)
        ax[2].axis('off')

        ax[3].imshow(np.array(org_image), cmap='gray')
        ax[3].set_title("Target Image", fontsize=fs)
        ax[3].axis('off')
        plt.show()

class GaussianCopyPaste(BaseAugmentation):

    def __init__(self,blur=5):
        super().__init__()
        self.blur = blur

    def augment_image(self,image,category,plot_image=True):
        """ Augment the image with a random image from the category."""
        org_image = image.copy()
        # Get a random image from the category
        image_with_fail_path = random.choice(list(self.fault_database[category].index))
        image_with_fail = self.load_image(f'{self.base_dir}{image_with_fail_path}')
        image_with_fail_mask = self.load_mask(f'{self.base_dir}{image_with_fail_path}',category)
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
        self.BLEND_TYPE = 2
        self.GRAD_MIX = True

    def augment_image(self,image,category,plot_image=False):
        """ Augment the image with a random image from the category."""
        org_image = image.copy()
        # Get a random image from the category
        image_with_fail_path = random.choice(list(self.fault_database[category].index))
        image_with_fail = self.load_image(f'{self.base_dir}{image_with_fail_path}')
        image_with_fail_mask = self.load_mask(f'{self.base_dir}{image_with_fail_path}',category)

        # Get a random placement in the image
        x,y = self.get_random_placement(image) # get a valid placement in the image for the crop

        # Get random rotation
        rotation = self.get_random_rotation()
        image_with_fail = image_with_fail.rotate(rotation,expand = True)
        image_with_fail_mask = image_with_fail_mask.rotate(rotation,expand = True)

        # Paste the image into the image
        image = self.poisson_blend_images(image,image_with_fail,image_with_fail_mask,(0,0))


        if plot_image:
            self.plot_image(image_with_fail_mask,image_with_fail,image,org_image)

        return image.copy(),image_with_fail_path

    def poisson_blend_images(self,image,image_with_fail,image_with_fail_mask,offset):
        """
        This implementation utilieses the openCV poisson blending function (seamlessClone) to blend a region of one image
        onto the other. The mask is used to define the region of the image to be blended.
        """

        # ready data
        image = np.array(image)
        image_with_fail = np.array(image_with_fail)
        image_with_fail_mask = np.array(image_with_fail_mask)

        #Poison bledning
        image_dat = load_image.load_image(image_with_fail, image_with_fail_mask, image, (0, 0))
        image_dat = preprocess.preprocess(image_dat)
        final_image = blend_image.blend_image(image_dat, self.BLEND_TYPE, self.GRAD_MIX)
        final_image = final_image*255
        augmented_image = Image.fromarray(final_image[:,:,0].astype(np.uint8))
        #plot_image(self,mask,image_with_fail,augmented_image,org_image):
        #self.plot_image(image_with_fail_mask,image_with_fail,final_image,image)

        return augmented_image.copy()

    def format_images(self,image,image_with_fail,image_with_fail_mask):
        """
        This method formats the images, such that they can be parsed directly to the blend function.
        image: PIL image of the image to be blended into
        image_with_fail: image containing the fault which is to be cropped out.
        image_with_fail_mask: the mask containing the region which is to be cropped out and pasted.
        """

        img_mask = np.array(image_with_fail_mask)
        img_mask.flags.writeable = True
        # img_mask = np.expand_dims(img_mask, axis=2)

        img_source = np.array(image_with_fail)
        img_source.flags.writeable = True
        img_source = np.expand_dims(img_source, axis=2)

        img_target = np.array(image)
        img_target = cv2.resize(img_target, (img_source.shape[1], img_source.shape[0])) #resize the target to the size of the source
        img_target.flags.writeable = True
        img_target = np.expand_dims(img_target, axis=2)

        return img_target, img_source, img_mask

    def __repr__(self):
        return "Poisson Copy Paste"

if __name__ == '__main__':
    random.seed(15)
    GausCP = PoisonCopyPaste()#GaussianCopyPaste(blur=5)
    #Poisson_CP = PoisonCopyPaste()
    image_dir = GausCP.no_faults.index[0]
    image = GausCP.load_image(image_dir)
    image = GausCP.augment_image(image,'Crack A',plot_image=True)
    #image.show()




