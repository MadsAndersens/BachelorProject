from PIL import Image, ImageFilter
import numpy as np
import pandas as pd
import random
from matplotlib import pyplot as plt
#from PoisonBlend import blend
import cv2
from BachelorProject.image_poisson_blending import load_image, blend_image, preprocess

#set the seed
np.random.seed(42)

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
            idx = np.random.randint(0,len(dirs))
            mask_dir = dirs[idx]#self.combine_masks(dirs)
            full_directory = f'{self.base_dir}{mask_dir}'
            mask = cv2.imread(full_directory)
            #mask = self.load_image(full_directory)
        else:
            dir = self.data_set_csv.loc[image_with_fail_path[72:], 'MaskDir']
            full_directory = f'{self.base_dir}{dir}'
            mask = cv2.imread(full_directory)#self.load_image(full_directory)
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

    def augment_image(self,target_path,category,plot_image=False):
        """ Augment the image with a random image from the category."""
        #org_image = image.copy()
        # Get a random image from the category
        image_with_fail_path = random.choice(list(self.fault_database[category].index))
        #image_with_fail = self.load_image(f'{self.base_dir}{image_with_fail_path}')
        #image_with_fail_mask = self.load_mask(f'{self.base_dir}{image_with_fail_path}',category)

        paths = random.choice(list(self.fault_database[category].index))

        #Get dirs and mask
        dirsrc = f'{self.base_dir}{random.choice(list(self.fault_database[category].index))}'
        dirdst = f'{self.base_dir}{target_path}'
        mask = self.load_mask(dirsrc,category)

        #Load src and dst
        src = cv2.imread(dirsrc)
        dst = cv2.imread(dirdst)

        synthetic_image = None
        failed = 0
        while synthetic_image is None:
            try:
                synthetic_image,synthetic_mask = self.gaussian_blend(src, dst, mask)
            except:
                failed += 1

            if failed > 10000:
                synthetic_image = src
                break
        print(failed)
        #synthetic_image = Image.fromarray(np.uint8(synthetic_image))
        if plot_image:
            self.plot_image(mask,src,synthetic_image,dst)
        return synthetic_image,image_with_fail_path,synthetic_mask

    def eq_guassian_blend(self,src,dst,mask,offset,blur=5):

        # Get the new random center
        center_blob = self.find_blob_center(mask)

        offset = (offset[0]-center_blob[0],offset[1]-center_blob[1])

        src_pil = Image.fromarray(src)
        mask_pil = Image.fromarray(mask[:, :, 0])
        dst_pil = Image.fromarray(dst)

        # Create a zero array of shape dst
        zero_mask = Image.fromarray(np.zeros(dst.shape[:2]))
        # Paste the mask
        zero_mask.paste(im=mask_pil, box=offset)

        # Apply gasussian blur to mask
        mask_pil = mask_pil.filter(ImageFilter.GaussianBlur(blur))

        # Blend the images
        dst_pil.paste(im=src_pil, mask=mask_pil, box=offset)
        return dst_pil.copy(), mask_pil.copy(),offset,zero_mask.copy()

    def gaussian_blend(self,src, dst, mask, blur=5):
        """ Blend src and dst using a gaussian mask. """
        #Randomly rotate the mask and src
        mask, src = self.random_rotation_180(mask, src)

        #Random scale the mask and src
        scaled_mask, scaled_src = self.random_scale(src, mask, [0.7, 1])

        # Apply gasussian blur to mask
        mask = cv2.GaussianBlur(mask, (blur, blur), blur)

        # Find the center of the mask
        center_blob = self.find_blob_center(mask)

        # Get the new random center
        center = self.random_clone_center(scaled_mask, dst)

        #Find the offset of the mask to the new center
        offset = (center[0]-center_blob[0],center[1]-center_blob[1])

        #Roll the mask and the source image to the new center
        #mask = np.roll(mask,offset,axis=(0,1))
        #src = np.roll(src,offset,axis=(0,1))

        src_pil = Image.fromarray(src)
        mask_pil = Image.fromarray(mask[:,:,0])
        dst_pil = Image.fromarray(dst)

        # Blend the images
        dst_pil.paste(im = src_pil, mask = mask_pil,box = offset)
        return dst_pil.copy(),mask_pil.copy()

    def find_blob_center(self,mask):
        # Find the contours in the mask
        contours, _ = cv2.findContours(mask[:,:,0], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # If no contours were found, return None
        if len(contours) == 0:
            return None

        # Find the largest contour by area
        contour = max(contours, key=cv2.contourArea)

        # Find the moments of the contour
        moments = cv2.moments(contour)

        # Calculate the x and y coordinates of the center of the contour
        center_x = int(moments["m10"] / moments["m00"])
        center_y = int(moments["m01"] / moments["m00"])

        return (center_x, center_y)

    def random_scale(self,image, mask, scale_range):
        # Generate a random scale factor
        scale_factor = np.random.uniform(*scale_range)

        # Compute the scaled image
        scaled_image = cv2.resize(image, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)

        # Compute the scaled mask
        scaled_mask = cv2.resize(mask, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)
        return scaled_mask, scaled_image

    def random_rotation_180(self,mask, img):
        # Generate a random angle between 0 and 1
        angle = np.random.randint(0, 2)

        # Rotate the mask and image by 180 degrees if angle is 1
        if angle == 1:
            mask = cv2.rotate(mask, cv2.ROTATE_180)
            img = cv2.rotate(img, cv2.ROTATE_180)
            #print('test')
        return mask, img

    def random_clone_center(self,mask, dst):
        # Get the size of the destination image
        h, w = dst.shape[:2]
        height, width, _ = np.where(mask > 0)

        roi_height = (max(height) - min(height)) // 2
        roi_width = (max(width) - min(width)) // 2

        # Generate a random center point for the cloned region
        x = np.random.randint(roi_width + 1, w - roi_width - 1)
        y = np.random.randint(roi_height + 1, h - roi_height - 1)

        #print(x, y)

        # Add half the width and height of the region to the random point
        center = (x, y)

        return center

class PoisonCopyPaste(BaseAugmentation):

    def __init__(self):
        super().__init__()
        self.BLEND_TYPE = 2
        self.GRAD_MIX = True

    def augment_image(self,target_path,category,plot_image=False):
        """ Augment the image with a random image from the category."""
        #org_image = image.copy()
        # Get a random image from the category
        image_with_fail_path = random.choice(list(self.fault_database[category].index))
        #image_with_fail = self.load_image(f'{self.base_dir}{image_with_fail_path}')
        #image_with_fail_mask = self.load_mask(f'{self.base_dir}{image_with_fail_path}',category)

        paths = random.choice(list(self.fault_database[category].index))

        #Get dirs and mask
        dirsrc = f'{self.base_dir}{random.choice(list(self.fault_database[category].index))}'
        dirdst = f'{self.base_dir}{target_path}'
        mask = self.load_mask(dirsrc,category)

        #Load src and dst
        src = cv2.imread(dirsrc)
        dst = cv2.imread(dirdst)

        synthetic_image = None
        failed = 0
        while synthetic_image is None:
            try:
                synthetic_image,synthetic_mask = self.poisson_blend(src, dst, mask)
            except Exception as e:
                failed += 1
                return None,None,None

            #if failed > 1000:
            #    #This is in the rare case that it fails to blend 1000 times
            #    synthetic_image = [Image.fromarray(src),Image.fromarray(src)]
            #    synthetic_mask = Image.fromarray(mask)
            #    print(failed)
            #    break
        #synthetic_image = Image.fromarray(np.uint8(synthetic_image))
        #synthetic_mask = Image.fromarray(np.uint8(synthetic_mask))
        if plot_image:
            self.plot_image(mask,src,synthetic_image,dst)
        return synthetic_image,image_with_fail_path,synthetic_mask

    def poisson_blend(self,src, dst, mask,gaussian = True):
        mask, src = self.random_rotation_180(mask, src)
        # print((src.shape,mask.shape))
        scaled_mask, scaled_src = self.random_scale(src, mask, [0.9, 1])
        offset = self.random_clone_center(scaled_mask, dst)
        #print(offset)
        result = cv2.seamlessClone(scaled_src, dst, scaled_mask, offset, cv2.NORMAL_CLONE)
        result = Image.fromarray(np.uint8(result))
        #self.plot_image(scaled_mask, scaled_src, result, dst)

        if gaussian:
            aug = GaussianCopyPaste()
            g_aug,_,ofs,new_mask = aug.eq_guassian_blend(scaled_src,dst,scaled_mask,offset)
            new_mask = np.array(new_mask, dtype=np.uint8)
            new_mask = Image.fromarray(new_mask)
            # Pad the mask to the size of the target image
        return [result,g_aug], new_mask

    def pad_mask(self, mask,dst):
        max_widt, max_height =dst.shape[1], dst.shape[0]
        width, height = mask.shape[2], mask.shape[1]
        pad_widt, pad_height = max_widt - width, max_height - height
        image = np.pad(mask, ((0, 0), (0, pad_height), (0, pad_widt)), 'constant')
        return image

    def random_scale(self,image, mask, scale_range):
        # Generate a random scale factor
        scale_factor = np.random.uniform(*scale_range)

        # Compute the scaled image
        scaled_image = cv2.resize(image, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)

        # Compute the scaled mask
        scaled_mask = cv2.resize(mask, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)
        return scaled_mask, scaled_image

    def random_rotation_180(self,mask, img):
        # Generate a random angle between 0 and 1
        angle = np.random.randint(0, 2)

        # Rotate the mask and image by 180 degrees if angle is 1
        if angle == 1:
            mask = cv2.rotate(mask, cv2.ROTATE_180)
            img = cv2.rotate(img, cv2.ROTATE_180)
            #print('test')
        return mask, img

    def random_clone_center(self,mask, dst):
        # Get the size of the destination image
        h, w = dst.shape[:2]
        height, width, _ = np.where(mask > 0)

        roi_height = (max(height) - min(height)) // 2
        roi_width = (max(width) - min(width)) // 2
        #print(roi_height)
        #print(roi_width)

        # Generate a random center point for the cloned region
        x = np.random.randint(roi_width + 1, w - roi_width - 1)
        y = np.random.randint(roi_height + 1, h - roi_height - 1)

        #print(x, y)

        # Add half the width and height of the region to the random point
        center = (x, y)

        return center


    def __repr__(self):
        return "Poisson Copy Paste"

if __name__ == '__main__':
    random.seed(15)
    GausCP = PoisonCopyPaste()#GaussianCopyPaste(blur=5)
    #Poisson_CP = PoisonCopyPaste()
    image_dir = GausCP.no_faults.index[0]
    #image = GausCP.load_image(image_dir)
    image = GausCP.augment_image(image_dir,'Crack A',plot_image=True)
    #image.show()




