import numpy as np
import PIL.Image as Image
from PIL import ImageFilter
import pandas as pd

# performs naive cut-paste from source to target
def naive_copy(image_data):
  # extract image data
  source = image_data['source']
  mask = image_data['mask']
  target = image_data['target']
  dims = image_data['dims']
  
  target[dims[0]:dims[1],dims[2]:dims[3],:] = target[dims[0]:dims[1],dims[2]:dims[3],:] * (1 - mask) + source * mask
  
  return target

if __name__ == '__main__':
  data = pd.read_csv('/Users/madsandersen/PycharmProjects/BscProjektData/BachelorProject/Data/VitusData/Train.csv')
  print(data['ImageDir'])

  # Load image
  #image = Image.open('image.jpg')