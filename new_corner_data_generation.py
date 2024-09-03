import albumentations as A
import numpy as np
from PIL import Image
import random

from matplotlib import pyplot as plt

import dataprocessor
import pandas as pd


class CornerCropper:

    def __init__(self, crop_width, crop_height):
        self.crop_width = crop_width
        self.crop_height = crop_height

    def get_starting_points(self, img, fixed_x, fixed_y):
        h, w = img.shape[:2]

        # Ensure the fixed point remains within the cropped area
        min_x = max(0, fixed_x - self.crop_width)
        max_x = min(fixed_x, w - self.crop_width)
        min_y = max(0, fixed_y - self.crop_height)
        max_y = min(fixed_y, h - self.crop_height)

        # Randomly choose the top-left corner within the valid range
        start_x = random.randint(min_x, max_x)
        start_y = random.randint(min_y, max_y)

        return start_x, start_y

    def crop(self, img, fixed_x, fixed_y):
        start_x, start_y = self.get_starting_points(img, fixed_x, fixed_y)

        cropped_img = img[ start_y:start_y+self.crop_height,start_x:start_x+self.crop_width]

        new_cordinates=(fixed_x-start_x,fixed_y-start_y)

        return cropped_img, new_cordinates
#%%
dataset_smart_doc_train= dataprocessor.DatasetFactory.get_dataset([r"C:\Users\isaac\PycharmProjects\document_localization\Recursive-CNNs\datasets\testDataset\smart-doc-train"], "document")
#%%

#%%
indexing=random.randint(0,len(dataset_smart_doc_train.myData[0])-1)
sample_path=dataset_smart_doc_train.myData[0][indexing]
sample_label=dataset_smart_doc_train.myData[1][indexing,:]
img=np.array(Image.open(sample_path))
fixed_point_x = int(sample_label[2]* img.shape[1])
fixed_point_y = int(sample_label[3]* img.shape[0])
crop_width = 200     # Width of the random crop
crop_height = 200    # Height of the random crop

plt.imshow(img)
plt.scatter(fixed_point_x, fixed_point_y)
plt.show()

cropper=CornerCropper(crop_width, crop_height)
cropped_img,new_cordinates=cropper.crop(img,fixed_point_x,fixed_point_y)
plt.imshow(cropped_img)
plt.scatter(new_cordinates[0], new_cordinates[1])
plt.show()