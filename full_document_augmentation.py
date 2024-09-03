
import albumentations as A
import numpy as np
from PIL import Image
import random
import os
import cv2
from matplotlib import pyplot as plt
from sympy.core.random import shuffle

import dataprocessor
import pandas as pd

dataset_smart_doc_train= dataprocessor.DatasetFactory.get_dataset(
    [r"C:\Users\isaac\PycharmProjects\document_localization\Recursive-CNNs\datasets\testDataset\smart-doc-train"],
    "document")
loader_smart_doc_train =dataprocessor.LoaderFactory.get_loader("hdd", dataset_smart_doc_train.myData,
                                              transform=None,
                                              cuda=False)


BASE_OUTPUT_PATH=r"C:\Users\isaac\PycharmProjects\document_localization\Recursive-CNNs\full_document_datasets\complete_documents\smart-doc\complete_doc"

NUMBER_OF_DOCUMENTS_with_50=500
NUMBER_OF_DOCUMENTS_with_100=500
NUMBER_OF_DOCUMENTS_with_150=500


#%%%

def crop_image_with_keypoints(image, include_points, exclude_point, padding=10):
    # Calculate the bounding box for the include points
    x_min = min([p[0] for p in include_points])
    x_max = max([p[0] for p in include_points])
    y_min = min([p[1] for p in include_points])
    y_max = max([p[1] for p in include_points])

    # Add padding to the bounding box
    x_min = max(x_min - padding, 0)
    x_max = min(x_max + padding, image.shape[1])
    y_min = max(y_min - padding, 0)
    y_max = min(y_max + padding, image.shape[0])

    # Ensure the exclude point is not in the crop
    while (x_min <= exclude_point[0] <= x_max) and (y_min <= exclude_point[1] <= y_max):
        # Adjust the bounding box to exclude the fourth point
        if exclude_point[0] < (x_max + x_min) / 2:
            x_min = exclude_point[0] + 1
        else:
            x_max = exclude_point[0] - 1

        if exclude_point[1] < (y_max + y_min) / 2:
            y_min = exclude_point[1] + 1
        else:
            y_max = exclude_point[1] - 1

    # Crop the image
    cropped_image = image[y_min:y_max, x_min:x_max]

    # Adjust the include points to the new cropped image
    # adjusted_include_points = [(p[0] - x_min, p[1] - y_min) for p in include_points]

    return cropped_image
#%%
sample=loader_smart_doc_train[random.randint(0,len(loader_smart_doc_train)-1)]

def create_crop(sample,number_of_corners,padding):
    img = np.array(sample[0])

    label=sample[1]#.reshape((4,2))
    x_cords = label[[0, 2, 4, 6]] * img.shape[1]
    y_cords = label[[1, 3, 5, 7]] * img.shape[0]


    transform = A.Compose(
        [A.Rotate(limit=360, p=1.0)],  # Rotate the image with a random angle between -45 and 45 degrees
        keypoint_params=A.KeypointParams(format='xy')  # Specify the keypoints format
    )

    keypoints=[(x,y) for x,y in zip(x_cords,y_cords)]

    transformed = transform(image=img, keypoints=keypoints)

    rotated_image = transformed['image']
    rotated_keypoints = transformed['keypoints']


    random.shuffle(rotated_keypoints)

    keep_points=rotated_keypoints
    exclude_cordinate=(img.shape[1],img.shape[0])

    # #%%
    cropped_image=crop_image_with_keypoints(rotated_image, keep_points, exclude_cordinate,padding)
    return cropped_image


#%%


for _ in range(NUMBER_OF_DOCUMENTS_with_50):
    sample = loader_smart_doc_train[random.randint(0, len(loader_smart_doc_train) - 1)]
    try:
        cut_img=create_crop(sample,4,padding=10)
        image = Image.fromarray(cut_img)
        print("NUMBER_OF_3_CORNERS:",_)
    except:
        pass
    # Save the image as a JPEG file
    image.save(os.path.join(BASE_OUTPUT_PATH,f'{_}_{50}.jpg'), format='JPEG')



for _ in range(NUMBER_OF_DOCUMENTS_with_100):
    sample = loader_smart_doc_train[random.randint(0, len(loader_smart_doc_train) - 1)]
    try:
        cut_img=create_crop(sample,4,padding=100)
        image = Image.fromarray(cut_img)
        print("NUMBER_OF_3_CORNERS:",_)
    except:
        pass
    # Save the image as a JPEG file
    image.save(os.path.join(BASE_OUTPUT_PATH,f'{_}_{100}.jpg'), format='JPEG')



for _ in range(NUMBER_OF_DOCUMENTS_with_150):
    sample = loader_smart_doc_train[random.randint(0, len(loader_smart_doc_train) - 1)]
    try:
        cut_img=create_crop(sample,4,padding=150)
        image = Image.fromarray(cut_img)
        print("NUMBER_OF_3_CORNERS:",_)
    except:
        pass
    # Save the image as a JPEG file
    image.save(os.path.join(BASE_OUTPUT_PATH,f'{_}_{150}.jpg'), format='JPEG')

