''' Document Localization using Recursive CNN
 Maintainer : Khurram Javed
 Email : kjaved@ualberta.ca '''

import os
import cv2
import numpy as np

import dataprocessor
from utils import utils
import matplotlib.pyplot as plt
import random
from PIL import Image

class CornerCropper:

    def __init__(self, crop_width, crop_height):
        self.crop_width = crop_width
        self.crop_height = crop_height

    def get_starting_points(self, img, fixed_x, fixed_y):
        h, w = img.shape[:2]
        print(h,w)
        print(fixed_x,fixed_y)
        # Ensure the fixed point remains within the cropped area
        min_x = max(0, fixed_x - self.crop_width)
        max_x = min(fixed_x, w - self.crop_width)
        min_y = max(0, fixed_y - self.crop_height)
        max_y = min(fixed_y, h - self.crop_height)

        # Randomly choose the top-left corner within the valid range
        start_x = random.randint(int(min_x), int(max_x))
        start_y = random.randint(int(min_y), int(max_y))

        return start_x, start_y

    def crop(self, img, fixed_x, fixed_y):
        start_x, start_y = self.get_starting_points(img, fixed_x, fixed_y)

        cropped_img = img[start_y:start_y+self.crop_height,start_x:start_x+self.crop_width]

        new_cordinates=(fixed_x-start_x,fixed_y-start_y)

        return cropped_img, new_cordinates


def rotate_image(image,cords ,angle):
    height,width=image.shape[:2]
    #cords = cords * np.array([width, height])
    new_cords = cords.copy()
    if angle==90:
        image=Image.fromarray(image).rotate(90,expand=True)
        new_cords[:,0]=cords[:,1]
        new_cords[:,1]=width-cords[:,0]
    elif angle==180:
        image=Image.fromarray(image).rotate(180,expand=True)
        new_cords[:, 0] = width- cords[:, 0]
        new_cords[:, 1] = height - cords[:, 1]
    elif angle==270:
        image=Image.fromarray(image).rotate(270,expand=True)
        new_cords[:, 0] = height- cords[:, 1]
        new_cords[:, 1] = cords[:, 0]
    elif angle==0:
        image=Image.fromarray(image)


    return np.array(image),new_cords


def args_processor():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input-dir", help="Path to data files (Extract images using video_to_image.py first")
    parser.add_argument("-o", "--output-dir", help="Directory to store results")
    parser.add_argument("--dataset", default="smartdoc", help="'smartdoc' or 'selfcollected' dataset")
    return parser.parse_args()


if __name__ == '__main__':
    # args = args_processor()
    input_directory = r"C:\Users\danie\OneDrive\Desktop\Trabajo Kosmos\Recursive-CNNs\data\self collected"
    output_dir=r"C:\Users\danie\OneDrive\Desktop\Trabajo Kosmos\Recursive-CNNs\corner-datasets\self-collected-train"

    dataset="selfcollected"
    csv_type="train.csv"

    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    import csv

    # Dataset iterator
    if dataset=="smartdoc":
        dataset_test = dataprocessor.dataset.SmartDocDirectories(input_directory)
    elif dataset=="selfcollected":
        dataset_test = dataprocessor.dataset.SelfCollectedDataset(input_directory)

    elif dataset=="smartdoc-dataset":
        dataset_test = dataprocessor.dataset.SmartDoc([input_directory],csv_type)
    else:
        print ("Incorrect dataset type; please choose between smartdoc or selfcollected")
        assert(False)
        
    with open(os.path.join(output_dir, 'gt.csv'), 'a') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        # Counter for file naming
        counter = 0

        corner_croper=CornerCropper(300,300)

        # corregir error: se está accediendo mal al dataset
        for path_img, label in dataset_test.myData:
            img_path = path_img
            target = label.reshape((4, 2))
            img = np.array(Image.open(img_path))

            #if dataset=="selfcollected" or dataset=="smartdoc-dataset":
                # target = target / (img.shape[1], img.shape[0])
                # target = target * (1920, 1920)
                # img = cv2.resize(img, (1920, 1920))
                #img = cv2.resize(img, (1920, 1920)) # creo que tambien debemos cambiar las coordenada de los bounding boxes
                # fig,ax=plt.subplots()
                # ax.imshow(img)
                # ax.scatter(target[:,1], target[:,0])
                # for idx in range(4):
                #     ax.text(target[idx,1],target[idx,0],["TL","TR","BR","BL"][idx])
                # plt.show()

            corner_cords = target

            for angle in [0,90,180,270]:

                img_rotate, gt_rotate = rotate_image(img, corner_cords, angle)

                # plt.imshow(img_rotate)
                # plt.scatter(gt_rotate[:,0],gt_rotate[:,1])
                # plt.show()
                print()
                for corner_idx in range(0, 4):

                    corner=gt_rotate[corner_idx,:]

                    for random_crop in range(3):
                        img_list, cordinate = corner_croper.crop(img_rotate,corner[0],corner[1] )
                        # plt.imshow(img_list)
                        # plt.scatter(cordinate[0],cordinate[1])
                        # plt.show()
                        counter += 1
                        f_name = str(counter).zfill(8)+f"{corner_idx}_{angle}"

                        gt_store = list(np.array(cordinate) / (300, 300))
                        img_store = cv2.resize(img_list, (100, 100))

                        # cv2.circle(img_store, tuple(list((np.array(gt_store)*64).astype(int))), 1, (255, 0, 0), 1)
                        # plt.imshow(img_store)
                        # plt.show()
                        print(os.path.join(output_dir, f_name + ".jpg"))
                        print("angle = ", angle)
                        print("random_crop = ", random_crop)
                        print("corner_idx=", corner_idx)
                        cv2.imwrite(os.path.join(output_dir, f_name + ".jpg"),
                                    img_store, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                        spamwriter.writerow((f_name + ".jpg", tuple(gt_store)))
