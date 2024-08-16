''' Document Localization using Recursive CNN
 Maintainer : Khurram Javed
 Email : kjaved@ualberta.ca '''

import os
import cv2
import numpy as np

import dataprocessor
from utils import utils
import matplotlib.pyplot as plt

def args_processor():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input-dir", help="Path to data files (Extract images using video_to_image.py first")
    parser.add_argument("-o", "--output-dir", help="Directory to store results")
    parser.add_argument("--dataset", default="smartdoc", help="'smartdoc' or 'selfcollected' dataset")
    return parser.parse_args()


if __name__ == '__main__':
    # args = args_processor()
    input_directory = r"C:\Users\isaac\PycharmProjects\document_localization\Recursive-CNNs\datasets\kosmos-dataset"
    output_dir=r"C:\Users\isaac\PycharmProjects\document_localization\Recursive-CNNs\corner-datasets\kosmos-test"

    dataset="smartdoc-dataset"

    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    import csv

    # Dataset iterator
    if dataset=="smartdoc":
        dataset_test = dataprocessor.dataset.SmartDocDirectories(input_directory)
    elif dataset=="selfcollected":
        dataset_test = dataprocessor.dataset.SelfCollectedDataset(input_directory)

    elif dataset=="smartdoc-dataset":
        dataset_test = dataprocessor.dataset.SmartDoc([input_directory],"test.csv")
    else:
        print ("Incorrect dataset type; please choose between smartdoc or selfcollected")
        assert(False)
    with open(os.path.join(output_dir, 'gt.csv'), 'a') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        # Counter for file naming
        counter = 0
        for path_img,label in zip(dataset_test.myData[0],dataset_test.myData[1]):

            img_path = path_img
            target = label.reshape((4, 2))
            img = cv2.imread(img_path)

            if dataset=="selfcollected" or dataset=="smartdoc-dataset":
                # target = target / (img.shape[1], img.shape[0])
                target = target * (1920, 1920)
                img = cv2.resize(img, (1920, 1920))
                # fig,ax=plt.subplots()
                # ax.imshow(img)
                # ax.scatter(target[:,1], target[:,0])
                # for idx in range(4):
                #     ax.text(target[idx,1],target[idx,0],["TL","TR","BR","BL"][idx])
                # plt.show()

            corner_cords = target

            for angle in range(0, 1, 90):

                img_rotate, gt_rotate = utils.rotate(img, corner_cords, angle)
                for random_crop in range(0, 1):
                    img_list, gt_list = utils.get_corners(img_rotate, gt_rotate)

                    for a in range(0, 4):

                        counter += 1
                        f_name = str(counter).zfill(8)+f"{a}_{angle}_{random_crop}"

                        gt_store = list(np.array(gt_list[a]) / (300, 300))
                        img_store = cv2.resize(img_list[a], (64, 64))
                        cv2.circle(img_store, tuple(list((np.array(gt_store)*64).astype(int))), 1, (255, 0, 0), 1)
                        print(os.path.join(output_dir, f_name + ".jpg"))
                        print("angle = ", angle)
                        print("random_crop = ", random_crop)
                        print("a=", a)
                        print(gt_list[a])
                        cv2.imwrite(os.path.join(output_dir, f_name + ".jpg"),
                                    img_store, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                        spamwriter.writerow((f_name + ".jpg", tuple(gt_store)))
