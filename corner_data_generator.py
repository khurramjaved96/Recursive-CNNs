''' Document Localization using Recursive CNN
 Maintainer : Khurram Javed
 Email : kjaved@ualberta.ca '''

import os
import cv2
import numpy as np

import dataprocessor
from utils import utils


def args_processor():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input-dir", help="Path to data files (Extract images using video_to_image.py first")
    parser.add_argument("-o", "--output-dir", help="Directory to store results")
    return parser.parse_args()


if __name__ == '__main__':
    args = args_processor()
    input_directory = args.input_dir
    if not os.path.isdir(args.output_dir):
        os.mkdir(args.output_dir)
    import csv

    # Dataset iterator
    dataset_test = dataprocessor.dataset.SmartDocDirectories(input_directory)
    with open(os.path.join(args.output_dir, 'gt.csv'), 'a') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        # Counter for file naming
        counter = 0
        for data_elem in dataset_test.myData:

            img_path = data_elem[0]
            target = data_elem[1].reshape((4, 2))
            img = cv2.imread(img_path)
            corner_cords = target

            for angle in range(0, 1):
                img_rotate, gt_rotate = utils.rotate(img, corner_cords, angle)
                for random_crop in range(0, 1):
                    img_list, gt_list = utils.get_corners(img_rotate, gt_rotate)
                    for a in range(0, 4):
                        counter += 1
                        f_name = str(counter).zfill(8)
                        print(gt_list[a])
                        gt_store = list(np.array(gt_list[a]) / (300, 300))
                        img_store = cv2.resize(img_list[a], (64, 64))
                        # cv2.circle(img_store, tuple(list((np.array(gt_store)*64).astype(int))), 2, (255, 0, 0), 2)

                        cv2.imwrite(os.path.join(args.output_dir, f_name + ".jpg"),
                                    img_store, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                        spamwriter.writerow((f_name + ".jpg", tuple(gt_store)))
