import os
from tqdm import tqdm

import cv2
import numpy as np
import utils
import dataprocessor

def args_processor():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input-dir", help="Path to data files (Extract images using video_to_image.py first")
    parser.add_argument("-o", "--output-dir", help="Directory to store results")
    parser.add_argument("--dataset", default="smartdoc", help="'smartdoc' or 'selfcollected' dataset")
    return parser.parse_args()


if __name__ == '__main__':
    if __name__ == '__main__':
        args = args_processor()
        input_directory = args.input_dir
        if not os.path.isdir(args.output_dir):
            os.mkdir(args.output_dir)
        import csv


        # Dataset iterator
        if args.dataset == "smartdoc":
            dataset_test = dataprocessor.dataset.SmartDocDirectories(input_directory)
        elif args.dataset == "selfcollected":
            dataset_test = dataprocessor.dataset.SelfCollectedDataset(input_directory)
        else:
            print("Incorrect dataset type; please choose between smartdoc or selfcollected")
            assert (False)
        with open(os.path.join(args.output_dir, 'gt.csv'), 'a') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter=',',
                                    quotechar='|', quoting=csv.QUOTE_MINIMAL)
            # Counter for file naming
            counter = 0
            for data_elem in tqdm(dataset_test.myData):

                img_path = data_elem[0]
                target = data_elem[1].reshape((4, 2))
                img = cv2.imread(img_path)

                if args.dataset == "selfcollected":
                    target = target / (img.shape[1], img.shape[0])
                    target = target * (1920, 1920)
                    img = cv2.resize(img, (1920, 1920))

                corner_cords = target

                for angle in range(0, 271, 90):
                    img_rotate, gt_rotate = utils.utils.rotate(img, corner_cords, angle)
                    for random_crop in range(0, 16):
                        counter += 1
                        f_name = str(counter).zfill(8)

                        img_crop, gt_crop = utils.utils.random_crop(img_rotate, gt_rotate)
                        mah_size = img_crop.shape
                        img_crop = cv2.resize(img_crop, (64, 64))
                        gt_crop = np.array(gt_crop)

                        # no=0
                        # for a in range(0,4):
                        #     no+=1
                        #     cv2.circle(img_crop, tuple(((gt_crop[a]*64).astype(int))), 2,(255-no*60,no*60,0),9)
                        # # cv2.imwrite("asda.jpg", img)

                        cv2.imwrite(os.path.join(args.output_dir, f_name+".jpg"), img_crop)
                        spamwriter.writerow((f_name+".jpg", tuple(list(gt_crop))))
