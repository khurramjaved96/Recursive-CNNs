import argparse
import time

import numpy as np
import torch

import DataLoader.dataset as dataset
import Evaluation.corner_refinement as corner_refinement
import Evaluation.getcorners as getcorners
from utils import utils
from PIL import Image

parser = argparse.ArgumentParser(description='iCarl2.0')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--debug', action='store_true', default=True,
                    help='Debug messages')
parser.add_argument('--model-type', default="resnet",
                    help='model type to be used. Example : resnet32, resnet20, densenet, test')
parser.add_argument('--name', default="noname",
                    help='Name of the experiment')
parser.add_argument('--outputDir', default="../",
                    help='Directory to store the results; a new folder "DDMMYYYY" will be created '
                         'in the specified directory to save the results.')
parser.add_argument('--dataset', default="document", help='Dataset to be used; example CIFAR, MNIST')
parser.add_argument('--loader', default="hdd", help='Dataset to be used; example CIFAR, MNIST')
parser.add_argument("-i", "--data-dir", default="/Users/khurramjaved96/bg5",
                    help="input Directory of test data")
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
if __name__ == '__main__':
    corners_extractor = getcorners.GetCorners("../documentModel")
    corner_refiner = corner_refinement.corner_finder("../cornerModel2")
    test_set_dir = args.data_dir
    iou_results = []
    dataset_test = dataset.SmartDocDirectories(test_set_dir)
    for data_elem in dataset_test.myData:
        img_path = data_elem[0]
        target = data_elem[1].reshape((4,2))
        img_array = np.array(Image.open(img_path))
        computation_start_time = time.clock()
        extracted_corners = corners_extractor.get(img_array)
        corner_address = []
        # Refine the detected corners using corner refiner
        for corner in extracted_corners:
            corner_img = corner[0]
            refined_corner = np.array(corner_refiner.get_location(corner_img, 0.85))
            # Converting from local co-ordinate to global co-ordinate of the image
            refined_corner[0] += corner[1]
            refined_corner[1] += corner[2]
            # Final results
            corner_address.append(refined_corner)

        computation_end_time = time.clock()
        print("TOTAL TIME : ", computation_end_time - computation_start_time)
        r2 = utils.intersection_with_corection(target, np.array(corner_address), img_array)

        assert (r2 > 0 and r2 < 1)
        iou_results.append(r2)
        print("MEAN CORRECTED: ", np.mean(np.array(iou_results)))

    print(np.mean(np.array(iou_results)))
