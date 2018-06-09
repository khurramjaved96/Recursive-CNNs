import argparse
import time

import numpy as np
import torch

import DataLoader
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
parser.add_argument("-i", "--data-dir", default="/Users/khurramjaved96/smartdocframestest",
                    help="input Directory of test data")
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
if __name__ == '__main__':
    corner_refiner = getcorners.GetCorners("../documentModel")
    corner_detector = corner_refinement.corner_finder("../cornerModel2")
    test_set_dir = args.data_dir
    iou_results = []
    dataset_test = dataset.SmartDocDirectories(test_set_dir)
    for data_elem in dataset_test.myData:
        img_path = data_elem[0]
        target = data_elem[1].reshape((4,2))
        img = np.array(Image.open(img_path))
        start = time.clock()
        data = corner_refiner.get(img)
        corner_address = []
        for b in data:
            a = b[0]
            temp = np.array(corner_detector.get_location(a, 0.85))
            temp[0] += b[1]
            temp[1] += b[2]
            corner_address.append(temp)

        end = time.clock()
        print("TOTAL TIME : ", end - start)
        r2 = utils.intersection_with_corection(target, np.array(corner_address), img)

        assert (r2 > 0 and r2 < 1)
        iou_results.append(r2)
        print("MEAN CORRECTED: ", np.mean(np.array(iou_results)))

    print(np.mean(np.array(iou_results)))
