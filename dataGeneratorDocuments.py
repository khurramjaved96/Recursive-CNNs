import numpy as np
import cv2
import tensorflow as tf
import utils.utils as utils
import os

def argsProcessor():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--outputDir", help="output Directory of Data")
    parser.add_argument("-i", "--inputDir", help="input Directory of data")
    parser.add_argument("-s", "--saveName", help="fileNameForSaving")
    return  parser.parse_args()

import math

args = argsProcessor()
inputDataDir = args.inputDir
outputDataDir = args.outputDir

GT_DIR = inputDataDir + "/gt.csv"

VALIDATION_PERCENTAGE = .2
TEST_PERCENTAGE = .01
Debug = True
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3
size = (224,224)


image_list, gt_list, file_name = utils.load_data_4(inputDataDir, GT_DIR, limit=-1, size=size)
image_list, gt_list = utils.unison_shuffled_copies(image_list, gt_list)


print len(image_list)


if (Debug):
    print ("(Image_list_len, gt_list_len)", (len(image_list), len(gt_list)))
train_image = image_list[0:max(1, int(len(image_list) * (1 - VALIDATION_PERCENTAGE)))]
validate_image = image_list[int(len(image_list) * (1 - VALIDATION_PERCENTAGE)):len(image_list) - 1]

train_gt = gt_list[0:max(1, int(len(image_list) * (1 - VALIDATION_PERCENTAGE)))]
validate_gt = gt_list[int(len(image_list) * (1 - VALIDATION_PERCENTAGE)):len(image_list) - 1]
if (Debug):
    print ("(Train_Image_len, Train_gt_len)", (len(train_image), len(train_gt)))
    print ("(Validate_Image_len, Validate_gt_len)", (len(validate_image), len(validate_gt)))
for a in range(0, 10):
    temp_image=np.copy(image_list[a])
    for b in range(0, 4):
        cv2.circle(temp_image, (gt_list[a][b*2], gt_list[a][b*2+1]), 2, (255, 0, 0), 4)
    cv2.imwrite("../temp"+str(a)+".jpg", temp_image)


np.save(outputDataDir + "/"+ args.saveName + "trainGt", train_gt)
np.save(outputDataDir + "/"+args.saveName + "trainImages", train_image)
np.save(outputDataDir + "/"+args.saveName + "validateGt", validate_gt)
np.save(outputDataDir + "/"+args.saveName + "validateImages", validate_image)

# 0/0
