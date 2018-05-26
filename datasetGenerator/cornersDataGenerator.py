import os
import numpy as np
import cv2
import csv
import utils

def argsProcessor():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--outputDir", help="output Directory of Data")
    parser.add_argument("-i", "--inputDir", help="input Directory of data")
    return  parser.parse_args()


args = argsProcessor()
outputDir = args.outputDir
inputDir = args.inputDir
if (not os.path.isdir(outputDir)):
    os.mkdir(outputDir)

import csv

with open(outputDir+ "/gt.csv", 'a') as csvfile:
    spamwriter_1 = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
    for image in os.listdir(inputDir):
        if image.endswith("jpg"):
            if os.path.isfile(inputDir+ "/"+image+ ".csv"):
                with open(inputDir+ "/"+image+ ".csv", 'r') as csvfile:
                    spamwriter = csv.reader(csvfile, delimiter=' ',
                                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
                    img = cv2.imread(inputDir + "/" + image)
                    print (image)
                    gt= []
                    for row in spamwriter:
                        gt.append(row)
                        # img = cv2.circle(img, (int(float(row[0])), int(float(row[1]))), 2,(255,0,0),90)
                    gt =np.array(gt).astype(np.float32)


                    # print gt
                    gt = gt / (img.shape[0], img.shape[1])

                    gt = gt * (1080, int((1080.0 / img.shape[0] * img.shape[1])))

                    img = cv2.resize(img, ( int((1080.0/img.shape[0]*img.shape[1])),1080))
                    # for a in range(0,4):
                    #     img = cv2.circle(img, tuple((gt[a].astype(int))), 2, (255, 0, 0), 9)
                    # cv2.imwrite("asda.jpg", img)
                    # 0/0
                    for angle in range(0,271,90):
                        img_rotate, gt_rotate = utils.rotate(img, gt,angle)
                        for random_crop in range(0,1):
                            img_list, gt_list = utils.getCorners(img_rotate, gt_rotate)
                            for a in range(0,4):
                                cv2.circle(img_list[a], tuple(gt_list[a]), 2,(255,0,0),2)
                                cv2.imwrite(outputDir + "/" + image + str(angle) + str(random_crop) + str(a) + ".jpg", img_list[a])
                                spamwriter_1.writerow(( image + str(angle) +str(random_crop) + str(a) +".jpg", tuple(gt_list[a])))
