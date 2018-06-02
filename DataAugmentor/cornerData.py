import os

import cv2
import numpy as np

from datasetGenerator import utils

output_dir = "./multipleBackgroundsCorners"
if (not os.path.isdir(output_dir)):
    os.mkdir(output_dir)

dir = "../data1"
import csv

with open(output_dir+"/gt.csv", 'a') as csvfile:
    spamwriter_1 = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
    for image in os.listdir(dir):
        if image.endswith("jpg"):
            if os.path.isfile(dir+"/"+image+".csv"):
                with open(dir+"/"+image+ ".csv", 'r') as csvfile:
                    spamwriter = csv.reader(csvfile, delimiter=' ',
                                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
                    img = cv2.imread(dir +"/"+ image)
                    print image
                    gt= []
                    for row in spamwriter:
                        gt.append(row)
                        # img = cv2.circle(img, (int(float(row[0])), int(float(row[1]))), 2,(255,0,0),90)
                    gt =np.array(gt).astype(np.float32)


                    # print gt
                    gt = gt / (img.shape[1], img.shape[0])

                    gt = gt * (1080, 1080)

                    img = cv2.resize(img, ( 1080,1080))
                    # for a in range(0,4):
                    #     img = cv2.circle(img, tuple((gt[a].astype(int))), 2, (255, 0, 0), 9)
                    # cv2.imwrite("asda.jpg", img)
                    # 0/0
                    for angle in range(0,271,90):
                        img_rotate, gt_rotate = utils.rotate(img, gt, angle)
                        for random_crop in range(0,1):
                            img_list, gt_list = utils.getCorners(img_rotate, gt_rotate)
                            for a in range(0,4):
                                cv2.circle(img_list[a], tuple(gt_list[a]), 2,(255,0,0),2)
                                cv2.imwrite( output_dir+"/"+image + str(angle) +str(random_crop) + str(a) +".jpg", img_list[a])
                                spamwriter_1.writerow(( image + str(angle) +str(random_crop) + str(a) +".jpg", tuple(gt_list[a])))
