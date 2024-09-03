import os

import cv2
import numpy as np

import utils


def argsProcessor():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--dataPath", help="DataPath")
    parser.add_argument("-o", "--outputFiles", help="outputFiles", default="bar")
    return parser.parse_args()

args = argsProcessor()

output_dir = args.outputFiles
if (not os.path.isdir(output_dir)):
    os.mkdir(output_dir)

dir = args.dataPath
import csv

with open(output_dir+"/gt.csv", 'a') as csvfile:
    spamwriter_1 = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
    for image in os.listdir(dir):
        if image.endswith("jpg") or image.endswith("JPG"):
            if os.path.isfile(dir+"/"+image+".csv"):
                with open(dir+"/"+image+ ".csv", 'r') as csvfile:
                    spamwriter = csv.reader(csvfile, delimiter=' ',
                                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
                    img = cv2.imread(dir +"/"+ image)
                    print (image)
                    gt= []
                    for row in spamwriter:
                        gt.append(row)
                        # img = cv2.circle(img, (int(float(row[0])), int(float(row[1]))), 2,(255,0,0),90)
                    gt =np.array(gt).astype(np.float32)
                    gt = gt / (img.shape[1], img.shape[0])
                    gt = gt * (1080, 1080)
                    img = cv2.resize(img, (1080, 1080))


                    print (gt)

                    for angle in range(0,271,90):
                        img_rotate, gt_rotate = utils.rotate(img, gt, angle)
                        for random_crop in range(0,16):
                            img_crop, gt_crop = utils.random_crop(img_rotate, gt_rotate)
                            mah_size = img_crop.shape
                            img_crop = cv2.resize(img_crop, (64, 64))
                            gt_crop = np.array(gt_crop)

                            # gt_crop = gt_crop*(1.0 / mah_size[1],1.0 / mah_size[0])

                            # for a in range(0,4):
                            # no=0
                            # for a in range(0,4):
                            #     no+=1
                            #     cv2.circle(img_crop, tuple(((gt_crop[a]*64).astype(int))), 2,(255-no*60,no*60,0),9)
                            # # # cv2.imwrite("asda.jpg", img)

                            cv2.imwrite(output_dir + "/" +str(angle)+str(random_crop)+ image, img_crop)
                            spamwriter_1.writerow((str(angle)+str(random_crop)+ image, tuple(list(gt_crop))))

