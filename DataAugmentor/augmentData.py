import os

import cv2
import numpy as np

from datasetGenerator import utils

output_dir = "multipleBackgrounds"
if (not os.path.isdir(output_dir)):
    os.mkdir(output_dir)

dir = "../data1"
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
                    print image
                    gt= []
                    for row in spamwriter:
                        gt.append(row)
                        # img = cv2.circle(img, (int(float(row[0])), int(float(row[1]))), 2,(255,0,0),90)
                    gt =np.array(gt).astype(np.float32)
                    gt = gt / (img.shape[1], img.shape[0])
                    gt = gt * (1080, 1080)
                    img = cv2.resize(img, (1080, 1080))


                    print gt

                    for angle in range(0,271,90):
                        img_rotate, gt_rotate = utils.rotate(img, gt, angle)
                        for random_crop in range(0,32):
                            img_crop, gt_crop = utils.random_crop(img_rotate, gt_rotate)
                            mah_size = img_crop.shape
                            img_crop = cv2.resize(img_crop, (300, 300))
                            gt_crop = np.array(gt_crop)

                            gt_crop = gt_crop*(300.0 / mah_size[1],300.0 / mah_size[0])

                            # for a in range(0,4):
                            no=0
                            # for a in range(0,4):
                            #     no+=1
                            #     img_crop = cv2.circle(img_crop, tuple((gt_crop[a].astype(int))), 2,(255-no*60,no*60,0),9)
                            # cv2.imwrite("asda.jpg", img)
                            # 0/0
                            cv2.imwrite(output_dir + "/" +str(angle)+str(random_crop)+ image, img_crop)
                            spamwriter_1.writerow((str(angle)+str(random_crop)+ image, gt_crop))

                            # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                            # cl1 = clahe.apply(img_crop)
                            #
                            # cv2.imwrite(output_dir + "/" + dir + "cli" + image, img_crop)
                            # spamwriter_1.writerow((dir + "cli" + image, gt_crop))
                            # for row_counter in range(0,4):
                            #     row=gt_crop[row_counter]
                            #     print row
                            #     img_crop = cv2.circle(img_crop, (int(float(row[0])), int(float(row[1]))), 2, (255, 0, 0), 2)
                            # img_temp = cv2.resize(img_temp, (300, 300))
