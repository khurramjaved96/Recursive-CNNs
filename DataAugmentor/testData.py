import os
import numpy as np
import cv2
import csv

dir = "../data1/"
for image in os.listdir(dir):
    if image.endswith("jpg") or image.endswith("JPG"):
        if os.path.isfile(dir+image+".csv"):
            with open(dir+image+ ".csv", 'r') as csvfile:
                spamwriter = csv.reader(csvfile, delimiter=' ',
                                        quotechar='|', quoting=csv.QUOTE_MINIMAL)
                img = cv2.imread(dir + image)
                no = 0
                for row in spamwriter:
                    no+=1
                    print row
                    img = cv2.circle(img, (int(float(row[0])), int(float(row[1]))), 2,(255-no*60,no*60,0),90)
                img = cv2.resize(img, (300,300))
                cv2.imshow("a",img)
                cv2.waitKey(0)