import os
import xml.etree.ElementTree as ET

import numpy as np
import tensorflow as tf
from PIL import Image

import Evaluation.corner_refinement as corner_refinement
import Evaluation.getcorners as getcorners
from utils import utils

if __name__ == '__main__':
    tf.reset_default_graph()

    corner_e = getcorners.GetCorners()
    model = corner_refinement.corner_finder()
    dir = "/Users/khurramjaved96/smartdocframestest"
    import csv

    ans = []
    ans2 = []

    with open('../bg1_2.csv', 'a') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for folder in os.listdir(dir):
            a = 0
            print(str(folder))
            if (os.path.isdir(dir + "/" + folder)):
                for file in os.listdir(dir + "/" + folder):
                    images_dir = dir + "/" + folder + "/" + file
                    if (os.path.isdir(images_dir)):

                        list_gt = []
                        tree = ET.parse(images_dir + "/" + file + ".gt")
                        root = tree.getroot()
                        for a in root.iter("frame"):
                            list_gt.append(a)

                        im_no = 0
                        for image in os.listdir(images_dir):
                            if image.endswith(".jpg"):
                                print(im_no)
                                im_no += 1

                                # Now we have opened the file and GT. Write code to create multiple files and scale gt
                                list_of_points = {}

                                # img = cv2.imread(images_dir + "/" + image)
                                img = np.array(Image.open(images_dir + "/" + image))
                                print("IMAGE NAME = ", images_dir + "/" + image)
                                for point in list_gt[int(float(image[0:-4])) - 1].iter("point"):
                                    myDict = point.attrib

                                    list_of_points[myDict["name"]] = (
                                        int(float(myDict['x'])), int(float(myDict['y'])))
                                299

                                doc_height = min(list_of_points["bl"][1] - list_of_points["tl"][1],
                                                 list_of_points["br"][1] - list_of_points["tr"][1])
                                doc_width = min(list_of_points["br"][0] - list_of_points["bl"][0],
                                                list_of_points["tr"][0] - list_of_points["tl"][0])

                                myGt = np.asarray((list_of_points["tl"], list_of_points["tr"], list_of_points["br"],
                                                   list_of_points["bl"]))

                                myGtTemp = myGt * myGt
                                sum_array = myGtTemp.sum(axis=1)
                                tl_index = np.argmin(sum_array)
                                tl = myGt[tl_index]
                                br = myGt[(tl_index + 2) % 4]
                                ptr1 = myGt[(tl_index + 1) % 4]

                                slope = (float(tl[1] - br[1])) / float(tl[0] - br[0])
                                # print "SLOPE = ", slope
                                y_pred = int(slope * (ptr1[0] - br[0]) + br[1])
                                y_zero = int(slope * (0 - br[0]) + br[1])
                                if y_pred < ptr1[1]:
                                    bl = ptr1
                                    tr = myGt[(tl_index + 3) % 4]
                                else:
                                    tr = ptr1
                                    bl = myGt[(tl_index + 3) % 4]

                                list_of_points["tr"] = tr
                                list_of_points["tl"] = tl
                                list_of_points["br"] = br
                                list_of_points["bl"] = bl

                                myGt = np.asarray((list_of_points["tl"], list_of_points["tr"], list_of_points["br"],
                                                   list_of_points["bl"]))

                                import time

                                result = np.copy(img)

                                start = time.clock()
                                data = corner_e.get(img)
                                # print time.clock() - start


                                corner_address = []
                                print("Gets here")

                                for b in data:
                                    a = b[0]

                                    temp = np.array(model.get_location(a))
                                    temp[0] += b[1]
                                    temp[1] += b[2]
                                    corner_address.append(temp)

                                end = time.clock()
                                print("TOTAL TIME : ", end - start)
                                r2 = utils.intersection_with_corection(myGt, np.array(corner_address), img)
                                spamwriter.writerow((images_dir + "/" + image, np.array((tl, tr, br, bl)),
                                                     np.array(corner_address), r2))
                                if r2 < 1 and r2 > 0:
                                    ans2.append(r2)
                                print("MEAN CORRECTED: ", np.mean(np.array(ans2)))
    print(np.mean(np.array(ans2)))
