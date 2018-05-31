import os
import xml.etree.ElementTree as ET

import cv2
import numpy as np


def argsProcessor():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataPath", help="DataPath")
    parser.add_argument("-o", "--outputFiles", help="outputFiles", default="bar")
    return parser.parse_args()


if __name__ == '__main__':
    args = argsProcessor()
    dir = args.dataPath
    output_dir = args.outputFiles
    if (not os.path.isdir(output_dir)):
        os.mkdir(output_dir)
    import csv

    with open(args.outputFiles + "/gt.csv", 'a') as csvfile:
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
                            if a.attrib["rejected"] != "false":
                                print("Some frames are not valid")
                                0 / 0
                            list_gt.append(a)

                        # print list_gt
                        for image in os.listdir(images_dir):
                            if image.endswith(".jpg"):
                                try:
                                    # Now we have opened the file and GT. Write code to create multiple files and scale gt
                                    list_of_points = {}
                                    img = cv2.imread(images_dir + "/" + image)
                                    # print image[0:-4]
                                    for point in list_gt[int(float(image[0:-4])) - 1].iter("point"):
                                        myDict = point.attrib

                                        list_of_points[myDict["name"]] = (
                                            int(float(myDict['x'])), int(float(myDict['y'])))

                                    doc_height = min(list_of_points["bl"][1] - list_of_points["tl"][1],
                                                     list_of_points["br"][1] - list_of_points["tr"][1])
                                    doc_width = min(list_of_points["br"][0] - list_of_points["bl"][0],
                                                    list_of_points["tr"][0] - list_of_points["tl"][0])

                                    ptr1 = (
                                    min(list_of_points["tl"][0], list_of_points["bl"][0], list_of_points["tr"][0],
                                        list_of_points["br"][0]),
                                    min(list_of_points["tr"][1], list_of_points["tl"][1], list_of_points["br"][1],
                                        list_of_points["bl"][1]))

                                    ptr2 = (
                                    max(list_of_points["tl"][0], list_of_points["bl"][0], list_of_points["tr"][0],
                                        list_of_points["br"][0]),
                                    max(list_of_points["tr"][1], list_of_points["tl"][1], list_of_points["br"][1],
                                        list_of_points["bl"][1]))

                                    start_x = np.random.randint(0, ptr1[0] - 2)
                                    start_y = np.random.randint(0, ptr1[1] - 2)

                                    end_x = np.random.randint(ptr2[0] + 2, img.shape[1])
                                    end_y = np.random.randint(ptr2[1] + 2, img.shape[0])

                                    myGt = np.asarray((list_of_points["tl"], list_of_points["tr"], list_of_points["br"],
                                                       list_of_points["bl"]))
                                    myGt = myGt - (start_x, start_y)
                                    sum_array = myGt.sum(axis=1)
                                    tl_index = np.argmin(sum_array)
                                    tl = myGt[tl_index]
                                    br = myGt[(tl_index + 2) % 4]
                                    ptr1 = myGt[(tl_index + 1) % 4]
                                    # print "TL : ", tl
                                    # print "BR : ", br
                                    # print myGt
                                    # print myGt.shape
                                    slope = (float(tl[1] - br[1])) / float(tl[0] - br[0])
                                    # print "SLOPE = ", slope
                                    y_pred = int(slope * (ptr1[0] - br[0]) + br[1])
                                    if y_pred < ptr1[1]:
                                        bl = ptr1
                                        tr = myGt[(tl_index + 3) % 4]
                                    else:
                                        tr = ptr1
                                        bl = myGt[(tl_index + 3) % 4]
                                    # print tl, tr, br, bl


                                    img = img[start_y:end_y, start_x:end_x]
                                    tl = [float(a) for a in tl]
                                    tr = [float(a) for a in tr]
                                    br = [float(a) for a in br]
                                    bl = [float(a) for a in bl]

                                    tl[0] /= float(img.shape[1])
                                    tl[1] /= float(img.shape[0])

                                    tr[0] /= float(img.shape[1])
                                    tr[1] /= float(img.shape[0])

                                    br[0] /= float(img.shape[1])
                                    br[1] /= float(img.shape[0])

                                    bl[0] /= float(img.shape[1])
                                    bl[1] /= float(img.shape[0])

                                    tl = [round(a, 4) for a in tl]
                                    tr = [round(a, 4) for a in tr]
                                    br = [round(a, 4) for a in br]
                                    bl = [round(a, 4) for a in bl]

                                    img = cv2.resize(img, (64, 64))
                                    cv2.imwrite(output_dir + "/" + folder + file + image, img)
                                    spamwriter.writerow((folder + file + image, (tl, tr, br, bl)))


                                except KeyboardInterrupt:
                                    raise
