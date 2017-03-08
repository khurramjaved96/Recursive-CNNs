import os
import cv2
import xml.etree.ElementTree as ET
import numpy as np
import cv2
import numpy as np
import corner_refinement
import getcorners
import tensorflow as tf

import random

if __name__ == '__main__':
    dir = "/home/khurramjaved/Dicta/bg1"
    import csv
    ans = []
    with open('../gt.csv', 'a') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for folder in os.listdir(dir):
            a = 0
            print str(folder)
            if (os.path.isdir(dir + "/" + folder)):
                for file in os.listdir(dir + "/" + folder):
                    images_dir = dir + "/" + folder + "/" + file
                    if (os.path.isdir(images_dir)):

                        list_gt = []
                        tree = ET.parse(images_dir + "/" + file + ".gt")
                        root = tree.getroot()
                        for a in root.iter("frame"):
                            list_gt.append(a)

                        im_no=0
                        for image in os.listdir(images_dir):
                            if image.endswith(".jpg"):
                                    print im_no
                                    im_no+=1

                                    # Now we have opened the file and GT. Write code to create multiple files and scale gt
                                    list_of_points = {}
                                    img = cv2.imread(images_dir + "/" + image)
                                    print "IMAGE NAME = ", image 
                                    for point in list_gt[int(float(image[0:-4])) - 1].iter("point"):
                                        myDict = point.attrib

                                        list_of_points[myDict["name"]] = (
                                        int(float(myDict['x'])), int(float(myDict['y'])))
                                    299

                                    doc_height = min(list_of_points["bl"][1] - list_of_points["tl"][1],
                                                     list_of_points["br"][1] - list_of_points["tr"][1])
                                    doc_width = min(list_of_points["br"][0] - list_of_points["bl"][0],
                                                    list_of_points["tr"][0] - list_of_points["tl"][0])
                                    #  print doc_height
                                    #  print doc_height
                                    #  print doc_width
                                    import random

                                    myGt = np.asarray((list_of_points["tl"], list_of_points["tr"], list_of_points["br"],
                                                       list_of_points["bl"]))

                                    import time

                                    result = np.copy(img)
                                    tf.reset_default_graph()
                                    corner_e = getcorners.get_corners()
                                    start = time.clock()
                                    data = corner_e.get(img)
                                    #print time.clock() - start

                                    tf.reset_default_graph()

                                    model = corner_refinement.corner_finder()
                                    corner_address = []
                                    import timeit

                                    start = timeit.timeit()
                                    start = time.clock()
                                    for b in data:
                                        a = b[0]

                                        temp = np.array(model.get_location(a))
                                        temp[0] += b[1]
                                        temp[1] += b[2]
                                        corner_address.append(temp)
                                       # print temp

                                    end = time.clock()
                                    #print "TOTAL TIME : ", end - start
                                    for a in range(0, len(data)):
                                        cv2.line(img, tuple(corner_address[a % 4]), tuple(corner_address[(a + 1) % 4]),
                                                 (255, 0, 0), 2)
                                    cv2.fillConvexPoly(img,np.array(corner_address),(255,0,0))
                                    cv2.imwrite("../result1.jpg", img)
                                    #spamwriter.writerow((folder + file + image + ".jpg", np.array(corner_address))
                                    #spamwriter.writerow((folder + file + image + ".jpg", myGt))

                                    #print myGt
                                   # print np.array(corner_address)
                                    import utils
                                    r = utils.intersection(myGt, np.array(corner_address),img)
                                    ans.append(r)
                                    print "MEAN : ", np.mean(np.array(ans))
    print np.mean(np.array(ans))


