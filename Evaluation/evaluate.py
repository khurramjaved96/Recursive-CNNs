import os
import xml.etree.ElementTree as ET

import corner_refinement
import cv2
import getcorners
import numpy as np
import tensorflow as tf

if __name__ == '__main__':
    tf.reset_default_graph()

    
    
    corner_e=getcorners.get_corners_moreBG()
    #corner_e = getcorners.get_corners_aug()   
    model = corner_refinement.corner_finder_aug()
    dir = "/home/khurram/Dicta_data/temp"
    import csv
    ans = []
    ans2 = []

    with open('../bg1_2.csv', 'a') as csvfile:
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
                                    print "IMAGE NAME = ", images_dir + "/" + image
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

                                    myGt = np.asarray((list_of_points["tl"], list_of_points["tr"], list_of_points["br"],
                                                       list_of_points["bl"]))



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
                                    #print time.clock() - start

                                    
                                    corner_address = []
                                    print "Gets here"
                                    
                                    #temp = np.array(model1.get_location((data[0][0], data[1][0], data[2][0], data[3][0])))
                                   # print temp.shape
                                    #print temp

                                    for b in data:
                                        a = b[0]

                                        temp = np.array(model.get_location1(a))
                                        temp[0] += b[1]
                                        temp[1] += b[2]
                                        corner_address.append(temp)
                                       # print temp

                                    end = time.clock()
                                    print "TOTAL TIME : ", end - start
                                    # for a in range(0, len(data)):
                                    #     cv2.line(img, tuple(corner_address[a % 4]), tuple(corner_address[(a + 1) % 4]),
                                    #              (255, 0, 0), 2)
                                    #cv2.fillConvexPoly(img,np.array(corner_address),(255,0,0))
                                    #cv2.imwrite("../result1.jpg", img)


                                    #spamwriter.writerow((folder + file + image + ".jpg", np.array(corner_address))
                                    #spamwriter.writerow((folder + file + image + ".jpg", myGt))

                                    #print myGt
                                   # print np.array(corner_address)
                                    from datasetGenerator import utils

                                    #  r = utils.intersection(myGt, np.array(corner_address),img)
				    r2 = utils.intersection_with_corection(myGt, np.array(corner_address), img)
			         #   if r2>r:
				#	print "Good scene"
				 #   else:
				#	print "Bad scene"
                                    spamwriter.writerow((images_dir + "/" + image, np.array((tl, tr, br, bl)),np.array(corner_address),r2))
                                    #if r <0.7:
                                     #   cv2.imwrite("../"+image, img)
                                    #if r<1 and r>0:
                                    #    ans.append(r)
                                    if r2<1 and r2>0:
					ans2.append(r2)
				    print "MEAN CORRECTED: ", np.mean(np.array(ans2))
                                    #print "MEAN OLD : ", np.mean(np.array(ans))
    print np.mean(np.array(ans2))


