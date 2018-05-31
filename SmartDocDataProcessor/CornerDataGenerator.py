import os
import cv2
import xml.etree.ElementTree as ET
import numpy as np
from tqdm._tqdm import tqdm
import random
w = 150
h = 150

def argsProcessor():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataPath", help="Path to data files (Extract images using video_to_image.py first")
    parser.add_argument("-o", "--outputFiles", help="Directory to store results")
    return  parser.parse_args()



def get_cords(cord, min_start, max_end, size = 299 , buf = 0, random_scale=True):
    #size = max(abs(cord-min_start), abs(cord-max_end))

    iter=0
    if(random_scale):
        size/= random.randint(1,4) 
    while(max_end - min_start)<size:
        size=size*.9
    x_start = random.randint(min(int(max(min_start, cord-size+buf)), int(cord-buf-1)), int(cord-buf))
    while(x_start+size> max_end):
        x_start = random.randint(min(int(max(min_start,int(cord - size + buf))),int(cord-buf-1)), int(cord - buf))
        size=size*.999
        iter+=1
        if(iter==1000):
            x_start= min_start
            break 
    return (x_start, int(x_start+size))

if __name__ == '__main__':
    args = argsProcessor()
    dir = args.dataPath
    if (not os.path.isdir(args.outputFiles)):
        os.mkdir(args.outputFiles)
    import csv

    with open(args.outputFiles + 'gt.csv', 'a') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for folder in tqdm(os.listdir(dir)):
            a=0
            # print (str(folder))
            if(os.path.isdir(dir+"/"+folder)):
                for file in tqdm(os.listdir(dir+"/"+folder)):
                    images_dir= dir+"/"+folder+"/"+file
                    if(os.path.isdir(images_dir)):
                                   
                        list_gt = []
                        tree = ET.parse(images_dir+"/"+file+".gt")
                        root = tree.getroot()
                        for a in root.iter("frame"):
                            list_gt.append(a)

                        # print (list_gt)
                        for image in os.listdir(images_dir):
                            if image.endswith(".jpg"):
                                try:
                                    #Now we have opened the file and GT. Write code to create multiple files and scale gt
                                    list_of_points = {}
                                    img = cv2.imread(images_dir+"/"+image)
                                    # print (image[0:-4])
                                    for point in list_gt[int(float(image[0:-4]))-1].iter("point"):
                                        myDict = point.attrib

                                        list_of_points[myDict["name"]] = (int(float(myDict['x'])), int(float(myDict['y'])))
                                    299

                                    doc_height = min(list_of_points["bl"][1] - list_of_points["tl"][1], list_of_points["br"][1] - list_of_points["tr"][1])
                                    doc_width = min(list_of_points["br"][0] - list_of_points["bl"][0], list_of_points["tr"][0] - list_of_points["tl"][0])
                                #  print doc_height
                                #  print doc_height
                                #  print doc_width
                                    import random

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

                                    for k,v in list_of_points.items():

                                        if(k=="tl"):

                                            cords_x = get_cords(v[0], 0,list_of_points["tr"][0], buf =10, size=abs(list_of_points["tr"][0]-v[0]))
                                            cords_y = get_cords(v[1], 0,list_of_points["bl"][1],buf=10, size=abs(list_of_points["bl"][1]-v[1]))
                                        #    print cords_y, cords_x
                                            gt= (v[0] - cords_x[0], v[1] - cords_y[0])

                                            cut_image = img[cords_y[0]:cords_y[1], cords_x[0]:cords_x[1]]



                                        if (k == "tr"):
                                            cords_x = get_cords(v[0], list_of_points["tl"][0], img.shape[1], buf=10, size=abs(list_of_points["tl"][0]-v[0]))
                                            cords_y = get_cords(v[1], 0, list_of_points["br"][1], buf=10, size=abs(list_of_points["br"][1]-v[1]))
                                        #   print cords_y, cords_x
                                            gt = (v[0] - cords_x[0], v[1] - cords_y[0])

                                            cut_image = img[cords_y[0]:cords_y[1], cords_x[0]:cords_x[1]]


                                        if (k == "bl"):
                                            cords_x = get_cords(v[0], 0, list_of_points["br"][0], buf=10,
                                                                                                size=abs(list_of_points["br"][0] - v[0]))
                                            cords_y = get_cords(v[1], list_of_points["tl"][1], img.shape[0], buf=10,
                                                                size=abs(list_of_points["tl"][1] - v[1]))
                                        #    print cords_y, cords_x
                                            gt = (v[0] - cords_x[0], v[1] - cords_y[0])

                                            cut_image = img[cords_y[0]:cords_y[1], cords_x[0]:cords_x[1]]

                                        if (k == "br"):
                                            cords_x = get_cords(v[0], list_of_points["bl"][0], img.shape[1], buf=10,
                                                                size=abs(list_of_points["bl"][0] - v[0]))
                                            cords_y = get_cords(v[1], list_of_points["tr"][1],img.shape[0], buf=10,
                                                                size=abs(list_of_points["tr"][1] - v[1]))
                                            #print cords_y, cords_x
                                            gt = (v[0] - cords_x[0], v[1] - cords_y[0])

                                            cut_image = img[cords_y[0]:cords_y[1], cords_x[0]:cords_x[1]]

                                        #cv2.circle(cut_image, gt, 2, (255, 0, 0), 6)
                                        mah_size = cut_image.shape
                                        cut_image = cv2.resize(cut_image, (64,64))
                                        a = round(float(gt[0]/float(mah_size[1])),4)
                                        b = round(float(gt[1]/float(mah_size[0])),4)
                                    
                                  
                                        cv2.imwrite(args.outputFiles +folder+file+image+k+".jpg", cut_image,[int(cv2.IMWRITE_JPEG_QUALITY), 80])
                                        spamwriter.writerow((folder+file+image+k+".jpg",(a,b)))
                                except KeyboardInterrupt:
                                    raise
                                # except:
                                #     print("Exception occured; skipping this image")
                        3

