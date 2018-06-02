import cv2
import numpy as np
import math
import random

def __rotateImage(image, angle):
  rot_mat = cv2.getRotationMatrix2D((image.shape[1]/2,image.shape[0]/2),angle,1)
  result = cv2.warpAffine(image, rot_mat, (image.shape[1], image.shape[0]),flags=cv2.INTER_LINEAR)
  return result, rot_mat

def rotate(img, gt, angle):

    img, mat = __rotateImage(img,angle)
    gt = gt.astype(np.float64)
    for a in range(0,4):
        gt[a] = np.dot(mat[...,0:2], gt[a]) + mat[...,2]
    a=float(angle) * math.pi / 180

    # gt[0,0] = gt[3,0] = img.shape[1]*math.sin(float(angle)*math.pi/180)*cos(a)
    # gt[0,1] = gt[1,1] = R2 math.sin(a) - W*(sin(a)**2)
    # gt[1,0] = gt[2,0] = gt[0,0]+H
    # gt[2,1] = gt[3,1] = gt[1,1] + W
    # gt = gt.astype(np.int32)
    return img, gt

def random_crop(img, gt):
    ptr1 = (min(gt[0][0], gt[1][0], gt[2][0], gt[3][0]),
            min(gt[0][1], gt[1][1], gt[2][1], gt[3][1]))

    ptr2 = ((max(gt[0][0], gt[1][0], gt[2][0], gt[3][0]),
             max(gt[0][1], gt[1][1], gt[2][1], gt[3][1])))

    start_x = np.random.randint(0, int(max(ptr1[0] - 1,1)))
    start_y = np.random.randint(0, int(max(ptr1[1] - 1,1)))

    end_x = np.random.randint(int(min(ptr2[0] + 1,img.shape[1]-1)), img.shape[1])
    end_y = np.random.randint(int(min(ptr2[1] + 1,img.shape[0]-1)), img.shape[0])

    myGt = gt - (start_x, start_y)
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
    y_zero = int(slope*(0-br[0])+br[1])
    if y_pred < ptr1[1] and y_zero<ptr1[1]:
        bl = ptr1
        tr = myGt[(tl_index + 3) % 4]
    else:
        tr = ptr1
        bl = myGt[(tl_index + 3) % 4]
    # print tl, tr, br, bl

    img = img[start_y:end_y, start_x:end_x]
    return img, (tl,tr,br,bl)

def getCorners(img,gt):
    gt = gt.astype(int)
    list_of_points={}
    myGt=gt
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
    y_zero = int(slope*(0-br[0])+br[1])
    if y_pred < ptr1[1] and y_zero<ptr1[1]:
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
    gt_list=[]
    images_list=[]
    for k, v in list_of_points.iteritems():

        if (k == "tl"):
            cords_x = __get_cords(v[0], 0, list_of_points["tr"][0], buf=10, size=abs(list_of_points["tr"][0] - v[0]))
            cords_y = __get_cords(v[1], 0, list_of_points["bl"][1], buf=10, size=abs(list_of_points["bl"][1] - v[1]))
            # print cords_y, cords_x
            gt = (v[0] - cords_x[0], v[1] - cords_y[0])

            cut_image = img[cords_y[0]:cords_y[1], cords_x[0]:cords_x[1]]

        if (k == "tr"):
            cords_x = __get_cords(v[0], list_of_points["tl"][0], img.shape[1], buf=10,
                                size=abs(list_of_points["tl"][0] - v[0]))
            cords_y = __get_cords(v[1], 0, list_of_points["br"][1], buf=10, size=abs(list_of_points["br"][1] - v[1]))
            # print cords_y, cords_x
            gt = (v[0] - cords_x[0], v[1] - cords_y[0])

            cut_image = img[cords_y[0]:cords_y[1], cords_x[0]:cords_x[1]]

        if (k == "bl"):
            cords_x = __get_cords(v[0], 0, list_of_points["br"][0], buf=10,
                                size=abs(list_of_points["br"][0] - v[0]))
            cords_y = __get_cords(v[1], list_of_points["tl"][1], img.shape[0], buf=10,
                                size=abs(list_of_points["tl"][1] - v[1]))
            # print cords_y, cords_x
            gt = (v[0] - cords_x[0], v[1] - cords_y[0])

            cut_image = img[cords_y[0]:cords_y[1], cords_x[0]:cords_x[1]]

        if (k == "br"):
            cords_x = __get_cords(v[0], list_of_points["bl"][0], img.shape[1], buf=10,
                                size=abs(list_of_points["bl"][0] - v[0]))
            cords_y = __get_cords(v[1], list_of_points["tr"][1], img.shape[0], buf=10,
                                size=abs(list_of_points["tr"][1] - v[1]))
            # print cords_y, cords_x
            gt = (v[0] - cords_x[0], v[1] - cords_y[0])

            cut_image = img[cords_y[0]:cords_y[1], cords_x[0]:cords_x[1]]

        # cv2.circle(cut_image, gt, 2, (255, 0, 0), 6)
        mah_size = cut_image.shape
        cut_image = cv2.resize(cut_image, (300, 300))
        a = int(gt[0] * 300 / mah_size[1])
        b = int(gt[1] * 300 / mah_size[0])
        images_list.append(cut_image)
        gt_list.append((a,b))
    return images_list, gt_list
def __get_cords(cord, min_start, max_end, size = 299 , buf = 5, random_scale=True):
    #size = max(abs(cord-min_start), abs(cord-max_end))
    iter=0
    if(random_scale):
        size/= random.randint(1,4)
    while(max_end - min_start)<size:
        size=size*.9
    x_start = random.randint(int(min(max(min_start, cord-size+buf), cord-buf-1)), cord-buf)
    while(x_start+size> max_end):
        x_start = random.randint(int(min(max(min_start,int(cord - size + buf)),cord-buf-1)), cord - buf)
        size=size*.999
        iter+=1
        if(iter==1000):
            x_start= min_start
            break
    return (x_start, int(x_start+size))