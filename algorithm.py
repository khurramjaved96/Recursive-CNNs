import cv2
import numpy as np
import corner_refinement
import getcorners
import tensorflow as tf

img = cv2.imread("/home/khurramjaved/Dicta/Test_data/1/paper001.avi/025.jpg",1)


result =np.copy(img)
data  =getcorners.get_corners(img)
corner_address=[]
for b in data:

    a = b[0]

    print a.shape
    tf.reset_default_graph()
    temp = np.array(corner_refinement.get_location(a))

    temp[0]+= b[1]
    temp[1]+= b[2]
    corner_address.append(temp)

#img = cv2.resize(img, (800,800))
for a in range(0,len(data)):
    cv2.line(img, tuple(corner_address[a%4]), tuple(corner_address[(a+1)%4]),(255,0,0),2)

cv2.imwrite("../result.jpg", img)
