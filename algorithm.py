import cv2
import numpy as np
import corner_refinement
import getcorners
import tensorflow as tf

img = cv2.imread("../temp/044.jpg",1)

result =np.copy(img)
data  =getcorners.get_corners(img)
corner_address=[]
for a in data:

    print a.shape
    tf.reset_default_graph()
    corner_address.append(corner_refinement.get_location(a))

for a in range(0,len(data)):
    cv2.circle(data[a], tuple(corner_address[a]),2,(255,0,0),2)
    cv2.imshow("asd", data[a])
    cv2.waitKey(0)
print corner_address
print len(data)