import cv2
import numpy as np
import corner_refinement
import getcorners
import tensorflow as tf

img = cv2.imread("abc.jpg",1)

import time
result =np.copy(img)
corner_e = getcorners.get_corners_aug()
start = time.clock()
data  =corner_e.get(img)
print time.clock() - start

tf.reset_default_graph()

model = corner_refinement.corner_finder_aug()
corner_address=[]
import timeit

start = timeit.timeit()
start = time.clock()
counter = 0
for b in data:
    a = b[0]
    cv2.imwrite(str(counter)+".jpg", a)
    temp = np.array(model.get_location1(a))
    temp[0]+= b[1]
    temp[1]+= b[2]
    corner_address.append(temp)
    print temp
    counter+=1



end = time.clock()
print "TOTAL TIME : ", end - start
for a in range(0,len(data)):
    cv2.line(img, tuple(corner_address[a%4]), tuple(corner_address[(a+1)%4]),(255,0,0),2)

cv2.imwrite("result1.jpg", img)
