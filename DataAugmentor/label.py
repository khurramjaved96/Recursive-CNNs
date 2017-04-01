import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import numpy as np
import os

current_file = None
def onclick(event):
    if event.dblclick:
        print('button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
              (event.button, event.x, event.y, event.xdata, event.ydata))
        import csv
        with open(current_file+".csv", 'a') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter=' ',
                                    quotechar='|', quoting=csv.QUOTE_MINIMAL)
            spamwriter.writerow([str(event.xdata), str(event.ydata)])




dir = "../data1/"
for image in os.listdir(dir):
    if image.endswith("jpg") or image.endswith("JPG"):
        if os.path.isfile(dir+image+".csv"):
            pass
        else:
            fig = plt.figure()
            cid = fig.canvas.mpl_connect('button_press_event', onclick)
            print dir+image
            current_file = dir+image
            img=mpimg.imread(dir+image)
            plt.imshow(img)
            plt.show()