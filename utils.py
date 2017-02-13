import cv2
import numpy as np
import csv

def load_data(DATA_DIR, GT_DIR, size=(300,300), debug=False, limit=-1):
    gt_list = []
    file_names = []
    image_list = []

    with open(GT_DIR, 'r') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        import ast
        a = 0
        temp = 0
        for row in spamreader:
            temp += 1
            if (temp == limit):
                break
            file_names.append(row[0])
            gt_list.append((ast.literal_eval(row[1])[0], ast.literal_eval(row[1])[1]))
    if(debug):
        print ("GT Loaded : ",len(gt_list), " Files")
    for a in file_names:
        img = cv2.imread(DATA_DIR + "/" + a)

        img = cv2.resize(img, size)
        image_list.append(img)
    print len(image_list)

    gt_list = np.array(gt_list)
    image_list = np.array(image_list)
    gt_list = gt_list*size/(300,300)
    return image_list, gt_list

def validate_gt(gt_list, size):
    for a in gt_list:
        assert(a[0] <= 32 and a[0]>=0)
        assert(a[1] <= 32 and a[1]>=0)