import cv2
import numpy as np
import csv

def load_data(DATA_DIR, GT_DIR, size=(300,300), debug=False, limit=-1, remove_background=0):
    gt_list = []
    file_names = []
    image_list = []
    with open(GT_DIR, 'r') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        import ast
        a = 0
        temp = 0
        for row in spamreader:
            if row[0][0:12] == "background0"+str(remove_background):
                print row[0]
                continue 
            temp += 1
            if (temp == limit):
                break
            file_names.append(row[0])
            gt_list.append((ast.literal_eval(row[1])[0], ast.literal_eval(row[1])[1]))
    if(debug):
        print ("GT Loaded : ",len(gt_list), " Files")
    gt_list = np.array(gt_list)
    print gt_list.shape
    validate_gt(gt_list, (300,300), file_names)
    counter = 0
    for a in file_names:
        img = cv2.imread(DATA_DIR + "/" + a)

        scale = (float(size[1])/float(img.shape[1]),float(size[0])/img.shape[0])
        gt_list[counter]= (gt_list[counter].astype(float)*scale).astype(int)
        img = cv2.resize(img, size)
        image_list.append(img)
        counter+=1

    print len(image_list)

    
    image_list = np.array(image_list)
    print image_list.shape
    print gt_list.shape 
    print gt_list[0]
    return image_list, gt_list, file_names

def load_data_4(DATA_DIR, GT_DIR, size=(300,300), debug=False, limit=-1, remove_background = 0):
    gt_list = []
    file_names = []
    image_list = []

    with open(GT_DIR, 'r') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        import ast
        a = 0
        temp = 0
        for row in spamreader:
            # print row
            if row[0][0:12] == "background0"+str(remove_background):
                print row[0]
                continue 
            temp += 1
            if (temp == limit):
                break
            
            file_names.append(row[0])
            test = row[1].replace("array","")
            
            gt_list.append((ast.literal_eval(test)))
            
    gt_list= np.array(gt_list)
    print "GT SHAPE : ", gt_list.shape
    if(debug):
        print ("GT Loaded : ",len(gt_list), " Files")
    counter = 0
    for a in file_names:
        img = cv2.imread(DATA_DIR + "/" + a)
        
        scale = (float(size[1])/float(img.shape[1]),float(size[0])/img.shape[0])
        
        gt_list[counter]= (gt_list[counter].astype(float)*scale).astype(int)
        X = gt_list[counter]
        
        img = cv2.resize(img, size)
        
        # cv2.circle(img,tuple(X[0]),2, (255,0,0),2)
        # cv2.circle(img, tuple(X[1]), 2, (0, 255, 0), 2)
        # cv2.circle(img, tuple(X[2]), 2, (0, 0, 255), 2)
        # cv2.circle(img, tuple(X[3]), 2, (255, 255, 0), 2)
        # cv2.imwrite("../im"+str(counter)+".jpg", img)
        
        image_list.append(img)
        counter+=1
    print len(image_list)

    print gt_list[0]
    gt_list = np.reshape(gt_list, (-1,8))
    print gt_list[0]

    image_list = np.array(image_list)
    
    return image_list, gt_list, file_names

def load_data_4_1(DATA_DIR, GT_DIR, size=(300,300), debug=False, limit=-1, remove_background = 0):
    gt_list = []
    file_names = []
    image_list = []

    with open(GT_DIR, 'r') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        import ast
        a = 0
        temp = 0
        for row in spamreader:
            print row
            if row[0][0:12] == "background0"+str(remove_background):
                print row[0]
                continue 
            temp += 1
            if (temp == limit):
                break
            
            file_names.append(row[0])
            test = row[1].replace("array","")
            
            gt_list.append((test))
            
    gt_list= np.array(gt_list)
    print "GT SHAPE : ", gt_list.shape
    if(debug):
        print ("GT Loaded : ",len(gt_list), " Files")
    counter = 0
    for a in file_names:
        img = cv2.imread(DATA_DIR + "/" + a)
        
        scale = (float(size[1])/float(img.shape[1]),float(size[0])/img.shape[0])
        
        gt_list[counter]= (gt_list[counter].astype(float)*scale).astype(int)
        X = gt_list[counter]
        
        img = cv2.resize(img, size)
        
        # cv2.circle(img,tuple(X[0]),2, (255,0,0),2)
        # cv2.circle(img, tuple(X[1]), 2, (0, 255, 0), 2)
        # cv2.circle(img, tuple(X[2]), 2, (0, 0, 255), 2)
        # cv2.circle(img, tuple(X[3]), 2, (255, 255, 0), 2)
        # cv2.imwrite("../im"+str(counter)+".jpg", img)
        
        image_list.append(img)
        counter+=1
    print len(image_list)

    print gt_list[0]
    gt_list = np.reshape(gt_list, (-1,8))
    print gt_list[0]

    image_list = np.array(image_list)
    
    return image_list, gt_list, file_names
def validate_gt(gt_list, size, name_list=None):
    no=0
    counter=0
    for a in gt_list:

        if(not ((a[0] <= size[0] and a[0]>=0) and (a[1] <= size[1] and a[1]>=0))):
            print a
            if(name_list==None):
                pass
            else:
                print name_list[counter]
            no+=1
        counter+=1
    print no
        # assert(a[0] <= size[0] and a[0]>=0)
        # assert(a[1] <= size[1] and a[1]>=0)


def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

def intersection(a,b, img):
    img1 = np.zeros_like(img)

    cv2.fillConvexPoly(img1,a,(255,0,0))
    img1 =np.sum(img1,axis=2)

    img1=img1/255

    img2 = np.zeros_like(img)
    cv2.fillConvexPoly(img2, b, (255, 0, 0))
    img2 =np.sum(img2, axis=2)
    img2 = img2 / 255

    inte = img1*img2
    union = np.logical_or(img1, img2)
    iou = np.sum(inte)/np.sum(union)
    print iou
    return iou
    print np.sum(inte)
    print np.sum(union)

def intersection_with_corection(a,b,img):
    img1 = np.zeros_like(img)
    cv2.fillConvexPoly(img1,a,(255,0,0))
    # cv2.fillConvexPoly(img1,a,(255,0,0))

    img2 = np.zeros_like(img)
    cv2.fillConvexPoly(img2,b,(255,0,0))
    min_x = min(a[0][0],a[1][0],a[2][0],a[3][0])
    min_y = min(a[0][1],a[1][1],a[2][0],a[3][0])
    max_x = max(a[0][0],a[1][0],a[2][0],a[3][0])
    max_y = max(a[0][1],a[1][1],a[2][0],a[3][0])

    dst = np.array(((min_x, min_y), (max_x, min_y), (max_x, max_y), (min_x, max_y)))
    mat = cv2.getPerspectiveTransform(a.astype(np.float32), dst.astype(np.float32))
    img1 = cv2.warpPerspective(img1, mat,tuple((img.shape[0],img.shape[1])))
    img2 = cv2.warpPerspective(img2, mat,tuple((img.shape[0],img.shape[1])))
    #cv2.imwrite("../temp.jpg", img1);
    #cv2.imwrite("../temp1.jpg", img2);

    img1 =np.sum(img1,axis=2)
    img1=img1/255
    img2 =np.sum(img2, axis=2)
    img2 = img2 / 255

    inte = img1*img2
    union = np.logical_or(img1, img2)
    iou = np.sum(inte)/np.sum(union)
    print iou
    return iou
    

if __name__ == "__main__":
    img = cv2.imread("/home/khurramjaved/Dicta/Data/background01/magazine003.avi/125.jpg")


    a = np.array(((653,234), (1027,312), (885,816), (430, 712))) 
    b = np.array(((646,237),(1023, 312), (887,814), (431,710)))
    intersection(a,b, img)
    intersection_with_corection(a,b,img)


