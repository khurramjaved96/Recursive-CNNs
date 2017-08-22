import tensorflow as tf
import cv2
import numpy as np
def load_graph(frozen_graph_filename, inputName, outputName):
    # We load the protobuf file from the disk and parse it to retrieve the
    # unserialized graph_def
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # Then, we can use again a convenient built-in function to import a graph_def into the
    # current default Graph
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(
            graph_def,
            input_map=None,
            return_elements=None,
            name="prefix",
            op_dict=None,
            producer_op_list=None
        )
    x = graph.get_tensor_by_name('prefix/'+inputName+':0')
    y = graph.get_tensor_by_name('prefix/'+outputName+':0')
    return graph, x, y

def refineCorner(img, sess, x,y_eval):
    ans_x =0.0
    ans_y=0.0


    this_is_temp = img.shape
    RESIZE = 1200

    # img = cv2.resize(img, (RESIZE, RESIZE))
    o_img = np.copy(img)
    draw = np.copy(img)


    import time

    y = None
    x_start = 0
    y_start = 0
    up_scale_factor = (img.shape[1], img.shape[0])
    crop_size = [img.shape[0] * .8, img.shape[1] * .8]

    myImage = np.copy(o_img)

    CROP_FRAC = .85
    start = time.clock()
    end_x=img.shape[1]
    end_y = img.shape[0]
    start_x=0
    start_y = 0
    no_of_calls = 40
    while(myImage.shape[0]>10 and myImage.shape[1]>10):

        img_temp = cv2.resize(myImage, (32, 32))
        img_temp = np.expand_dims(img_temp, axis=0)
        response = y_eval.eval(feed_dict={
            x: img_temp},session=sess)

        response_up = response[0]

        response_up = response_up * up_scale_factor

        y = response_up + (x_start, y_start)

        img1 = np.copy(img)



        x_loc = int(y[0])
        y_loc = int(y[1])

        if x_loc > myImage.shape[1] / 2:
            start_x = min(x_loc + int(round(myImage.shape[1] * CROP_FRAC / 2)), myImage.shape[1]) - int(round(
                myImage.shape[1] * CROP_FRAC))
        else:
            start_x = max(x_loc - int(myImage.shape[1] * CROP_FRAC / 2), 0)
        if y_loc > myImage.shape[0] / 2:
            start_y = min(y_loc + int(myImage.shape[0] * CROP_FRAC / 2), myImage.shape[0]) - int(
                myImage.shape[0] * CROP_FRAC)
        else:
            start_y = max(y_loc - int(myImage.shape[0] * CROP_FRAC / 2), 0)

        ans_x+= start_x
        ans_y+= start_y

        myImage = myImage[start_y:start_y + int(myImage.shape[0] * CROP_FRAC),
                  start_x:start_x + int(myImage.shape[1] * CROP_FRAC)]
        img = img[start_y:start_y + int(img.shape[0] * CROP_FRAC), start_x:start_x + int(img.shape[1] * CROP_FRAC)]
        up_scale_factor = (img.shape[1], img.shape[0])
        end_y = ans_y + start_y + int(myImage.shape[0] * CROP_FRAC)
        end_x = ans_x + start_x + int(myImage.shape[1] * CROP_FRAC)


    end = time.clock()

    ans_x += y[0]
    ans_y += y[1]

    return (int(round(ans_x)), int(round(ans_y)))

def getCorners(img, sess, x, output):
        
        o_img = np.copy(img)
        import timeit

        y = None
        x_start = 0
        y_start = 0
        up_scale_factor = (img.shape[0], img.shape[1])
        crop_size = [img.shape[0] * .8, img.shape[1] * .8]
        start = timeit.timeit()
        myImage = np.copy(o_img)

        CROP_FRAC = .95
       #print myImage.shape

        img_temp = cv2.resize(myImage, (32, 32))

        img_temp = np.expand_dims(img_temp, axis=0)

        response = output.eval(feed_dict={
            x: img_temp}, session=sess)

        response = response[0]
        #print response
        x = response[[0,2,4,6]]
        y = response[[1,3,5,7]]
        x = x*myImage.shape[1]
        y = y*myImage.shape[0]
        # for a in range(0,4):
        #     cv2.circle(myImage, (x[a], y[a]), 2,(255,0,0),2)
        tl = myImage[max(0,int(2*y[0] -(y[3]+y[0])/2)):int((y[3]+y[0])/2),max(0,int(2*x[0] -(x[1]+x[0])/2)):int((x[1]+x[0])/2)]

        tr = myImage[max(0,int(2*y[1] -(y[1]+y[2])/2)):int((y[1]+y[2])/2),int((x[1]+x[0])/2):min(myImage.shape[1]-1, int(x[1]+(x[1]-x[0])/2))]

        br = myImage[int((y[1]+y[2])/2):min(myImage.shape[0]-1,int(y[2]+(y[2]-y[1])/2)),int((x[2]+x[3])/2):min(myImage.shape[1]-1, int(x[2]+(x[2]-x[3])/2))]

        bl = myImage[int((y[0]+y[3])/2):min(myImage.shape[0]-1,int(y[3]+(y[3]-y[0])/2)),max(0,int(2*x[3] -(x[2]+x[3])/2)):int((x[3]+x[2])/2)]

        tl =  (tl,max(0,int(2*x[0] -(x[1]+x[0])/2)),max(0,int(2*y[0] -(y[3]+y[0])/2)))
        tr = (tr, int((x[1]+x[0])/2), max(0,int(2*y[1] -(y[1]+y[2])/2)))
        br = (br,int((x[2]+x[3])/2) ,int((y[1]+y[2])/2))
        bl = (bl, max(0,int(2*x[3] -(x[2]+x[3])/2)),int((y[0]+y[3])/2))

        return tl, tr, br, bl

if __name__ == "__main__":
    graph,x ,y = load_graph("cornerRefiner.pb","Corner/inputTensor", "Corner/outputTensor")
    graphCorners, xCorners, yCorners = load_graph("getCorners.pb","Input/inputTensor", "FCLayers/outputTensor")
    
    
    img = cv2.imread("img.jpg")
    sess = tf.Session(graph=graph)
    sessCorners = tf.Session(graph=graphCorners)
  


    result =np.copy(img)

    
    data  =getCorners(img,sessCorners, xCorners,yCorners)

    corner_address=[]


    counter = 0
    for b in data:
        a = b[0]
        temp = np.array(refineCorner(a, sess, x,y))
        temp[0]+= b[1]
        temp[1]+= b[2]
        corner_address.append(temp)
        print temp
        counter+=1



    for a in range(0,len(data)):
        cv2.line(img, tuple(corner_address[a%4]), tuple(corner_address[(a+1)%4]),(255,0,0),10)

    cv2.imwrite("result1.jpg", img)

    
