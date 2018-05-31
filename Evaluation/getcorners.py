import cv2
import numpy as np
import tensorflow as tf


class get_corners:
    def __init__(self):
        BATCH_SIZE = 1
        NO_OF_STEPS = 50000
        CHECKPOINT_DIR = "../checkpoints_4_point_multi_multilayer_v2/"
        DATA_DIR = "../../DataSet Generator/data_set"
        GT_DIR = "../../DataSet Generator/Untitled Folder/gt1.csv"
        VALIDATION_PERCENTAGE = .20
        TEST_PERCENTAGE = .10
        Debug = True

        # img = cv2.imread("../temp/044.jpg")
        # img = cv2.resize(img, (800,800))


        self.sess = tf.Session()

        # In[ ]:

        def weight_variable(shape, name="temp"):
            initial = tf.truncated_normal(shape, stddev=0.1, name=name)
            return tf.Variable(initial)

        def bias_variable(shape, name="temp"):
            initial = tf.constant(0.1, shape=shape, name=name)
            return tf.Variable(initial)

        # In[ ]:

        def conv2d(x, W):
            return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

        def max_pool_2x2(x):
            return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                                  strides=[1, 2, 2, 1], padding='SAME')

        # In[ ]:

       

        W_conv1 = weight_variable([5, 5, 3, 20], name="W_conv1")
        b_conv1 = bias_variable([20], name="b_conv1")

        # In[ ]:

        self.x = tf.placeholder(tf.float32, shape=[None, 32, 32, 3])
        self.y_ = tf.placeholder(tf.float32, shape=[None, 8])

        h_conv1 = tf.nn.relu(conv2d(self.x, W_conv1) + b_conv1)
        h_pool1 = max_pool_2x2(h_conv1)

        W_conv2 = weight_variable([5, 5, 20, 40], name="W_conv2")
        b_conv2 = bias_variable([40], name="b_conv2")
        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
        h_pool2 = max_pool_2x2(h_conv2)

        W_conv3 = weight_variable([5, 5, 40, 60], name="W_conv3")
        b_conv3 = bias_variable([60], name="b_conv3")
        h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
        h_pool3 = max_pool_2x2(h_conv3)

        W_conv4 = weight_variable([5, 5, 60, 80], name="W_conv4")
        b_conv4 = bias_variable([80], name="b_conv4")
        h_conv4 = tf.nn.relu(conv2d(h_pool3, W_conv4) + b_conv4)
        h_pool4 = max_pool_2x2(h_conv4)

        W_conv5 = weight_variable([5, 5, 80, 100], name="W_conv5")
        b_conv5 = bias_variable([100], name="b_conv5")
        h_conv5 = tf.nn.relu(conv2d(h_pool4, W_conv5) + b_conv5)
        h_pool5 = max_pool_2x2(h_conv5)

        #print h_pool3.get_shape()

        temp_size = h_pool5.get_shape()
        temp_size = temp_size[1] * temp_size[2] * temp_size[3]
        temp_size = int(temp_size)
        # In[ ]:

        #print temp_size
        W_fc1 = weight_variable([int(temp_size), 500], name="W_fc1")
        b_fc1 = bias_variable([500], name="b_fc1")

        h_pool4_flat = tf.reshape(h_pool5, [-1, temp_size])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool4_flat, W_fc1) + b_fc1)

        # In[ ]:

        # Adding dropout
        self.keep_prob = tf.placeholder(tf.float32)
        h_fc1_drop = tf.nn.dropout(h_fc1, self.keep_prob)

        # In[ ]:

        W_fc2 = weight_variable([500, 500], name="W_fc2")
        b_fc2 = bias_variable([500], name="b_fc2")

        y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

        W_fc3 = weight_variable([500, 8], name="W_fc3")
        b_fc3 = bias_variable([8], name="b_fc3")

        self.y_conv = tf.matmul(y_conv, W_fc3) + b_fc3

        # In[ ]:


        cross_entropy = tf.nn.l2_loss(self.y_conv - self.y_)

        mySum = tf.summary.scalar('loss', cross_entropy)
        train_step = tf.train.AdamOptimizer(3e-3).minimize(cross_entropy)
        correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(self.y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        merged = tf.summary.merge_all()

        train_writer = tf.summary.FileWriter('../train', 
            self.sess.graph)

        # In[ ]:

        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(CHECKPOINT_DIR)
        if ckpt and ckpt.model_checkpoint_path:
            #print ("PRINTING CHECKPOINT PATH")
            #print(ckpt.model_checkpoint_path)
            init = saver.restore(self.sess, ckpt.model_checkpoint_path)

        else:
            #print("Starting from scratch")
            init = tf.global_variables_initializer()
            self.sess.run(init)
    def get(self,img):
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
        response = self.y_conv.eval(feed_dict={
            self.x: img_temp, self.keep_prob: 1.0}, session=self.sess)

        response = response[0]/32
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
        cv2.imshow("asd", tl)
        cv2.waitKey(0)
        cv2.imshow("asd", tr)
        cv2.waitKey(0)
        cv2.imshow("asd", br)
        cv2.waitKey(0)
        cv2.imshow("asd", bl)
        cv2.waitKey(0)
        end = timeit.timeit()
        #print end - start

        # In[ ]:


class get_corners_singlefc:
    def __init__(self):
        BATCH_SIZE = 1
        NO_OF_STEPS = 50000
        CHECKPOINT_DIR = "../checkpoints_4_point_multi_multilayer_v3/"
        DATA_DIR = "../../DataSet Generator/data_set"
        GT_DIR = "../../DataSet Generator/Untitled Folder/gt1.csv"
        VALIDATION_PERCENTAGE = .20
        TEST_PERCENTAGE = .10
        Debug = True

        # img = cv2.imread("../temp/044.jpg")
        # img = cv2.resize(img, (800,800))


        self.sess = tf.Session()
        sess = self.sess

        # In[ ]:

        def weight_variable(shape, name="temp"):
            initial = tf.truncated_normal(shape, stddev=0.1, name=name)
            return tf.Variable(initial)


        def bias_variable(shape, name="temp"):
            initial = tf.constant(0.1, shape=shape, name=name)
            return tf.Variable(initial)


        # In[ ]:

        def conv2d(x, W):
            return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


        def max_pool_2x2(x):
            return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                                  strides=[1, 2, 2, 1], padding='SAME')


        # In[ ]:

        W_conv1 = weight_variable([5, 5, 3, 20], name="W_conv1")
        b_conv1 = bias_variable([20], name="b_conv1")

        # In[ ]:

        self.x = x = tf.placeholder(tf.float32, shape=[None, 32, 32, 3])
        self.y_ = y_ = tf.placeholder(tf.float32, shape=[None, 8])

        h_conv1 = tf.nn.relu(conv2d(x, W_conv1) + b_conv1)
        h_pool1 = max_pool_2x2(h_conv1)

        W_conv2 = weight_variable([5, 5, 20, 40], name="W_conv2")
        b_conv2 = bias_variable([40], name="b_conv2")
        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
        h_pool2 = max_pool_2x2(h_conv2)

        W_conv3 = weight_variable([5, 5, 40, 60], name="W_conv3")
        b_conv3 = bias_variable([60], name="b_conv3")
        h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
        h_pool3 = max_pool_2x2(h_conv3)

        W_conv4 = weight_variable([5, 5, 60, 80], name="W_conv4")
        b_conv4 = bias_variable([80], name="b_conv4")
        h_conv4 = tf.nn.relu(conv2d(h_pool3, W_conv4) + b_conv4)
        h_pool4 = max_pool_2x2(h_conv4)

        W_conv5 = weight_variable([5, 5, 80, 100], name="W_conv5")
        b_conv5 = bias_variable([100], name="b_conv5")
        h_conv5 = tf.nn.relu(conv2d(h_pool4, W_conv5) + b_conv5)
        h_pool5 = max_pool_2x2(h_conv5)

        print h_pool3.get_shape()

        temp_size = h_pool5.get_shape()
        temp_size = temp_size[1] * temp_size[2] * temp_size[3]
        temp_size = int(temp_size)
        # In[ ]:

        print temp_size
        W_fc1 = weight_variable([int(temp_size), 500], name="W_fc1")
        b_fc1 = bias_variable([500], name="b_fc1")

        h_pool4_flat = tf.reshape(h_pool5, [-1, temp_size])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool4_flat, W_fc1) + b_fc1)


        # In[ ]:

        # Adding dropout
        self.keep_prob = keep_prob = tf.placeholder(tf.float32)
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

        # In[ ]:


        W_fc3 = weight_variable([500, 8], name="W_fc3")
        b_fc3 = bias_variable([8], name="b_fc3")

        self.y_conv = y_conv = tf.matmul(h_fc1_drop, W_fc3) + b_fc3



        # In[ ]:


        cross_entropy = tf.nn.l2_loss(y_conv - y_)

        mySum = tf.summary.scalar('loss', cross_entropy)
        train_step = tf.train.AdamOptimizer(6e-6).minimize(cross_entropy)
        correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        merged = tf.summary.merge_all()

        train_writer = tf.summary.FileWriter('../train', sess.graph)

        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(CHECKPOINT_DIR)
        if ckpt and ckpt.model_checkpoint_path:
            #print ("PRINTING CHECKPOINT PATH")
            #print(ckpt.model_checkpoint_path)
            init = saver.restore(self.sess, ckpt.model_checkpoint_path)

        else:
            #print("Starting from scratch")
            init = tf.global_variables_initializer()
            self.sess.run(init)
    def get(self,img):
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
        response = self.y_conv.eval(feed_dict={
            self.x: img_temp, self.keep_prob: 1.0}, session=self.sess)

        response = response[0]/32
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
        cv2.imshow("asd", tl)
        cv2.waitKey(0)
        cv2.imshow("asd", tr)
        cv2.waitKey(0)
        cv2.imshow("asd", br)
        cv2.waitKey(0)
        cv2.imshow("asd", bl)
        cv2.waitKey(0)
        end = timeit.timeit()
        #print end - start

        # In[ ]:


class get_corners_alex:
    def __init__(self):
        BATCH_SIZE = 1
        NO_OF_STEPS = 50000
        CHECKPOINT_DIR = "../checkpoints_4_point_multi_multilayer_v6/"
        DATA_DIR = "../../DataSet Generator/data_set"
        GT_DIR = "../../DataSet Generator/Untitled Folder/gt1.csv"
        VALIDATION_PERCENTAGE = .20
        TEST_PERCENTAGE = .10
        Debug = True

        # img = cv2.imread("../temp/044.jpg")
        # img = cv2.resize(img, (800,800))

        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.4
        self.sess = tf.Session(config=config)
        sess = self.sess
        train_image = np.load("train_image.npy")
        mean_train = np.mean(train_image, axis=(0,1,2))

        mean_train = np.expand_dims(mean_train, axis=0)
        mean_train = np.expand_dims(mean_train, axis=0)
        self.mean_train = np.expand_dims(mean_train, axis=0)
        # In[ ]:

        def weight_variable(shape, name="temp"):
            initial = tf.truncated_normal(shape, stddev=0.1, name=name)
            return tf.Variable(initial)


        def bias_variable(shape, name="temp"):
            initial = tf.constant(0.1, shape=shape, name=name)
            return tf.Variable(initial)


        # In[ ]:

        def conv2d(x, W):
            return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


        def max_pool_2x2(x):
            return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                                  strides=[1, 2, 2, 1], padding='SAME')


        # In[ ]:

        W_conv1 = weight_variable([5, 5, 3, 20], name="W_conv1")
        b_conv1 = bias_variable([20], name="b_conv1")

        # In[ ]:

        self.x = x = tf.placeholder(tf.float32, shape=[None, 32, 32, 3])
        self.y_ = y_ = tf.placeholder(tf.float32, shape=[None, 8])

        h_conv1 = tf.nn.relu(conv2d(x, W_conv1) + b_conv1)
        h_pool1 = max_pool_2x2(h_conv1)

        W_conv2 = weight_variable([5, 5, 20, 40], name="W_conv2")
        b_conv2 = bias_variable([40], name="b_conv2")
        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

        W_conv2_1 = weight_variable([5, 5, 40, 40], name="W_conv2_1")
        b_conv2_1= bias_variable([40], name="b_conv2_1")
        h_conv2_1 = tf.nn.relu(conv2d(h_conv2, W_conv2_1) + b_conv2_1)

        h_pool2 = max_pool_2x2(h_conv2_1)

        W_conv3 = weight_variable([5, 5, 40, 60], name="W_conv3")
        b_conv3 = bias_variable([60], name="b_conv3")
        h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)

        W_conv3_1 = weight_variable([5, 5, 60, 60], name="W_conv3_1")
        b_conv3_1 = bias_variable([60], name="b_conv3_1")
        h_conv3_1 = tf.nn.relu(conv2d(h_conv3, W_conv3_1) + b_conv3_1)

        h_pool3 = max_pool_2x2(h_conv3_1)

        W_conv4 = weight_variable([5, 5, 60, 80], name="W_conv4")
        b_conv4 = bias_variable([80], name="b_conv4")
        h_conv4 = tf.nn.relu(conv2d(h_pool3, W_conv4) + b_conv4)
        h_pool4 = max_pool_2x2(h_conv4)

        W_conv5 = weight_variable([5, 5, 80, 100], name="W_conv5")
        b_conv5 = bias_variable([100], name="b_conv5")
        h_conv5 = tf.nn.relu(conv2d(h_pool4, W_conv5) + b_conv5)
        h_pool5 = max_pool_2x2(h_conv5)



        print h_pool5.get_shape()

        temp_size = h_pool5.get_shape()
        temp_size = temp_size[1] * temp_size[2] * temp_size[3]
        temp_size = int(temp_size)
        # In[ ]:

        print temp_size
        W_fc1 = weight_variable([int(temp_size), 500], name="W_fc1")
        b_fc1 = bias_variable([500], name="b_fc1")

        h_pool4_flat = tf.reshape(h_pool5, [-1, temp_size])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool4_flat, W_fc1) + b_fc1)


        # In[ ]:

        # Adding dropout
        self.keep_prob = keep_prob = tf.placeholder(tf.float32)
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

        # In[ ]:


        W_fc2 = weight_variable([500, 500], name="W_fc2")
        b_fc2 = bias_variable([500], name="b_fc2")

        y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2


        W_fc3 = weight_variable([500, 8], name="W_fc3")
        b_fc3 = bias_variable([8], name="b_fc3")

        self.y_conv = y_conv = tf.matmul(y_conv, W_fc3) + b_fc3



        # In[ ]:


        self.cross_entropy = cross_entropy = tf.nn.l2_loss(y_conv - y_)

        self.mySum = mySum = tf.summary.scalar('loss', cross_entropy)
        self.train_step = train_step = tf.train.AdamOptimizer(4e-5).minimize(cross_entropy)
        correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        merged = tf.summary.merge_all()

        train_writer = tf.summary.FileWriter('../train', sess.graph)

        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(CHECKPOINT_DIR)
        if ckpt and ckpt.model_checkpoint_path:
            #print ("PRINTING CHECKPOINT PATH")
            #print(ckpt.model_checkpoint_path)
            init = saver.restore(self.sess, ckpt.model_checkpoint_path)

        else:
            #print("Starting from scratch")
            init = tf.global_variables_initializer()
            self.sess.run(init)
    def get(self,img):
        
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
        img_temp = img_temp - self.mean_train
        response = self.y_conv.eval(feed_dict={
            self.x: img_temp, self.keep_prob: 1.0}, session=self.sess)

        response = response[0]/32
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
        cv2.imshow("asd", tl)
        cv2.waitKey(0)
        cv2.imshow("asd", tr)
        cv2.waitKey(0)
        cv2.imshow("asd", br)
        cv2.waitKey(0)
        cv2.imshow("asd", bl)
        cv2.waitKey(0)
        end = timeit.timeit()
        #print end - start

        # In[ ]:



class get_corners_aug:
    def __init__(self):
        BATCH_SIZE = 1
        NO_OF_STEPS = 50000
        CHECKPOINT_DIR = "../c8/"
        DATA_DIR = "../../DataSet Generator/data_set"
        GT_DIR = "../../DataSet Generator/Untitled Folder/gt1.csv"
        VALIDATION_PERCENTAGE = .20
        TEST_PERCENTAGE = .10
        Debug = True

        # img = cv2.imread("../temp/044.jpg")
        # img = cv2.resize(img, (800,800))

        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.1
        self.sess = tf.Session(config=config)
        sess = self.sess
        train_image = np.load("train_image.npy")
        mean_train = np.mean(train_image, axis=(0,1,2))

        mean_train = np.expand_dims(mean_train, axis=0)
        mean_train = np.expand_dims(mean_train, axis=0)
        self.mean_train = np.expand_dims(mean_train, axis=0)
        # In[ ]:

        def weight_variable(shape, name="temp"):
            initial = tf.truncated_normal(shape, stddev=0.1, name=name)
            return tf.Variable(initial)


        def bias_variable(shape, name="temp"):
            initial = tf.constant(0.1, shape=shape, name=name)
            return tf.Variable(initial)


        # In[ ]:

        def conv2d(x, W):
            return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


        def max_pool_2x2(x):
            return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                                  strides=[1, 2, 2, 1], padding='SAME')


        # In[ ]:



        # In[ ]:


        with tf.name_scope("Input"):
            self.x = x = tf.placeholder(tf.float32, shape=[None, 32, 32, 3])
            



            x_ = tf.image.random_brightness(x, 5)
            x_ = tf.image.random_contrast(x_, lower=0.9, upper=1.1)
        with tf.name_scope("gt"):
            self.y_= y_ = tf.placeholder(tf.float32, shape=[None, 8])

        with tf.name_scope("Conv1"):
            W_conv1 = weight_variable([5, 5, 3, 20], name="W_conv1")
            b_conv1 = bias_variable([20], name="b_conv1")
            h_conv1 = tf.nn.relu(conv2d(x_, W_conv1) + b_conv1)
        with tf.name_scope("MaxPool1"):
            h_pool1 = max_pool_2x2(h_conv1)
        with tf.name_scope("Conv2"):
            W_conv2 = weight_variable([5, 5, 20, 40], name="W_conv2")
            b_conv2 = bias_variable([40], name="b_conv2")
            h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
        with tf.name_scope("Conv2_1"):
            W_conv2_1 = weight_variable([5, 5, 40, 40], name="W_conv2_1")
            b_conv2_1= bias_variable([40], name="b_conv2_1")
            h_conv2_1 = tf.nn.relu(conv2d(h_conv2, W_conv2_1) + b_conv2_1)
        with tf.name_scope("MaxPool2"):
            h_pool2 = max_pool_2x2(h_conv2_1)
        with tf.name_scope("Conv3"):
            W_conv3 = weight_variable([5, 5, 40, 60], name="W_conv3")
            b_conv3 = bias_variable([60], name="b_conv3")
            h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)

            W_conv3_1 = weight_variable([5, 5, 60, 60], name="W_conv3_1")
            b_conv3_1 = bias_variable([60], name="b_conv3_1")
            h_conv3_1 = tf.nn.relu(conv2d(h_conv3, W_conv3_1) + b_conv3_1)
        with tf.name_scope("MaxPool3"):
            h_pool3 = max_pool_2x2(h_conv3_1)
        with tf.name_scope("Conv4"):
            W_conv4 = weight_variable([5, 5, 60, 80], name="W_conv4")
            b_conv4 = bias_variable([80], name="b_conv4")
            h_conv4 = tf.nn.relu(conv2d(h_pool3, W_conv4) + b_conv4)
        with tf.name_scope("Maxpool4"):
            h_pool4 = max_pool_2x2(h_conv4)
        with tf.name_scope("Conv5"):
            W_conv5 = weight_variable([5, 5, 80, 100], name="W_conv5")
            b_conv5 = bias_variable([100], name="b_conv5")
            h_conv5 = tf.nn.relu(conv2d(h_pool4, W_conv5) + b_conv5)
            h_pool5 = max_pool_2x2(h_conv5)



        print h_pool5.get_shape()

        temp_size = h_pool5.get_shape()
        temp_size = temp_size[1] * temp_size[2] * temp_size[3]
        temp_size = int(temp_size)
        # In[ ]:

        print temp_size
        with tf.name_scope("FCLayers"):
            W_fc1 = weight_variable([int(temp_size), 500], name="W_fc1")
            b_fc1 = bias_variable([500], name="b_fc1")

            h_pool4_flat = tf.reshape(h_pool5, [-1, temp_size])
            h_fc1 = tf.nn.relu(tf.matmul(h_pool4_flat, W_fc1) + b_fc1)


            # In[ ]:

            # Adding dropout
            self.keep_prob = keep_prob = tf.placeholder(tf.float32)
            h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

            # In[ ]:


            W_fc2 = weight_variable([500, 500], name="W_fc2")
            b_fc2 = bias_variable([500], name="b_fc2")

            y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2


            W_fc3 = weight_variable([500, 8], name="W_fc3")
            b_fc3 = bias_variable([8], name="b_fc3")

            self.y_conv =y_conv = tf.matmul(y_conv, W_fc3) + b_fc3



        # In[ ]:

        with tf.name_scope("loss"):
            cross_entropy = tf.nn.l2_loss(y_conv - y_)

            mySum = tf.summary.scalar('Train_loss', cross_entropy)
            validate_loss = tf.summary.scalar('Validate_loss', cross_entropy)
        with tf.name_scope("Train"):
            train_step = tf.train.AdamOptimizer(1e-5).minimize(cross_entropy)


        merged = tf.summary.merge_all()

        train_writer = tf.summary.FileWriter('../train', sess.graph)

        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(CHECKPOINT_DIR)
        if ckpt and ckpt.model_checkpoint_path:
            #print ("PRINTING CHECKPOINT PATH")
            #print(ckpt.model_checkpoint_path)
            init = saver.restore(self.sess, ckpt.model_checkpoint_path)

        else:
            #print("Starting from scratch")
            init = tf.global_variables_initializer()
            self.sess.run(init)
    def get(self,img):
        
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
        img_temp = img_temp - self.mean_train
        response = self.y_conv.eval(feed_dict={
            self.x: img_temp, self.keep_prob: 1.0}, session=self.sess)

        response = response[0]/32
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
        cv2.imshow("asd", tl)
        cv2.waitKey(0)
        cv2.imshow("asd", tr)
        cv2.waitKey(0)
        cv2.imshow("asd", br)
        cv2.waitKey(0)
        cv2.imshow("asd", bl)
        cv2.waitKey(0)
        end = timeit.timeit()
        #print end - start

        # In[ ]:





class get_corners_moreBG:
    def __init__(self):
        BATCH_SIZE = 1
        NO_OF_STEPS = 50000
        CHECKPOINT_DIR = "../4PointAllBg"
        DATA_DIR = "../../DataSet Generator/data_set"
        GT_DIR = "../../DataSet Generator/Untitled Folder/gt1.csv"
        VALIDATION_PERCENTAGE = .20
        TEST_PERCENTAGE = .10
        Debug = True

        # img = cv2.imread("../temp/044.jpg")
        # img = cv2.resize(img, (800,800))

        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.1
        self.sess = tf.Session(config=config)
        sess = self.sess
        train_image = np.load("../train_image_all_bg.npy")
        mean_train = np.mean(train_image, axis=(0,1,2))

        mean_train = np.expand_dims(mean_train, axis=0)
        mean_train = np.expand_dims(mean_train, axis=0)
        self.mean_train = np.expand_dims(mean_train, axis=0)
        # In[ ]:

        def weight_variable(shape, name="temp"):
            initial = tf.truncated_normal(shape, stddev=0.1, name=name)
            return tf.Variable(initial)


        def bias_variable(shape, name="temp"):
            initial = tf.constant(0.1, shape=shape, name=name)
            return tf.Variable(initial)


        # In[ ]:

        def conv2d(x, W):
            return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


        def max_pool_2x2(x):
            return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                                  strides=[1, 2, 2, 1], padding='SAME')


        # In[ ]:



        # In[ ]:


        with tf.name_scope("Input"):
            self.x = x = tf.placeholder(tf.float32, shape=[None, 32, 32, 3])
            



            x_ = tf.image.random_brightness(x, 5)
            x_ = tf.image.random_contrast(x_, lower=0.9, upper=1.1)
        with tf.name_scope("gt"):
            self.y_= y_ = tf.placeholder(tf.float32, shape=[None, 8])

        with tf.name_scope("Conv1"):
            W_conv1 = weight_variable([5, 5, 3, 20], name="W_conv1")
            b_conv1 = bias_variable([20], name="b_conv1")
            h_conv1 = tf.nn.relu(conv2d(x_, W_conv1) + b_conv1)
        with tf.name_scope("MaxPool1"):
            h_pool1 = max_pool_2x2(h_conv1)
        with tf.name_scope("Conv2"):
            W_conv2 = weight_variable([5, 5, 20, 40], name="W_conv2")
            b_conv2 = bias_variable([40], name="b_conv2")
            h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
        with tf.name_scope("Conv2_1"):
            W_conv2_1 = weight_variable([5, 5, 40, 40], name="W_conv2_1")
            b_conv2_1= bias_variable([40], name="b_conv2_1")
            h_conv2_1 = tf.nn.relu(conv2d(h_conv2, W_conv2_1) + b_conv2_1)
        with tf.name_scope("MaxPool2"):
            h_pool2 = max_pool_2x2(h_conv2_1)
        with tf.name_scope("Conv3"):
            W_conv3 = weight_variable([5, 5, 40, 60], name="W_conv3")
            b_conv3 = bias_variable([60], name="b_conv3")
            h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)

            W_conv3_1 = weight_variable([5, 5, 60, 60], name="W_conv3_1")
            b_conv3_1 = bias_variable([60], name="b_conv3_1")
            h_conv3_1 = tf.nn.relu(conv2d(h_conv3, W_conv3_1) + b_conv3_1)
        with tf.name_scope("MaxPool3"):
            h_pool3 = max_pool_2x2(h_conv3_1)
        with tf.name_scope("Conv4"):
            W_conv4 = weight_variable([5, 5, 60, 80], name="W_conv4")
            b_conv4 = bias_variable([80], name="b_conv4")
            h_conv4 = tf.nn.relu(conv2d(h_pool3, W_conv4) + b_conv4)
        with tf.name_scope("Maxpool4"):
            h_pool4 = max_pool_2x2(h_conv4)
        with tf.name_scope("Conv5"):
            W_conv5 = weight_variable([5, 5, 80, 100], name="W_conv5")
            b_conv5 = bias_variable([100], name="b_conv5")
            h_conv5 = tf.nn.relu(conv2d(h_pool4, W_conv5) + b_conv5)
            h_pool5 = max_pool_2x2(h_conv5)



        print h_pool5.get_shape()

        temp_size = h_pool5.get_shape()
        temp_size = temp_size[1] * temp_size[2] * temp_size[3]
        temp_size = int(temp_size)
        # In[ ]:

        print temp_size
        with tf.name_scope("FCLayers"):
            W_fc1 = weight_variable([int(temp_size), 500], name="W_fc1")
            b_fc1 = bias_variable([500], name="b_fc1")

            h_pool4_flat = tf.reshape(h_pool5, [-1, temp_size])
            h_fc1 = tf.nn.relu(tf.matmul(h_pool4_flat, W_fc1) + b_fc1)


            # In[ ]:

            # Adding dropout
            self.keep_prob = keep_prob = tf.placeholder(tf.float32)
            h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

            # In[ ]:


            W_fc2 = weight_variable([500, 500], name="W_fc2")
            b_fc2 = bias_variable([500], name="b_fc2")

            y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2


            W_fc3 = weight_variable([500, 8], name="W_fc3")
            b_fc3 = bias_variable([8], name="b_fc3")

            self.y_conv =y_conv = tf.matmul(y_conv, W_fc3) + b_fc3



        # In[ ]:

        with tf.name_scope("loss"):
            cross_entropy = tf.nn.l2_loss(y_conv - y_)

            mySum = tf.summary.scalar('Train_loss', cross_entropy)
            validate_loss = tf.summary.scalar('Validate_loss', cross_entropy)
        with tf.name_scope("Train"):
            train_step = tf.train.AdamOptimizer(1e-5).minimize(cross_entropy)


        merged = tf.summary.merge_all()

        train_writer = tf.summary.FileWriter('../train', sess.graph)

        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(CHECKPOINT_DIR)
        if ckpt and ckpt.model_checkpoint_path:
            #print ("PRINTING CHECKPOINT PATH")
            #print(ckpt.model_checkpoint_path)
            init = saver.restore(self.sess, ckpt.model_checkpoint_path)

        else:
            #print("Starting from scratch")
            init = tf.global_variables_initializer()
            self.sess.run(init)
    def get(self,img):
        
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
        img_temp = img_temp - self.mean_train
        response = self.y_conv.eval(feed_dict={
            self.x: img_temp, self.keep_prob: 1.0}, session=self.sess)

        response = response[0]/32
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
        cv2.imshow("asd", tl)
        cv2.waitKey(0)
        cv2.imshow("asd", tr)
        cv2.waitKey(0)
        cv2.imshow("asd", br)
        cv2.waitKey(0)
        cv2.imshow("asd", bl)
        cv2.waitKey(0)
        end = timeit.timeit()
        #print end - start

