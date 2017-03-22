import numpy as np
import cv2
import tensorflow as tf

class corner_finder():
    def __init__(self, CHECKPOINT_DIR="../corner_full_data_shallow"):



        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.4
        self.sess = tf.Session(config=config)


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
        with tf.variable_scope('Corner'):

            W_conv1 = weight_variable([5, 5, 3, 10], name="W_conv1")
            b_conv1 = bias_variable([10], name="b_conv1")

            # In[ ]:

            self.x = tf.placeholder(tf.float32, shape=[None, 32, 32, 3])
            self.y_ = tf.placeholder(tf.float32, shape=[None, 2])
            x = self.x
            y_ = self.y_

            h_conv1 = tf.nn.relu(conv2d(self.x, W_conv1) + b_conv1)
            h_pool1 = max_pool_2x2(h_conv1)

            W_conv2 = weight_variable([5, 5, 10, 20], name="W_conv2")
            b_conv2 = bias_variable([20], name="b_conv2")
            h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
            h_pool2 = max_pool_2x2(h_conv2)

            W_conv3 = weight_variable([5, 5, 20, 30], name="W_conv3")
            b_conv3 = bias_variable([30], name="b_conv3")
            h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
            h_pool3 = max_pool_2x2(h_conv3)

            W_conv4 = weight_variable([5, 5, 30, 40], name="W_conv4")
            b_conv4 = bias_variable([40], name="b_conv4")
            h_conv4 = tf.nn.relu(conv2d(h_pool3, W_conv4) + b_conv4)
            h_pool4 = max_pool_2x2(h_conv4)

            print h_pool4.get_shape()

            temp_size = h_pool4.get_shape()
            temp_size = temp_size[1] * temp_size[2] * temp_size[3]

            # In[ ]:

            W_fc1 = weight_variable([int(temp_size), 300], name="W_fc1")
            b_fc1 = bias_variable([300], name="b_fc1")

            h_pool4_flat = tf.reshape(h_pool4, [-1, int(temp_size)])
            h_fc1 = tf.nn.relu(tf.matmul(h_pool4_flat, W_fc1) + b_fc1)

            # In[ ]:

            # Adding dropout
            self.keep_prob = tf.placeholder(tf.float32)
            keep_prob = self.keep_prob
            h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

            # In[ ]:

            W_fc2 = weight_variable([300, 2], name="W_fc2")
            b_fc2 = bias_variable([2], name="b_fc2")

            self.y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

            # W_fc3 = weight_variable([500, 500], name="W_fc3")
            # b_fc3 = bias_variable([500], name="b_fc3")

            # y_conv = tf.matmul(y_conv, W_fc3) + b_fc3

            # W_fc4 = weight_variable([500, 2], name="W_fc4")
            # b_fc4 = bias_variable([2], name="b_fc4")

            # y_conv = tf.matmul(h_fc1_drop, W_fc4) + b_fc4



            # In[ ]:


            cross_entropy = tf.nn.l2_loss(self.y_conv - y_)

            mySum = tf.summary.scalar('loss', cross_entropy)
            train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
            correct_prediction = tf.equal(tf.argmax(self.y_conv, 1), tf.argmax(y_, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        merged = tf.summary.merge_all()
            
        all_vars = tf.GraphKeys.GLOBAL_VARIABLES
        saver = tf.train.Saver( tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Corner'))
        ckpt = tf.train.get_checkpoint_state(CHECKPOINT_DIR)
        if ckpt and ckpt.model_checkpoint_path:
             #print ("PRINTING CHECKPOINT PATH")
             #print(ckpt.model_checkpoint_path)
             init = saver.restore(self.sess, ckpt.model_checkpoint_path)
        
        else:
             #print("Starting from scratch")
             init = tf.global_variables_initializer()
             self.sess.run(init)

    def get_location(self,img):

        ans_x =0.0
        ans_y=0.0

        this_is_temp = img.shape
        RESIZE = 1200
        img = cv2.resize(img, (RESIZE, RESIZE))
        o_img = np.copy(img)

        import time

        y = None
        x_start = 0
        y_start = 0
        up_scale_factor = (img.shape[1], img.shape[0])
        crop_size = [img.shape[0] * .8, img.shape[1] * .8]

        myImage = np.copy(o_img)

        CROP_FRAC = .95
        start = time.clock()
        for counter in range(0, 60):

            img_temp = cv2.resize(myImage, (32, 32))
            img_temp = np.expand_dims(img_temp, axis=0)
            response = self.y_conv.eval(feed_dict={
                self.x: img_temp, self.keep_prob: 1.0}, session=self.sess)

            response_up = response[0] / 32

            response_up = response_up * up_scale_factor

            y = response_up + (x_start, y_start)

            img1 = np.copy(img)
            #cv2.circle(img1, (int(response_up[0] + x_start), int(response_up[1] + y_start)), 2, (255, 0, 0), 2)
            #cv2.imwrite("../sample_" + str(counter) + ".jpg", img1)
            # cv2.waitKey(0)
            #cv2.circle(img_temp[0], (int(response[0][0]), int(response[0][1])), 2, (255, 0, 0), 2)
            #cv2.imwrite("../down_result" + str(counter) + ".jpg", img_temp[0])
            # cv2.waitKey(0)

            x_loc = int(y[0])
            y_loc = int(y[1])

            if x_loc > myImage.shape[0] / 2:
                start_x = min(x_loc + int(round(myImage.shape[0] * CROP_FRAC / 2)), myImage.shape[0]) - int(round(
                    myImage.shape[0] * CROP_FRAC))
            else:
                start_x = max(x_loc - int(myImage.shape[0] * CROP_FRAC / 2), 0)
            if y_loc > myImage.shape[1] / 2:
                start_y = min(y_loc + int(myImage.shape[1] * CROP_FRAC / 2), myImage.shape[1]) - int(
                    myImage.shape[1] * CROP_FRAC)
            else:
                start_y = max(y_loc - int(myImage.shape[1] * CROP_FRAC / 2), 0)

            ans_x+= start_x
            ans_y+= start_y

            myImage = myImage[start_y:start_y + int(myImage.shape[0] * CROP_FRAC),
                      start_x:start_x + int(myImage.shape[1] * CROP_FRAC)]
            img = img[start_y:start_y + int(img.shape[0] * CROP_FRAC), start_x:start_x + int(img.shape[1] * CROP_FRAC)]
            up_scale_factor = (img.shape[1], img.shape[0])


        end = time.clock()

        ans_x += y[0]
        ans_y += y[1]
        ans_x/=RESIZE
        ans_x*= this_is_temp[1]
        ans_y /=RESIZE
        ans_y*= this_is_temp[0]
        # print "TIME : ", end - start
        return (int(round(ans_x)), int(round(ans_y)))

    def get_location1(self,img):
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

        CROP_FRAC = .95
        start = time.clock()
        end_x=img.shape[1]
        end_y = img.shape[0]
        start_x=0
        start_y = 0
        no_of_calls = 40
        while(myImage.shape[0]>5 and myImage.shape[1]>5):

            img_temp = cv2.resize(myImage, (32, 32))
            img_temp = np.expand_dims(img_temp, axis=0)
            response = self.y_conv.eval(feed_dict={
                self.x: img_temp, self.keep_prob: 1.0},session=self.sess)

            response_up = response[0] / 32

            response_up = response_up * up_scale_factor

            y = response_up + (x_start, y_start)

            img1 = np.copy(img)
            # cv2.circle(img1, (int(response_up[0] + x_start), int(response_up[1] + y_start)), 2, (0, int(255-(255.0/no_of_calls)*counter), int(255.0/no_of_calls)*counter), 4)
            # cv2.imwrite("../sample_" + str(counter) + ".jpg", img1)
            # cv2.waitKey(0)
            #cv2.circle(img_temp[0], (int(response[0][0]), int(response[0][1])), 2, (255, 0, 0), 2)
            #cv2.imwrite("../down_result" + str(counter) + ".jpg", img_temp[0])
            # cv2.waitKey(0)z
            # cv2.line(draw, (int(ans_x), int(ans_y)), (int(ans_x),int(end_y)), (0, int(255-(155.0/no_of_calls)*counter), int(155.0/no_of_calls)*counter), 3)
            # cv2.line(draw, (int(ans_x), int(ans_y)), (int(end_x), int(ans_y)), (0, int(255-(155.0/no_of_calls)*counter), int(155.0/no_of_calls)*counter), 3)
            # cv2.line(draw, (int(end_x), int(end_y)), (int(end_x), int(ans_y)), (0, int(255-(155.0/no_of_calls)*counter), int(155.0/no_of_calls)*counter), 3)
            # cv2.line(draw, (int(end_x), int(end_y)), (int(ans_x), int(end_y)), (0, int(255-(155.0/no_of_calls)*counter), int(155.0/no_of_calls)*counter), 3)
            # # cv2.line(img, (b[1]+a.shape[1], b[2]), (b[1]+a.shape[1], b[2]+a.shape[0]), (0, 255 - 60*temp_counter, 60*temp_counter), 18)
            # # cv2.line(img, (b[1]+a.shape[1], b[2]+a.shape[0]), (b[1], b[2]+a.shape[0]), (0, 255 - 60*temp_counter, 60*temp_counter), 18)
            # # cv2.line(img, (b[1], b[2]+a.shape[0]), (b[1], b[2]), (0, 255 - 60*temp_counter, 60*temp_counter), 18)
            # cv2.imwrite("../bas"+str(counter)+".jpg",draw)


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
        # ans_x/=RESIZE
        # ans_x*= this_is_temp[1]
        # ans_y /=RESIZE
        # ans_y*= this_is_temp[0]
        # print "TIME : ", end - start
        return (int(round(ans_x)), int(round(ans_y)))

# In[ ]:

class corner_finder_aug():
    def __init__(self, CHECKPOINT_DIR="../corner_withoutbg1"):



        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.1
        self.sess = tf.Session(config=config)

        train_image = np.load("train_image_bg1.npy")
       
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
        with tf.variable_scope('Corner'):

            W_conv1 = weight_variable([5, 5, 3, 10], name="W_conv1")
            b_conv1 = bias_variable([10], name="b_conv1")

            # In[ ]:

            self.x = tf.placeholder(tf.float32, shape=[None, 32, 32, 3])
            self.y_ = tf.placeholder(tf.float32, shape=[None, 2])
            x = self.x
            y_ = self.y_

            h_conv1 = tf.nn.relu(conv2d(self.x, W_conv1) + b_conv1)
            h_pool1 = max_pool_2x2(h_conv1)

            W_conv2 = weight_variable([5, 5, 10, 20], name="W_conv2")
            b_conv2 = bias_variable([20], name="b_conv2")
            h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
            h_pool2 = max_pool_2x2(h_conv2)

            W_conv3 = weight_variable([5, 5, 20, 30], name="W_conv3")
            b_conv3 = bias_variable([30], name="b_conv3")
            h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
            h_pool3 = max_pool_2x2(h_conv3)

            W_conv4 = weight_variable([5, 5, 30, 40], name="W_conv4")
            b_conv4 = bias_variable([40], name="b_conv4")
            h_conv4 = tf.nn.relu(conv2d(h_pool3, W_conv4) + b_conv4)
            h_pool4 = max_pool_2x2(h_conv4)

            print h_pool4.get_shape()

            temp_size = h_pool4.get_shape()
            temp_size = temp_size[1] * temp_size[2] * temp_size[3]

            # In[ ]:

            W_fc1 = weight_variable([int(temp_size), 300], name="W_fc1")
            b_fc1 = bias_variable([300], name="b_fc1")

            h_pool4_flat = tf.reshape(h_pool4, [-1, int(temp_size)])
            h_fc1 = tf.nn.relu(tf.matmul(h_pool4_flat, W_fc1) + b_fc1)

            # In[ ]:

            # Adding dropout
            self.keep_prob = tf.placeholder(tf.float32)
            keep_prob = self.keep_prob
            h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

            # In[ ]:

            W_fc2 = weight_variable([300, 2], name="W_fc2")
            b_fc2 = bias_variable([2], name="b_fc2")

            self.y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

            # W_fc3 = weight_variable([500, 500], name="W_fc3")
            # b_fc3 = bias_variable([500], name="b_fc3")

            # y_conv = tf.matmul(y_conv, W_fc3) + b_fc3

            # W_fc4 = weight_variable([500, 2], name="W_fc4")
            # b_fc4 = bias_variable([2], name="b_fc4")

            # y_conv = tf.matmul(h_fc1_drop, W_fc4) + b_fc4



            # In[ ]:


            cross_entropy = tf.nn.l2_loss(self.y_conv - y_)

            mySum = tf.summary.scalar('loss', cross_entropy)
            train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
            correct_prediction = tf.equal(tf.argmax(self.y_conv, 1), tf.argmax(y_, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        merged = tf.summary.merge_all()
            
        all_vars = tf.GraphKeys.GLOBAL_VARIABLES
        saver = tf.train.Saver( tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Corner'))
        ckpt = tf.train.get_checkpoint_state(CHECKPOINT_DIR)
        if ckpt and ckpt.model_checkpoint_path:
             #print ("PRINTING CHECKPOINT PATH")
             #print(ckpt.model_checkpoint_path)
             init = saver.restore(self.sess, ckpt.model_checkpoint_path)
        
        else:
             #print("Starting from scratch")
             init = tf.global_variables_initializer()
             self.sess.run(init)



    def get_location1(self,img):
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
            img_temp = img_temp -self.mean_train
            response = self.y_conv.eval(feed_dict={
                self.x: img_temp, self.keep_prob: 1.0},session=self.sess)

            response_up = response[0] / 32

            response_up = response_up * up_scale_factor

            y = response_up + (x_start, y_start)

            img1 = np.copy(img)
            # cv2.circle(img1, (int(response_up[0] + x_start), int(response_up[1] + y_start)), 2, (0, int(255-(255.0/no_of_calls)*counter), int(255.0/no_of_calls)*counter), 4)
            # cv2.imwrite("../sample_" + str(counter) + ".jpg", img1)
            # cv2.waitKey(0)
            #cv2.circle(img_temp[0], (int(response[0][0]), int(response[0][1])), 2, (255, 0, 0), 2)
            #cv2.imwrite("../down_result" + str(counter) + ".jpg", img_temp[0])
            # cv2.waitKey(0)z
            # cv2.line(draw, (int(ans_x), int(ans_y)), (int(ans_x),int(end_y)), (0, int(255-(155.0/no_of_calls)*counter), int(155.0/no_of_calls)*counter), 3)
            # cv2.line(draw, (int(ans_x), int(ans_y)), (int(end_x), int(ans_y)), (0, int(255-(155.0/no_of_calls)*counter), int(155.0/no_of_calls)*counter), 3)
            # cv2.line(draw, (int(end_x), int(end_y)), (int(end_x), int(ans_y)), (0, int(255-(155.0/no_of_calls)*counter), int(155.0/no_of_calls)*counter), 3)
            # cv2.line(draw, (int(end_x), int(end_y)), (int(ans_x), int(end_y)), (0, int(255-(155.0/no_of_calls)*counter), int(155.0/no_of_calls)*counter), 3)
            # # cv2.line(img, (b[1]+a.shape[1], b[2]), (b[1]+a.shape[1], b[2]+a.shape[0]), (0, 255 - 60*temp_counter, 60*temp_counter), 18)
            # # cv2.line(img, (b[1]+a.shape[1], b[2]+a.shape[0]), (b[1], b[2]+a.shape[0]), (0, 255 - 60*temp_counter, 60*temp_counter), 18)
            # # cv2.line(img, (b[1], b[2]+a.shape[0]), (b[1], b[2]), (0, 255 - 60*temp_counter, 60*temp_counter), 18)
            # cv2.imwrite("../bas"+str(counter)+".jpg",draw)


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
        # ans_x/=RESIZE
        # ans_x*= this_is_temp[1]
        # ans_y /=RESIZE
        # ans_y*= this_is_temp[0]
        # print "TIME : ", end - start
        return (int(round(ans_x)), int(round(ans_y)))

# In[ ]:

if __name__ == "__main__":
    pass

