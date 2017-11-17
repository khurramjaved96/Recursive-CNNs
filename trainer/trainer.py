import numpy as np
import cv2
import tensorflow as tf
import os
import math


class detector:
    def __init__(self, trainDir, validateDir, checkpointDir, trainMode=True, noOfSteps=10000, batchSize=10, size=(32,32), verbose=True, concatinateFeatures=False):
        self.batchSize = batchSize
        self.noOfSteps = noOfSteps
        self.concatinateFeatures=concatinateFeatures
        self.checkpointDir = checkpointDir
        self.verbose = verbose
        self.trainDir = trainDir
        self.validateDir = validateDir
        self.trainMode=trainMode
        self.size = size
        if not os.path.exists(checkpointDir):
            os.makedirs(checkpointDir)
    def loadData(self):
        self.train_image = np.load(self.trainDir + "Images.npy")
        self.validate_image = np.load(self.validateDir + "Images.npy")
        self.train_gt =  np.load(self.trainDir+"Gt.npy")
        self.validate_gt = np.load(self.validateDir+"Gt.npy")

class documentDetector(detector):
    def __init__(self, trainDir, validateDir, checkpointDir,trainMode=True,  size = (32,32), noOfSteps=1000, batchSize=3, verbose=2,concatinateFeatures=True):
        detector.__init__(self, trainDir, validateDir, checkpointDir, trainMode, noOfSteps, batchSize,size,  verbose,concatinateFeatures)


    def setupModel(self):
        checkpointDir = self.checkpointDir
        if (not os.path.isdir(checkpointDir)):
            os.mkdir(checkpointDir)
        config = tf.ConfigProto()

        if self.verbose:
            print self.train_image.shape
        mean_train = np.mean(self.train_image, axis=(0, 1, 2))

        mean_train = np.expand_dims(mean_train, axis=0)
        mean_train = np.expand_dims(mean_train, axis=0)
        mean_train = np.expand_dims(mean_train, axis=0)

        train_image = self.train_image - mean_train
        self.validate_image = self.validate_image - mean_train

        self.sess = sess = tf.InteractiveSession(config=config)

        def weight_variable(shape, name="temp"):
            initial = tf.truncated_normal(shape, stddev=0.1, name=name)
            return tf.Variable(initial)

        def bias_variable(shape, name="temp"):
            initial = tf.constant(0.1, shape=shape, name=name)
            return tf.Variable(initial)


        def conv2d(x, W,t):
            return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

        def conv2dBatchNorm(x,W, phase):
            h1 = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
            h2 = tf.contrib.layers.batch_norm(h1,
                                              center=True, scale=True,
                                              is_training=phase)
            return h2

        def max_pool_2x2(x):
            return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                                  strides=[1, 2, 2, 1], padding='SAME')

        with tf.name_scope("Input"):
            inputImage = tf.placeholder(tf.float32, shape=[None, None, None, 3])
            self.x = x = tf.image.resize_images(inputImage, [self.size[0], self.size[1]])

            x_ = tf.image.random_brightness(x, 5)
            x_ = tf.image.random_contrast(x_, lower=0.2, upper=1.8)
        with tf.name_scope("gt"):
            self.y_ = y_ = tf.placeholder(tf.float32, shape=[None, 8])

        with tf.name_scope("Conv1"):
            W_conv1 = weight_variable([5, 5, 3, 20], name="W_conv1")
            b_conv1 = bias_variable([20], name="b_conv1")
            h_conv1 = tf.nn.relu(conv2dBatchNorm(x_, W_conv1, self.trainMode) + b_conv1)
        with tf.name_scope("MaxPool1"):
            h_pool1 = max_pool_2x2(h_conv1)
        with tf.name_scope("Conv2"):
            W_conv2 = weight_variable([5, 5, 20, 40], name="W_conv2")
            b_conv2 = bias_variable([40], name="b_conv2")
            h_conv2 = tf.nn.relu(conv2dBatchNorm(h_pool1, W_conv2, self.trainMode) + b_conv2)
        with tf.name_scope("Conv2_1"):
            W_conv2_1 = weight_variable([5, 5, 40, 40], name="W_conv2_1")
            b_conv2_1 = bias_variable([40], name="b_conv2_1")
            h_conv2_1 = tf.nn.relu(conv2dBatchNorm(h_conv2, W_conv2_1,self.trainMode) + b_conv2_1)
        with tf.name_scope("MaxPool2"):
            h_pool2 = max_pool_2x2(h_conv2_1)
        with tf.name_scope("Conv3"):
            W_conv3 = weight_variable([5, 5, 40, 60], name="W_conv3")
            b_conv3 = bias_variable([60], name="b_conv3")
            h_conv3 = tf.nn.relu(conv2dBatchNorm(h_pool2, W_conv3, self.trainMode) + b_conv3)

            W_conv3_1 = weight_variable([5, 5, 60, 60], name="W_conv3_1")
            b_conv3_1 = bias_variable([60], name="b_conv3_1")
            h_conv3_1 = tf.nn.relu(conv2dBatchNorm(h_conv3, W_conv3_1, self.trainMode) + b_conv3_1)
        with tf.name_scope("MaxPool3"):
            h_pool3 = max_pool_2x2(h_conv3_1)
        with tf.name_scope("Conv4"):
            W_conv4 = weight_variable([5, 5, 60, 80], name="W_conv4")
            b_conv4 = bias_variable([80], name="b_conv4")
            h_conv4 = tf.nn.relu(conv2dBatchNorm(h_pool3, W_conv4, self.trainMode) + b_conv4)
        with tf.name_scope("Maxpool4"):
            h_pool4 = max_pool_2x2(h_conv4)
        with tf.name_scope("Conv5"):
            W_conv5 = weight_variable([5, 5, 80, 100], name="W_conv5")
            b_conv5 = bias_variable([100], name="b_conv5")
            h_conv5 = tf.nn.relu(conv2dBatchNorm(h_pool4, W_conv5, self.trainMode) + b_conv5)
            h_pool5 = max_pool_2x2(h_conv5)
        featureSet = h_pool5
        if self.concatinateFeatures:
            set1 = tf.image.resize_images(h_pool5, [self.size[0],self.size[1]])
            set2 = tf.image.resize_images(h_pool4, [self.size[0], self.size[1]])
            set3 = tf.image.resize_images(h_pool3, [self.size[0], self.size[1]])
            featureSet = tf.concat([set1,set2,set3],3)


        featureSetSize = featureSet.get_shape()
        featureSetSize = featureSetSize[1] * featureSetSize[2] * featureSetSize[3]
        featureSetSize = int(featureSetSize)


        with tf.name_scope("FCLayers"):
            W_fc1 = weight_variable([int(featureSetSize), 300], name="W_fc1")
            b_fc1 = bias_variable([300], name="b_fc1")

            fullyConnected1 = tf.reshape(featureSet, [-1, featureSetSize])
            h_fc1 = tf.nn.relu(tf.matmul(fullyConnected1, W_fc1) + b_fc1)

            self.keepProb = tf.placeholder(tf.float32)
            fcDropout = tf.nn.dropout(h_fc1, self.keepProb)
            #
            # W_fc2 = weight_variable([500, 500], name="W_fc2")
            # b_fc2 = bias_variable([500], name="b_fc2")
            #
            # y_conv = tf.matmul(fcDropout, W_fc2) + b_fc2

            W_fc3 = weight_variable([300, 8], name="W_fc3")
            b_fc3 = bias_variable([8], name="b_fc3")

            self.y_conv =y_conv =  tf.matmul(fcDropout, W_fc3) + b_fc3


        with tf.name_scope("loss"):
            self.cross_entropy = tf.nn.l2_loss(y_conv - y_)

            self.mySum = tf.summary.scalar('Train_loss', self.cross_entropy)
            self.validate_loss = tf.summary.scalar('Validate_loss', self.cross_entropy)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            with tf.name_scope("Train"):
                self.train_step = tf.train.AdamOptimizer(1e-5).minimize(self.cross_entropy)

        tf.summary.merge_all()

        self.train_writer = tf.summary.FileWriter('../../train', sess.graph)

        self.saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(checkpointDir)
        if ckpt and ckpt.model_checkpoint_path:
            if self.verbose>2:
                print ("PRINTING CHECKPOINT PATH")
            print(ckpt.model_checkpoint_path)
            self.saver.restore(sess, ckpt.model_checkpoint_path)

        else:
            if self.verbose>2:
                print("Starting from scratch")
            init = tf.global_variables_initializer()
            self.sess.run(init)
    def train(self, steps):
        batchSizez = self.batchSize
        keepProb = self.keepProb
        for i in range(steps):
            rand_list = np.random.randint(0, len(self.train_image) - 1, self.batchSize)
            batch = self.train_image[rand_list]
            gt = self.train_gt[rand_list]

            if i % 10 == 0:
                loss_mine = self.cross_entropy.eval(feed_dict={
                    self.x: self.train_image[0:batchSizez], self.y_: self.train_gt[0:batchSizez], keepProb: 1.0})
                print("Loss on Train : ", math.sqrt((loss_mine / self.batchSize) * 2))
                summary = self.mySum.eval(feed_dict={
                    self.x: self.train_image[0:batchSizez], self.y_: self.train_gt[0:batchSizez], keepProb: 1.0})
                self.train_writer.add_summary(summary, i)

                rand_list = np.random.randint(0, len(self.validate_image) - 1, self.batchSize)
                batch = self.validate_image[rand_list]
                gt = self.validate_gt[rand_list]
                loss_mine = self.cross_entropy.eval(feed_dict={
                    self.x: batch, self.y_: gt, keepProb: 1.0})
                print("Loss on Val : ", math.sqrt((loss_mine / self.batchSize) * 2))
                val_sum = self.validate_loss.eval(feed_dict={
                    self.x: batch, self.y_: gt, keepProb: 1.0})
                self.train_writer.add_summary(val_sum, i)
                temp_temp = np.random.randint(0, len(self.validate_image) - 1, 1)
                batch = self.validate_image[temp_temp]
                gt = self.validate_gt[temp_temp]
                response = self.y_conv.eval(feed_dict={
                    self.x: batch, self.y_: gt, keepProb: 1.0})
                cv2.circle(batch[0], (response[0][0], response[0][1]), 2, (255, 0, 0), 2)
                cv2.circle(batch[0], (gt[0][0], gt[0][1]), 2, (0, 255, 255), 2)

                cv2.circle(batch[0], (response[0][2], response[0][3]), 2, (0, 255, 0), 2)
                cv2.circle(batch[0], (gt[0][2], gt[0][3]), 2, (0, 255, 255), 2)

                cv2.circle(batch[0], (response[0][4], response[0][5]), 2, (0, 0, 255), 2)
                cv2.circle(batch[0], (gt[0][4], gt[0][5]), 2, (0, 255, 255), 2)

                cv2.circle(batch[0], (response[0][6], response[0][7]), 2, (255, 255, 0), 2)
                cv2.circle(batch[0], (gt[0][6], gt[0][7]), 2, (0, 255, 255), 2)

                img = batch[0]
                img = cv2.resize(img, (320, 320))
                cv2.imwrite("../../temp" + str(temp_temp) + ".jpg", img)
            if i % 50 == 0 and i != 0:
                self.saver.save(self.sess, self.checkpointDir + '/model.ckpt', global_step=i + 1)
            else:
                self.sess.run([self.train_step, self.mySum], feed_dict={self.x: batch, self.y_: gt, keepProb: 1.0})
