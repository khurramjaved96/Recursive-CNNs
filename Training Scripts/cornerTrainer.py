import math
import os

import cv2
import numpy as np
import tensorflow as tf

from datasetGenerator import utils

BATCH_SIZE = 100
NO_OF_STEPS = 500000
CHECKPOINT_DIR = "../corner_withoutbg1all"
DATA_DIR = "../../corner_data"
if (not os.path.isdir(CHECKPOINT_DIR)):
    os.mkdir(CHECKPOINT_DIR)
GT_DIR = "../../corner_data/gt.csv"
VALIDATION_PERCENTAGE = .20
TEST_PERCENTAGE = .10
Debug = True
size= (32,32)

train_gt = np.load("../train_gt_corners_all.npy")
train_image = np.load("../train_image_corners_all.npy")
validate_gt = np.load("../validate_gt_corners_all.npy")
validate_image = np.load("../validate_image_corners_all.npy")


utils.validate_gt(validate_gt, size)
utils.validate_gt(train_gt, size)

mean_train = np.mean(train_image, axis=(0,1,2))

mean_train = np.expand_dims(mean_train, axis=0)
mean_train = np.expand_dims(mean_train, axis=0)
mean_train = np.expand_dims(mean_train, axis=0)
print mean_train.shape
train_image = train_image - mean_train
validate_image = validate_image - mean_train


config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.1

sess = tf.InteractiveSession(config = config)


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

    x = tf.placeholder(tf.float32, shape=[None, 32, 32, 3])
    x_ = tf.image.random_contrast(x, lower=0.2, upper=1.8)
    print x_.get_shape()
    x_ = tf.image.random_brightness(x_, max_delta=50)
    print x_.get_shape()


    y_ = tf.placeholder(tf.float32, shape=[None, 2])

    h_conv1 = tf.nn.relu(conv2d(x_, W_conv1) + b_conv1)
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
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # In[ ]:

    W_fc2 = weight_variable([300, 2], name="W_fc2")
    b_fc2 = bias_variable([2], name="b_fc2")

    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2




    cross_entropy = tf.nn.l2_loss(y_conv - y_)

    mySum = tf.summary.scalar('loss', cross_entropy)
    train_step = tf.train.AdamOptimizer(1e-5).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

merged = tf.summary.merge_all()

train_writer = tf.summary.FileWriter('../train', sess.graph)

# In[ ]:

saver = tf.train.Saver()
ckpt = tf.train.get_checkpoint_state(CHECKPOINT_DIR)
if ckpt and ckpt.model_checkpoint_path:
    print ("PRINTING CHECKPOINT PATH")
    print(ckpt.model_checkpoint_path)
    init = saver.restore(sess, ckpt.model_checkpoint_path)

else:
    print("Starting from scratch")
    init = tf.global_variables_initializer()
    sess.run(init)

for i in range(NO_OF_STEPS):
    rand_list = np.random.randint(0, len(train_image) - 1, BATCH_SIZE)
    batch = train_image[rand_list]
    gt = train_gt[rand_list]
    if i % 100 == 0:
        loss_mine = cross_entropy.eval(feed_dict={
            x: train_image[0:BATCH_SIZE], y_: train_gt[0:BATCH_SIZE], keep_prob: 1.0})
        print("Loss on Train : ", math.sqrt((loss_mine/BATCH_SIZE)*2))
	

        rand_list = np.random.randint(0, len(validate_image) - 1, BATCH_SIZE)
        batch = validate_image[rand_list]
        gt = validate_gt[rand_list]
        loss_mine = cross_entropy.eval(feed_dict={
            x: batch, y_: gt, keep_prob: 1.0})
        print("Loss on Val : ", math.sqrt((loss_mine/BATCH_SIZE)*2))
        temp_temp = np.random.randint(0,len(validate_image)-1,1)
        batch = validate_image[temp_temp]
        gt = validate_gt[temp_temp]
        response = y_conv.eval(feed_dict={
            x: batch, y_: gt, keep_prob: 1.0})
        cv2.circle(batch[0], (response[0][0], response[0][1]), 2, (255,0,0),2)
        cv2.circle(batch[0], (gt[0][0], gt[0][1]), 2, (0,255,0),2)
        img = batch[0]
        img = cv2.resize(img, (320,320))
        cv2.imwrite("../temp"+str(temp_temp)+".jpg", img)
    if i % 1000 == 0 and i != 0:
        saver.save(sess, CHECKPOINT_DIR + '/model.ckpt', global_step=i + 1)
    else:
        a, summary = sess.run([train_step, mySum], feed_dict={x: batch, y_: gt, keep_prob: 0.8})
        train_writer.add_summary(summary, i)



# In[ ]:



