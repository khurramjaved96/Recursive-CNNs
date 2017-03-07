import numpy as np
import cv2
import tensorflow as tf
import utils
import os
import math

BATCH_SIZE = 1
NO_OF_STEPS = 50000
CHECKPOINT_DIR = "../checkpoints_4_point_multi_multilayer/"
DATA_DIR = "../../DataSet Generator/data_set"
GT_DIR = "../../DataSet Generator/Untitled Folder/gt1.csv"
VALIDATION_PERCENTAGE = .20
TEST_PERCENTAGE = .10
Debug = True

img = cv2.imread("../temp/044.jpg")
# img = cv2.resize(img, (600,600))
o_img = np.copy(img)

sess = tf.InteractiveSession()


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

x = tf.placeholder(tf.float32, shape=[None, 32, 32, 3])
y_ = tf.placeholder(tf.float32, shape=[None, 8])

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
# In[ ]:

sess = tf.InteractiveSession()


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

x = tf.placeholder(tf.float32, shape=[None, 32, 32, 3])
y_ = tf.placeholder(tf.float32, shape=[None, 8])

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
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# In[ ]:

W_fc2 = weight_variable([500, 500], name="W_fc2")
b_fc2 = bias_variable([500], name="b_fc2")

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

W_fc3 = weight_variable([500, 8], name="W_fc3")
b_fc3 = bias_variable([8], name="b_fc3")

y_conv = tf.matmul(y_conv, W_fc3) + b_fc3



# In[ ]:


cross_entropy = tf.nn.l2_loss(y_conv - y_)

mySum = tf.summary.scalar('loss', cross_entropy)
train_step = tf.train.AdamOptimizer(3e-3).minimize(cross_entropy)
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
import timeit

y = None
x_start = 0
y_start = 0
up_scale_factor = (img.shape[0], img.shape[1])
crop_size = [img.shape[0] * .8, img.shape[1] * .8]
start = timeit.timeit()
myImage = np.copy(o_img)

CROP_FRAC = .95
print myImage.shape
for counter in range(0, 40):

    img_temp = cv2.resize(myImage, (32, 32))
    img_temp = np.expand_dims(img_temp, axis=0)
    response = y_conv.eval(feed_dict={
        x: img_temp, keep_prob: 1.0})
    print response

end = timeit.timeit()
print end - start

# In[ ]:


