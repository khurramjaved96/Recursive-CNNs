import numpy as np
import cv2
import tensorflow as tf
import utils
import os
import math

BATCH_SIZE = 1000
NO_OF_STEPS = 50000
CHECKPOINT_DIR = "../checkpoints"
DATA_DIR = "../../DataSet Generator/data_set"
if (not os.path.isdir(CHECKPOINT_DIR)):
    os.mkdir(CHECKPOINT_DIR)
GT_DIR = "../../DataSet Generator/Untitled Folder/gt1.csv"
VALIDATION_PERCENTAGE = .20
TEST_PERCENTAGE = .10
Debug = True

image_list, gt_list = utils.load_data(DATA_DIR, GT_DIR, limit=50000, size=(32, 32))

if (Debug):
    print ("(Image_list_len, gt_list_len)", (len(image_list), len(gt_list)))
train_image = image_list[0:max(1, int(len(image_list) * (1 - VALIDATION_PERCENTAGE)))]
validate_image = image_list[int(len(image_list) * (1 - VALIDATION_PERCENTAGE)):len(image_list) - 1]

train_gt = gt_list[0:max(1, int(len(image_list) * (1 - VALIDATION_PERCENTAGE)))]
validate_gt = gt_list[int(len(image_list) * (1 - VALIDATION_PERCENTAGE)):len(image_list) - 1]
if (Debug):
    print ("(Train_Image_len, Train_gt_len)", (len(train_image), len(train_gt)))
    print ("(Validate_Image_len, Validate_gt_len)", (len(validate_image), len(validate_gt)))

rand_list = np.random.randint(0, len(image_list) - 1, 10)
batch = validate_image[rand_list]
gt = validate_gt[rand_list]
for g, b in zip(gt, batch):
    img = b
    cv2.circle(img, (g[0], g[1]), 2, (255, 0, 0), 4)
    cv2.imwrite("../" + str(g[0] + g[1]) + ".jpg", img)

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
y_ = tf.placeholder(tf.float32, shape=[None, 2])

h_conv1 = tf.nn.relu(conv2d(x, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([5, 5, 20, 40], name="W_conv2")
b_conv2 = bias_variable([40], name="b_conv2")

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

print h_pool2.get_shape()

temp_size = h_pool2.get_shape()
temp_size = temp_size[1] * temp_size[2] * temp_size[3]

# In[ ]:

W_fc1 = weight_variable([8*8*40, 128], name="W_fc1")
b_fc1 = bias_variable([128], name="b_fc1")

h_pool4_flat = tf.reshape(h_pool2, [-1, 8*8*40])
h_fc1 = tf.nn.relu(tf.matmul(h_pool4_flat, W_fc1) + b_fc1)

# In[ ]:

# Adding dropout
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# In[ ]:

W_fc2 = weight_variable([128, 2], name="W_fc2")
b_fc2 = bias_variable([2], name="b_fc2")

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

# In[ ]:


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
    if i % 1000 == 0 and i != 0:
        saver.save(sess, CHECKPOINT_DIR + '/model.ckpt', global_step=i + 1)
    else:
        a, summary = sess.run([train_step, mySum], feed_dict={x: batch, y_: gt, keep_prob: 0.5})
        train_writer.add_summary(summary, i)



# In[ ]:



