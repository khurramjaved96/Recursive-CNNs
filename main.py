import numpy as np
import csv
import cv2
import cv2
from matplotlib import pyplot as plt
import tensorflow as tf

# In[ ]:

BATCH_SIZE = 100
NO_OF_STEPS = 20000
CHECKPOINT_DIR = "../checkpoints"
DATA_DIR = "../../Dicta/data/data_set"
GT_DIR="../../Dicta/gt1.csv"
VALIDATION_PERCENTAGE = .20
TEST_PERCENTAGE=.10


#Loading data
gt_list=[]
file_names=[]
image_list=[]

with open(GT_DIR, 'r') as csvfile:
      spamreader = csv.reader(csvfile, delimiter=',',quotechar='|', quoting=csv.QUOTE_MINIMAL)
      import ast
      a = 0
      temp = 0
      for row in spamreader:
            temp+=1
            if(temp ==50):
                break
            file_names.append(row[0])
            gt_list.append((ast.literal_eval(row[1])[0],anumpy.random.choicest.literal_eval(row[1])[1]))

print len(gt_list)
for a in file_names:
    img = cv2.imread(DATA_DIR+"/"+a)
    img = cv2.resize(img, (300,300))
    image_list.append(img)
print len(image_list)


# In[ ]:


gt_list = np.array(gt_list)
image_list = np.array(image_list)

train_image = image_list[0:max(1, len(image_list)*(1-VALIDATION_PERCENTAGE))]
validate_image = image_list[len(image_list)*(1-VALIDATION_PERCENTAGE)):len(image_list)-1]

train_gt = gt_list[0:max(1, len(image_list)*(1-VALIDATION_PERCENTAGE))]
validate_gt = gt_list[len(image_list)*(1-VALIDATION_PERCENTAGE):len(image_list)-1]

0/0

# In[ ]:

#Sanity checks 

print gt_list[2]
for a in range(0,5):
    img = image_list[a]
    cv2.circle(img, (gt_list[a][0], gt_list[a][1]), 2, (255,0,0),4)
    plt.imshow(img)
    plt.show()


# In[ ]:

sess = tf.InteractiveSession()


# In[ ]:

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)


# In[ ]:

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')


# In[ ]:

W_conv1 = weight_variable([5, 5, 3, 32])
b_conv1 = bias_variable([32])


# In[ ]:

x = tf.placeholder(tf.float32, shape=[None, 300,300,3])
y_ = tf.placeholder(tf.float32, shape=[None, 2])

x_image = tf.reshape(x, [-1,300,300,3])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)



# In[ ]:

W_fc1 = weight_variable([75 * 75 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 75*75*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)


# In[ ]:

#Adding dropout 
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)


# In[ ]:

W_fc2 = weight_variable([1024, 2])
b_fc2 = bias_variable([2])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2


# In[ ]:



cross_entropy = tf.reduce_mean(tf.nn.l2_loss(y_conv - y_))
cross_entrop = tf.Print(cross_entropy, [y_conv, y_])
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entrop)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
sess.run(tf.global_variables_initializer())


# In[ ]:

for i in range(20000):
  rand_list = np.random.randint(0,len(image_list)-1,5)
  batch = image_list[rand_list]
  gt = gt_list[rand_list]
  if i%100 == 0:
    train_accuracy = accuracy.eval(feed_dict={
        x:batch, y_: gt, keep_prob: 1.0})
    print("step %d, training accuracy %g"%(i, train_accuracy))
  train_step.run(feed_dict={x: batch, y_: gt, keep_prob: 0.5})

print("test accuracy %g"%accuracy.eval(feed_dict={
    x: image_list, y_: gt, keep_prob: 1.0}))


# In[ ]:



