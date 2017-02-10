import numpy as np
import csv
import cv2
import cv2
from matplotlib import pyplot as plt
import tensorflow as tf

# In[ ]:

BATCH_SIZE = 100
NO_OF_STEPS = 50000
CHECKPOINT_DIR = "../checkpoints"
DATA_DIR = "../../DataSet Generator/data_set"
GT_DIR="../../DataSet Generator/Untitled Folder/gt1.csv"
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
            gt_list.append((ast.literal_eval(row[1])[0],ast.literal_eval(row[1])[1]))

print len(gt_list)
for a in file_names:
    img = cv2.imread(DATA_DIR+"/"+a)
    img = cv2.resize(img, (300,300))
    image_list.append(img)
print len(image_list)


# In[ ]:


gt_list = np.array(gt_list)
image_list = np.array(image_list)

train_image = image_list[0:max(1, int(len(image_list)*(1-VALIDATION_PERCENTAGE)))]
validate_image = image_list[int(len(image_list)*(1-VALIDATION_PERCENTAGE)):len(image_list)-1]

train_gt = gt_list[0:max(1, int(len(image_list)*(1-VALIDATION_PERCENTAGE)))]
validate_gt = gt_list[int(len(image_list)*(1-VALIDATION_PERCENTAGE)):len(image_list)-1]


# In[ ]:

#Sanity checks 

print gt_list[2]
rand_list = np.random.randint(0,len(image_list)-1,10)
batch = image_list[rand_list]
gt = gt_list[rand_list]
for g, b in zip(gt, batch):
    img = b
    cv2.circle(img, (g[0], g[1]), 2, (255,0,0),4)
    cv2.imwrite("../"+str(g[0]+g[1])+".jpg", img)

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

W_conv1 = weight_variable([5, 5, 3, 32], name = "W_conv1")
b_conv1 = bias_variable([32], name="b_conv1")


# In[ ]:

x = tf.placeholder(tf.float32, shape=[None, 300,300,3])
y_ = tf.placeholder(tf.float32, shape=[None, 2])


h_conv1 = tf.nn.relu(conv2d(x, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([5, 5, 32, 64], name="W_conv2")
b_conv2 = bias_variable([64], name="b_conv2")

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

W_conv3 = weight_variable([5, 5, 64, 128], name="W_conv2")
b_conv3 = bias_variable([128], name="b_conv2")

h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
h_pool3 = max_pool_2x2(h_conv3)

W_conv4 = weight_variable([5, 5, 128, 256], name="W_conv2")
b_conv4 = bias_variable([256], name="b_conv2")

h_conv4 = tf.nn.relu(conv2d(h_pool3, W_conv4) + b_conv4)
h_pool4 = max_pool_2x2(h_conv4)



# In[ ]:

W_fc1 = weight_variable([19 * 19 * 256, 1024], name = "W_fc1")
b_fc1 = bias_variable([1024], name="b_fc1")

h_pool4_flat = tf.reshape(h_pool4, [-1, 19*19*256])
h_fc1 = tf.nn.relu(tf.matmul(h_pool4_flat, W_fc1) + b_fc1)


# In[ ]:

#Adding dropout 
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)


# In[ ]:

W_fc2 = weight_variable([1024, 2], name="W_fc2")
b_fc2 = bias_variable([2], name="b_fc2")

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2


# In[ ]:



cross_entropy = tf.nn.l2_loss(y_conv - y_)
cross_entro = tf.div(cross_entropy, BATCH_SIZE)
mySum = tf.summary.scalar('loss', cross_entro)
cross_entrop = tf.Print(cross_entropy, [y_conv, y_])
train_step = tf.train.AdamOptimizer(1e-5).minimize(cross_entrop)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


merged = tf.summary.merge_all()

train_writer = tf.summary.FileWriter('../train',sess.graph)

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
  rand_list = np.random.randint(0,len(train_image)-1,BATCH_SIZE)
  batch = train_image[rand_list]
  gt = train_gt[rand_list]
  if i%100 == 0:
        
    y_results = y_conv.eval(feed_dict={
        x:batch[0:BATCH_SIZE/10], y_: gt[0:BATCH_SIZE/10], keep_prob: 1.0})
    print("Train set", y_results-gt[0:BATCH_SIZE/10])
    loss_mine = cross_entro.eval(feed_dict={
        x:train_image, y_: train_gt, keep_prob: 1.0})
    print("Loss on Train : ", loss_mine)

    rand_list = np.random.randint(0,len(validate_image)-1,BATCH_SIZE)
    batch = validate_image[rand_list]
    gt = validate_gt[rand_list]
    y_results = y_conv.eval(feed_dict={
        x:batch[0:BATCH_SIZE/10], y_: gt[0:BATCH_SIZE/10], keep_prob: 1.0})
    print("Val set", y_results-gt[0:BATCH_SIZE/10])
    loss_mine = cross_entro.eval(feed_dict={
        x:validate_image, y_: validate_gt, keep_prob: 1.0})
    print("Loss on Val : ", loss_mine)
  if i%1000==0 and i!=0:
    saver.save(sess, CHECKPOINT_DIR+ '/model.ckpt',global_step=i+1)
  else:
    a, summary = sess.run([train_step, mySum], feed_dict={x: batch, y_: gt, keep_prob: 0.5})
    train_writer.add_summary(summary, i)



# In[ ]:



