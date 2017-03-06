import numpy as np
import cv2
import tensorflow as tf
import utils
import os
import math

BATCH_SIZE = 1
NO_OF_STEPS = 50000
CHECKPOINT_DIR = "../checkpoints_4_point_multi_multilayer"
DATA_DIR = "../../4pointdata"
if (not os.path.isdir(CHECKPOINT_DIR)):
    os.mkdir(CHECKPOINT_DIR)
GT_DIR = DATA_DIR + "/gt.csv"
VALIDATION_PERCENTAGE = .1
TEST_PERCENTAGE = .01
Debug = True

size = (32,32)
# image_list, gt_list, file_name = utils.load_data_4(DATA_DIR, GT_DIR, limit=-1, size=size)
# image_list, gt_list = utils.unison_shuffled_copies(image_list, gt_list)


# print len(image_list)


# if (Debug):
#     print ("(Image_list_len, gt_list_len)", (len(image_list), len(gt_list)))
# train_image = image_list[0:max(1, int(len(image_list) * (1 - VALIDATION_PERCENTAGE)))]
# validate_image = image_list[int(len(image_list) * (1 - VALIDATION_PERCENTAGE)):len(image_list) - 1]

# train_gt = gt_list[0:max(1, int(len(image_list) * (1 - VALIDATION_PERCENTAGE)))]
# validate_gt = gt_list[int(len(image_list) * (1 - VALIDATION_PERCENTAGE)):len(image_list) - 1]
# if (Debug):
#     print ("(Train_Image_len, Train_gt_len)", (len(train_image), len(train_gt)))
#     print ("(Validate_Image_len, Validate_gt_len)", (len(validate_image), len(validate_gt)))

# np.save("train_gt", train_gt)
# np.save("train_image", train_image)
# np.save("validate_gt", validate_gt)
# np.save("validate_image", validate_image)
# 0/0
train_gt = np.load("train_gt.npy")
train_image = np.load("train_image.npy")
validate_gt = np.load("validate_gt.npy")
validate_image = np.load("validate_image.npy")
rand_list = np.random.randint(0, len(validate_image) - 1, 10)
batch = validate_image[rand_list]
gt = validate_gt[rand_list]
# for g, b in zip(gt, batch):
#     img = b
#     cv2.circle(img, (g[0], g[1]), 2, (255, 0, 0), 4)
#     cv2.imwrite("../" + str(g[0] + g[1]) + ".jpg", img)


 sum_array = myGt.sum(axis=1)
                                    tl_index = np.argmin(sum_array)
                                    tl = myGt[tl_index]
                                    br = myGt[(tl_index + 2) % 4]
                                    ptr1 = myGt[(tl_index + 1) % 4]
                                    # print "TL : ", tl
                                    # print "BR : ", br
                                    # print myGt
                                    # print myGt.shape
                                    slope = (float(tl[1] - br[1])) / float(tl[0] - br[0])
                                    # print "SLOPE = ", slope
                                    y_pred = int(slope * (ptr1[0] - br[0]) + br[1])
                                    if y_pred < ptr1[1

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
        cv2.circle(batch[0], (gt[0][0], gt[0][1]), 2, (0,255,255),2)

        cv2.circle(batch[0], (response[0][2], response[0][3]), 2, (0,255,0),2)
        cv2.circle(batch[0], (gt[0][2], gt[0][3]), 2, (0,255,255),2)

        cv2.circle(batch[0], (response[0][4], response[0][5]), 2, (0,0,255),2)
        cv2.circle(batch[0], (gt[0][4], gt[0][5]), 2, (0,255,255),2)

        cv2.circle(batch[0], (response[0][6], response[0][7]), 2, (255,255,0),2)
        cv2.circle(batch[0], (gt[0][6], gt[0][7]), 2, (0,255,255),2)

        img = batch[0]
        img = cv2.resize(img, (320,320))
        cv2.imwrite("../temp"+str(temp_temp)+".jpg", img)
    if i % 1000 == 0 and i != 0:
        saver.save(sess, CHECKPOINT_DIR + '/model.ckpt', global_step=i + 1)
    else:
        a, summary = sess.run([train_step, mySum], feed_dict={x: batch, y_: gt, keep_prob: 0.5})
        train_writer.add_summary(summary, i)



# In[ ]:



