
# coding: utf-8




import tensorflow as tf
import numpy as np
import os, pdb
import cv2
import numpy as np
import random as rn
import threading
import time
from tqdm import tqdm
from sklearn import metrics
import utils
global n_classes
import triplet_loss as tri
import os.path

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
'''
#===========================================================================
Parameters:
        TRAIN_WO_SPEC_GAN: Excluding GAN. E.f. progressGAN means We dont include progressGAN for tranining phase
        n_classes: Number of classes (2 for now. one for fake and one for real)
        data_dir:  The path to the file list directory
        image_dir: The path to the images directory (if the image list is stored in absoluate path, set this to './')
        margin:    Marginal value in triplet loss function
Data Preparation
        All training image list should put on the subfolder 'data' named by train_wo_[TRAIN_WO_SPEC_GAN].txt, wheere
        the text file should have image path with its label (which GAN) such that 
        image_path1 0
        image_path2 1
        image_path3 5
        image_path4 0
        The data list in validation set is the same structure with training set.
#===========================================================================
'''
TRAIN_WO_SPEC_GAN = 'progressGAN'                         
n_classes = 2
data_dir = '/home/jsj/WQ/Fake-Face-Images-Detection-Tensorflow/data'
#image_dir = 'D:/Fake-Face-Images-Detection-Tensorflow/data/img'
batch_size = 64
display_step = 20
learning_rate = tf.placeholder(tf.float32)      # Learning rate to be fed
lr = 1e-4     
margin = 0.8

#========================Mode basic components============================
def activation(x,name="activation"):
    return tf.nn.swish(x, name=name)
    
def conv2d(name, l_input, w, b, s, p):
    l_input = tf.nn.conv2d(l_input, w, strides=[1,s,s,1], padding=p, name=name)
    l_input = l_input+b

    return l_input

def batchnorm(conv, isTraining, name='bn'):
    return tf.layers.batch_normalization(conv, training=isTraining, name="bn"+name)

def initializer(in_filters, out_filters, name, k_size=3):
    w1 = tf.get_variable(name+"W", [k_size, k_size, in_filters, out_filters], initializer=tf.truncated_normal_initializer())
    b1 = tf.get_variable(name+"B", [out_filters], initializer=tf.truncated_normal_initializer())
    return w1, b1
  
def residual_block(in_x, in_filters, out_filters, stride, isDownSampled, name, isTraining, k_size=3):
    global ema_gp
    # first convolution layer
    if isDownSampled:
      in_x = tf.nn.avg_pool(in_x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
      
    x = batchnorm(in_x, isTraining, name=name+'FirstBn')
    x = activation(x)
    w1, b1 = initializer(in_filters, in_filters, name+"first_res", k_size=k_size)
    x = conv2d(name+'r1', x, w1, b1, 1, "SAME")

    # second convolution layer
    x = batchnorm(x, isTraining, name=name+'SecondBn')
    x = activation(x)
    w2, b2 = initializer(in_filters, out_filters, name+"Second_res",k_size=k_size)
    x = conv2d(name+'r2', x, w2, b2, 1, "SAME")
    
    if in_filters != out_filters:
        difference = out_filters - in_filters
        left_pad = difference // 2
        right_pad = difference - left_pad
        identity = tf.pad(in_x, [[0, 0], [0, 0], [0, 0], [left_pad, right_pad]])
        return x + identity
    else:
        return in_x + x


'''
#===========================================================================
Network architecture based on ResNet
#===========================================================================
'''      
def ResNet(_X, isTraining):
    global n_classes
    w1 = tf.get_variable("initWeight", [7, 7, 3, 64], initializer=tf.truncated_normal_initializer())
    b1 = tf.get_variable("initBias", [64], initializer=tf.truncated_normal_initializer())
    initx = conv2d('conv1', _X, w1, b1, 4, "VALID")
    
    
    filters_num = [64,96,128]
    block_num = [2,4,3]
    l_cnt = 1
    x = initx
    
    # ============Feature extraction network with kernel size 3x3============
    
    for i in range(len(filters_num)):
        for j in range(block_num[i]):
          
            if ((j==block_num[i]-1) & (i<len(filters_num)-1)):
                x = residual_block(x, filters_num[i], filters_num[i+1], 2, True, 'ResidualBlock%d'%(l_cnt), isTraining)
                print('[L-%d] Build %dth connection layer %d from %d to %d channels' % (l_cnt, i, j, filters_num[i], filters_num[i+1]))
            else:
                x = residual_block(x, filters_num[i], filters_num[i], 1, False, 'ResidualBlock%d'%(l_cnt), isTraining)
                print('[L-%d] Build %dth residual block %d with %d channels' % (l_cnt,i, j, filters_num[i]))
            l_cnt +=1
    
    layer_33 = x
    x = initx
    
    # ============Feature extraction network with kernel size 5x5============
    for i in range(len(filters_num)):
        for j in range(block_num[i]):
          
            if ((j==block_num[i]-1) & (i<len(filters_num)-1)):
                x = residual_block(x, filters_num[i], filters_num[i+1], 2, True, 'Residual5Block%d'%(l_cnt), isTraining, k_size=5)
                print('[L-%d] Build %dth connection layer %d from %d to %d channels' % (l_cnt, i, j, filters_num[i], filters_num[i+1]))
            else:
                x = residual_block(x, filters_num[i], filters_num[i], 1, False, 'Residual5Block%d'%(l_cnt), isTraining, k_size=5)
                print('[L-%d] Build %dth residual block %d with %d channels' % (l_cnt,i, j, filters_num[i]))
            l_cnt +=1
    layer_55 = x
    print("Layer33's shape", layer_33.get_shape().as_list())
    print("Layer55's shape", layer_55.get_shape().as_list())

    x = tf.concat([layer_33, layer_55], 3)
    
    # ============ Classifier Learning============
    
    x_shape = x.get_shape().as_list()
    dense1 = x_shape[1]*x_shape[2]*x_shape[3]
    W = tf.get_variable("featW", [dense1, 128], initializer=tf.truncated_normal_initializer())
    b = tf.get_variable("featB", [128], initializer=tf.truncated_normal_initializer())
    dense1 = tf.reshape(x, [-1, dense1])
    feat = tf.nn.softmax(tf.matmul(dense1, W) + b)
    
    with tf.variable_scope('Final'):
        x = batchnorm(x, isTraining, name='FinalBn')
        x = activation(x)
        wo, bo=initializer(filters_num[-1]*2, n_classes, "FinalOutput")
        x = conv2d('final', x, wo, bo, 1, "SAME")
        saliency = tf.argmax(x, 3)
        x=tf.reduce_mean(x, [1, 2])

        W = tf.get_variable("FinalW", [n_classes, n_classes], initializer=tf.truncated_normal_initializer())
        b = tf.get_variable("FinalB", [n_classes], initializer=tf.truncated_normal_initializer())

        out = tf.matmul(x, W) + b
                            

    return out, feat, saliency


#==========================================================================
#=============Reading data in multithreading manner========================
#==========================================================================
def read_labeled_image_list(image_list_file):
    f = open(image_list_file, 'r')
    filenames = []
    labels = []

    for line in f:
        filename, label=line.strip().split(',')
#        filename = training_img_dir+filename
        filenames.append(filename)
        labels.append(int(label))
#    print(filenames)
#    print(labels)
    return filenames, labels
    
    
def read_images_from_disk(input_queue, size1=64):
    label = input_queue[1]
    fn=input_queue[0]
    file_contents = tf.read_file(input_queue[0])
    example = tf.image.decode_jpeg(file_contents, channels=3)
    
    #example = tf.image.decode_png(file_contents, channels=3, name="dataset_image") # png fo rlfw
    example=tf.image.resize_images(example, [size1,size1])
    return example, label, fn
    
def setup_inputs(sess, filenames, image_size=64, crop_size=64, isTest=False, batch_size=128):
    
    # Read each image file
    image_list, label_list = read_labeled_image_list(filenames)

    images = tf.cast(image_list, tf.string)
    labels = tf.cast(label_list, tf.int64)
     # Makes an input queue
    if isTest is False:
        isShuffle = True
        numThr = 4
    else:
        isShuffle = False
        numThr = 1
        
    input_queue = tf.train.slice_input_producer([images, labels], shuffle=isShuffle)
    image, y ,fn = read_images_from_disk(input_queue)

    channels = 3
    image.set_shape([None, None, channels])
        
    # Crop and other random augmentations
    if isTest is False:
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_saturation(image, .95, 1.05)
        image = tf.image.random_brightness(image, .05)
        image = tf.image.random_contrast(image, .95, 1.05)
        
    image = tf.cast(image, tf.float32)/255.0
    
    image, y,fn = tf.train.batch([image, y, fn], batch_size=batch_size, capacity=batch_size*3, num_threads=numThr, name='labels_and_images')

    tf.train.start_queue_runners(sess=sess)

    return image, y, fn, len(label_list)




config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

print("Preparing the test data...")
pth3 = os.path.join(data_dir, "test_wo_%s.txt"%(TRAIN_WO_SPEC_GAN))
test_data, test_labels, filelist3, glen3 = setup_inputs(sess, pth3, batch_size=batch_size)
print("Found %d test images..." % (glen3))




with tf.variable_scope("ResNet") as scope:
    testpred, _,_ = ResNet(test_data, True)
    scope.reuse_variables()

with tf.name_scope('Test_Accuracy'):
    correct_prediction3 = tf.equal(tf.argmax(testpred, 1), test_labels)
    accuracy3 = tf.reduce_mean(tf.cast(correct_prediction3, tf.float32))


log_dir='/home/jsj/WQ/Fake-Face-Images-Detection-Tensorflow/logs'
save_path='/home/jsj/WQ/Fake-Face-Images-Detection-Tensorflow/checkpoints'
saver = tf.train.Saver()
merge = tf.summary.merge_all()
init = tf.global_variables_initializer()



with tf.Session() as sess:
    sess.run(init)
    test_total_batch = int(glen3/batch_size)
    ckpt = tf.train.latest_checkpoint(save_path)# 
    saver.restore(sess, ckpt)
    print('finish loading model!')
    test_writer = tf.summary.FileWriter(log_dir + '/restore')
    testacc = []
    test_vis = []
    test_tis = []
    for k in tqdm(range(test_total_batch)):
        summary,a3, test_vi, test_ti = sess.run([merge, accuracy3, tf.argmax(testpred, 1), test_labels])
        testacc.append(a3)
        test_vis.append(test_vi)
        test_tis.append(test_ti)
    test_tis = np.reshape(np.asarray(test_tis), [-1])
    test_vis = np.reshape(np.asarray(test_vis), [-1])
    precision_test = metrics.precision_score(test_tis, test_vis)
    recall_test = metrics.recall_score(test_tis, test_vis)
    test_writer.add_summary(summary, k)
    print("Test Accuracy=%.6f, Precision=%.6f, Recall=%.6f" % (np.mean(testacc), precision_test, recall_test))






