# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# %% Borrowed utils from here: https://github.com/pkmital/tensorflow_tutorials/
import tensorflow as tf
import numpy as np
from math import sqrt

def conv2d(x, W,strides=[1, 1, 1, 1],dilations=[1,1,1,1],padding='SAME'):
    """conv2d returns a 2d convolution layer with full stride."""
    return tf.nn.conv2d(x, W, strides=strides,padding=padding)

def dilated2d2(x, W,rate,padding='VALID'):
    """conv2d returns a 2d convolution layer with full stride."""
    return tf.nn.atrous_conv2d(x, W, rate, padding=padding)

def dilated2d(x, W,rate,padding='VALID'):
    """conv2d returns a 2d convolution layer with full stride. depend on rate"""
    channels = tf.shape(W)[2:4]
    width_dia = tf.zeros([1, rate[1]-1, channels[0], channels[1]])
    expand_W1 = tf.concat([W[0:1,0:1,:,:], width_dia, W[0:1,1:2,:,:], width_dia, W[0:1,2:3,:,:]], axis=1)
    for ii in range(rate[0]-1):
        expand_W1 = tf.concat([expand_W1, tf.zeros([1, 2*rate[1]+1, channels[0], channels[1]])], axis=0)
        
    expand_W2 = tf.concat([W[1:2,0:1,:,:], width_dia, W[1:2,1:2,:,:], width_dia, W[1:2,2:3,:,:]], axis=1)
    for ii in range(rate[0]-1):
        expand_W2 = tf.concat([expand_W2, tf.zeros([1, 2*rate[1]+1, channels[0], channels[1]])], axis=0)
        
    expand_W3 = tf.concat([W[2:3,0:1,:,:], width_dia, W[2:3,1:2,:,:], width_dia, W[2:3,2:3,:,:]], axis=1)
    
    expand_W = tf.concat([expand_W1, expand_W2, expand_W3], axis = 0)
    
    return tf.nn.conv2d(x, expand_W, strides = [1,1,1,1], padding=padding)

def max_pool_2x2(x):
    """max_pool_2x2 downsamples a feature map by 2X."""
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')
# %%
def weight_variable(shape, std = 0.1, trainable = True):
    """weight_variable generates a weight variable of a given shape."""
    initial = tf.truncated_normal(shape, stddev=std)
    return tf.Variable(initial, trainable = trainable)

# %%
def bias_variable(shape, std = 0.1, trainable = True):
    '''Helper function to create a bias variable initialized with
    a constant value.
    Parameters
    ----------
    shape : list
        Size of weight variable
    '''
    initial = tf.random_normal(shape, mean=0.0, stddev=std)
    return tf.Variable(tf.abs(initial), trainable = trainable)

# %% 
def dense_to_one_hot(labels, n_classes=2):
    """Convert class labels from scalars to one-hot vectors."""
    labels = np.array(labels)
    n_labels = labels.shape[0]
    index_offset = np.arange(n_labels) * n_classes
    labels_one_hot = np.zeros((n_labels, n_classes), dtype=np.float32)
    labels_one_hot.flat[index_offset + labels.ravel()] = 1
    return labels_one_hot

def put_kernels_on_grid (kernel, pad = 1):

    '''Visualize conv. filters as an image (mostly for the 1st layer).
    Arranges filters into a grid, with some paddings between adjacent filters.
    Args:
      kernel:            tensor of shape [Y, X, NumChannels, NumKernels]
      pad:               number of black pixels around each filter (between them)
    Return:
      Tensor of shape [1, (Y+2*pad)*grid_Y, (X+2*pad)*grid_X, NumChannels].
    '''
    # get shape of the grid. NumKernels == grid_Y * grid_X
    def factorization(n):
        for i in range(int(sqrt(float(n))), 0, -1):
            if n % i == 0:
                if i == 1: print('Who would enter a prime number of filters')
                return (i, int(n / i))
            
    (grid_Y, grid_X) = factorization (kernel.get_shape()[3].value)
    #print ('grid: %d = (%d, %d)' % (kernel.get_shape()[3].value, grid_Y, grid_X))

    x_min = tf.reduce_min(kernel)
    x_max = tf.reduce_max(kernel)
    kernel = 1. - (kernel - x_min) / (x_max - x_min)

    # pad X and Y
    x = tf.pad(kernel, tf.constant( [[pad,pad],[pad, pad],[0,0],[0,0]] ), 
                                   mode = 'CONSTANT')
    x = 1. - x
    # X and Y dimensions, w.r.t. padding
    Y = kernel.get_shape()[0] + 2 * pad
    X = kernel.get_shape()[1] + 2 * pad

    channels = kernel.get_shape()[2]

    # put NumKernels to the 1st dimension
    x = tf.transpose(x, (3, 0, 1, 2))
    # organize grid on Y axis
    x = tf.reshape(x, tf.stack([grid_X, Y * grid_Y, X, channels]))

    # switch X and Y axes
    x = tf.transpose(x, (0, 2, 1, 3))
    # organize grid on X axis
    x = tf.reshape(x, tf.stack([1, X * grid_X, Y * grid_Y, channels]))

    # back to normal order (not combining with the next step for clarity)
    x = tf.transpose(x, (2, 1, 3, 0))

    # to tf.image_summary order [batch_size, height, width, channels],
    #   where in this case batch_size == 1
    x = tf.transpose(x, (3, 0, 1, 2))

    # scaling to [0, 255] is not necessary for tensorboard
    return x

def dilatedNet(FLAGS, mu, x_f, x_l, x_r, keep_prob=1.0, isTraining=False):
    
    face = x_f - mu
    left_eye = x_l - mu
    right_eye = x_r - mu
    
    vgg = np.load(FLAGS.vgg_dir)
    with tf.variable_scope("transfer"):
        W_conv1_1 = tf.Variable(vgg['conv1_1_W'])
        b_conv1_1 = tf.Variable(vgg['conv1_1_b'])      
        W_conv1_2 = tf.Variable(vgg['conv1_2_W'])
        b_conv1_2 = tf.Variable(vgg['conv1_2_b'])
        
        W_conv2_1 = tf.Variable(vgg['conv2_1_W'])
        b_conv2_1 = tf.Variable(vgg['conv2_1_b'])      
        W_conv2_2 = tf.Variable(vgg['conv2_2_W'])
        b_conv2_2 = tf.Variable(vgg['conv2_2_b'])
        
    del vgg
    
    # create data flow
    face_h_conv1_1 = tf.nn.relu(conv2d(face, W_conv1_1) + b_conv1_1)
    face_h_conv1_2 = tf.nn.relu(conv2d(face_h_conv1_1, W_conv1_2) + b_conv1_2)
    face_h_pool1 = max_pool_2x2(face_h_conv1_2)
 
    face_h_conv2_1 = tf.nn.relu(conv2d(face_h_pool1, W_conv2_1) + b_conv2_1)
    face_h_conv2_2 = tf.nn.relu(conv2d(face_h_conv2_1, W_conv2_2) + b_conv2_2) / 100.
        
    rf = [[2,2], [3,3], [5,5], [11,11]]
    num_face = (64, 128, 64, 64, 128, 256, 64)
    with tf.variable_scope("face"):
        
        face_W_conv2_3 = weight_variable([1, 1, num_face[1], num_face[2]], std = 0.125)
        face_b_conv2_3 = bias_variable([num_face[2]], std = 0.001)
        
        face_W_conv3_1 = weight_variable([3, 3, num_face[2], num_face[3]], std = 0.06)
        face_b_conv3_1 = bias_variable([num_face[3]], std = 0.001)
        face_W_conv3_2 = weight_variable([3, 3, num_face[3], num_face[3]], std = 0.06)
        face_b_conv3_2 = bias_variable([num_face[3]], std = 0.001)
        
        face_W_conv4_1 = weight_variable([3, 3, num_face[3], num_face[4]], std = 0.08)
        face_b_conv4_1 = bias_variable([num_face[4]], std = 0.001)
        face_W_conv4_2 = weight_variable([3, 3, num_face[4], num_face[4]], std = 0.07)
        face_b_conv4_2 = bias_variable([num_face[4]], std = 0.001)
        
        face_W_fc1 = weight_variable([6*6*num_face[4], num_face[5]], std = 0.035)
        face_b_fc1 = bias_variable([num_face[5]], std = 0.001)
        
        face_W_fc2 = weight_variable([num_face[5], num_face[6]], std = 0.1)
        face_b_fc2 = bias_variable([num_face[6]], std = 0.001)
        
        """ original network """
        face_h_conv2_3 = tf.nn.relu(conv2d(face_h_conv2_2, face_W_conv2_3) + face_b_conv2_3) 
        face_h_conv2_3_norm = tf.layers.batch_normalization(face_h_conv2_3, training = isTraining, scale=False, renorm=True,
                                                            name="f_conv2_3")
        
        face_h_conv3_1 = tf.nn.relu(dilated2d(face_h_conv2_3_norm, face_W_conv3_1, rf[0]) + face_b_conv3_1)
        face_h_conv3_1_norm = tf.layers.batch_normalization(face_h_conv3_1, training = isTraining, scale=False, renorm=True,
                                                            name="f_conv3_1")
        
        face_h_conv3_2 = tf.nn.relu(dilated2d(face_h_conv3_1_norm, face_W_conv3_2, rf[1]) + face_b_conv3_2) 
        face_h_conv3_2_norm = tf.layers.batch_normalization(face_h_conv3_2, training = isTraining, scale=False, renorm=True,
                                                            name="f_conv3_2")
        
        face_h_conv4_1 = tf.nn.relu(dilated2d(face_h_conv3_2_norm, face_W_conv4_1, rf[2]) + face_b_conv4_1)
        face_h_conv4_1_norm = tf.layers.batch_normalization(face_h_conv4_1, training = isTraining, scale=False, renorm=True,
                                                            name="f_conv4_1")
        
        face_h_conv4_2 = tf.nn.relu(dilated2d(face_h_conv4_1_norm, face_W_conv4_2, rf[3]) + face_b_conv4_2)
        face_h_conv4_2_norm = tf.layers.batch_normalization(face_h_conv4_2, training = isTraining, scale=False, renorm=True,
                                                            name="f_conv4_2")
    
        face_h_pool4_flat = tf.reshape(face_h_conv4_2_norm, [-1, 6*6*num_face[4]])
        
        face_h_fc1 = tf.nn.relu(tf.matmul(face_h_pool4_flat, face_W_fc1) + face_b_fc1)
        face_h_fc1_norm = tf.layers.batch_normalization(face_h_fc1, training = isTraining, scale=False, renorm=True,
                                                        name="f_fc1")
        face_h_fc1_drop = tf.nn.dropout(face_h_fc1_norm, keep_prob)
        
        face_h_fc2 = tf.nn.relu(tf.matmul(face_h_fc1_drop, face_W_fc2) + face_b_fc2)
        face_h_fc2_norm = tf.layers.batch_normalization(face_h_fc2, training = isTraining, scale=False, renorm=True,
                                                        name="f_fc2")
        face_h_fc2_drop = tf.nn.dropout(face_h_fc2_norm, keep_prob)
        
    # left eye
    """ original """
    eye1_h_conv1_1 = tf.nn.relu(conv2d(left_eye, W_conv1_1) + b_conv1_1)
    eye1_h_conv1_2 = tf.nn.relu(conv2d(eye1_h_conv1_1, W_conv1_2) + b_conv1_2)
    eye1_h_pool1 = max_pool_2x2(eye1_h_conv1_2)
    
    eye1_h_conv2_1 = tf.nn.relu(conv2d(eye1_h_pool1, W_conv2_1) + b_conv2_1)
    eye1_h_conv2_2 = tf.nn.relu(conv2d(eye1_h_conv2_1, W_conv2_2) + b_conv2_2) / 100.
    
    eye2_h_conv1_1 = tf.nn.relu(conv2d(right_eye, W_conv1_1) + b_conv1_1)
    eye2_h_conv1_2 = tf.nn.relu(conv2d(eye2_h_conv1_1, W_conv1_2) + b_conv1_2)
    eye2_h_pool1 = max_pool_2x2(eye2_h_conv1_2)
    
    eye2_h_conv2_1 = tf.nn.relu(conv2d(eye2_h_pool1, W_conv2_1) + b_conv2_1)
    eye2_h_conv2_2 = tf.nn.relu(conv2d(eye2_h_conv2_1, W_conv2_2) + b_conv2_2) / 100.
    
    r = [[2,2], [3,3], [4,5], [5,11]]
    num_cls1 = (64, 128, 64, 64, 128, 256)
    with tf.variable_scope("eye"):

        eye_W_conv2_3 = weight_variable([1, 1, num_cls1[1], num_cls1[2]], std = 0.125)
        eye_b_conv2_3 = bias_variable([num_cls1[2]], std = 0.001)
        
        eye_W_conv3_1 = weight_variable([3, 3, num_cls1[2], num_cls1[3]], std = 0.06)
        eye_b_conv3_1 = bias_variable([num_cls1[3]], std = 0.001)
        eye_W_conv3_2 = weight_variable([3, 3, num_cls1[3], num_cls1[3]], std = 0.06)
        eye_b_conv3_2 = bias_variable([num_cls1[3]], std = 0.001)
        
        eye_W_conv4_1 = weight_variable([3, 3, num_cls1[3], num_cls1[4]], std = 0.06)
        eye_b_conv4_1 = bias_variable([num_cls1[4]], std = 0.001)
        eye_W_conv4_2 = weight_variable([3, 3, num_cls1[4], num_cls1[4]], std = 0.04)
        eye_b_conv4_2 = bias_variable([num_cls1[4]], std = 0.001)
        
        eye1_W_fc1 = weight_variable([4*6*num_cls1[4], num_cls1[5]], std = 0.026)
        eye1_b_fc1 = bias_variable([num_cls1[5]], std = 0.001)
        
        eye2_W_fc1 = weight_variable([4*6*num_cls1[4], num_cls1[5]], std = 0.026)
        eye2_b_fc1 = bias_variable([num_cls1[5]], std = 0.001)
        #flow
        """ original """
        eye1_h_conv2_3 = tf.nn.relu(conv2d(eye1_h_conv2_2, eye_W_conv2_3) + eye_b_conv2_3) 
        eye1_h_conv2_3_norm = tf.layers.batch_normalization(eye1_h_conv2_3, training = isTraining, scale=False, renorm=True,
                                                            name="e_conv2_3")
        
        eye1_h_conv3_1 = tf.nn.relu(dilated2d(eye1_h_conv2_3_norm, eye_W_conv3_1, r[0]) + eye_b_conv3_1)
        eye1_h_conv3_1_norm = tf.layers.batch_normalization(eye1_h_conv3_1, training = isTraining, scale=False, renorm=True,
                                                            name="e_conv3_1")
        
        eye1_h_conv3_2 = tf.nn.relu(dilated2d(eye1_h_conv3_1_norm, eye_W_conv3_2, r[1]) + eye_b_conv3_2) 
        eye1_h_conv3_2_norm = tf.layers.batch_normalization(eye1_h_conv3_2, training = isTraining, scale=False, renorm=True,
                                                            name="e_conv3_2")
        
        eye1_h_conv4_1 = tf.nn.relu(dilated2d(eye1_h_conv3_2_norm, eye_W_conv4_1, r[2]) + eye_b_conv4_1)
        eye1_h_conv4_1_norm = tf.layers.batch_normalization(eye1_h_conv4_1, training = isTraining, scale=False, renorm=True,
                                                            name="e_conv4_1")
        
        eye1_h_conv4_2 = tf.nn.relu(dilated2d(eye1_h_conv4_1_norm, eye_W_conv4_2, r[3]) + eye_b_conv4_2)
        eye1_h_conv4_2_norm = tf.layers.batch_normalization(eye1_h_conv4_2, training = isTraining, scale=False, renorm=True,
                                                            name="e_conv4_2")
    
        eye1_h_pool4_flat = tf.reshape(eye1_h_conv4_2_norm, [-1, 4*6*num_cls1[4]])
        
        eye1_h_fc1 = tf.nn.relu(tf.matmul(eye1_h_pool4_flat, eye1_W_fc1) + eye1_b_fc1)
        eye1_h_fc1_norm = tf.layers.batch_normalization(eye1_h_fc1, training = isTraining, scale=False, renorm=True,
                                                        name="e1_fc1")
        eye1_h_fc1_drop = tf.nn.dropout(eye1_h_fc1_norm, keep_prob)
    
        # right eye
        eye2_h_conv2_3 = tf.nn.relu(conv2d(eye2_h_conv2_2, eye_W_conv2_3) + eye_b_conv2_3)
        eye2_h_conv2_3_norm = tf.layers.batch_normalization(eye2_h_conv2_3, training = isTraining, scale=False, renorm=True,
                                                            name="e_conv2_3", reuse=True)
        
        eye2_h_conv3_1 = tf.nn.relu(dilated2d(eye2_h_conv2_3_norm, eye_W_conv3_1, r[0]) + eye_b_conv3_1)
        eye2_h_conv3_1_norm = tf.layers.batch_normalization(eye2_h_conv3_1, training = isTraining, scale=False, renorm=True,
                                                            name="e_conv3_1", reuse=True)
        
        eye2_h_conv3_2 = tf.nn.relu(dilated2d(eye2_h_conv3_1_norm, eye_W_conv3_2, r[1]) + eye_b_conv3_2)
        eye2_h_conv3_2_norm = tf.layers.batch_normalization(eye2_h_conv3_2, training = isTraining, scale=False, renorm=True,
                                                            name="e_conv3_2", reuse=True)
        
        eye2_h_conv4_1 = tf.nn.relu(dilated2d(eye2_h_conv3_2_norm, eye_W_conv4_1, r[2]) + eye_b_conv4_1)
        eye2_h_conv4_1_norm = tf.layers.batch_normalization(eye2_h_conv4_1, training = isTraining, scale=False, renorm=True,
                                                            name="e_conv4_1", reuse=True)
        
        eye2_h_conv4_2 = tf.nn.relu(dilated2d(eye2_h_conv4_1_norm, eye_W_conv4_2, r[3]) + eye_b_conv4_2)
        eye2_h_conv4_2_norm = tf.layers.batch_normalization(eye2_h_conv4_2, training = isTraining, scale=False, renorm=True,
                                                            name="e_conv4_2", reuse=True)
        
        eye2_h_pool4_flat = tf.reshape(eye2_h_conv4_2_norm, [-1, 4*6*num_cls1[4]])
        
        eye2_h_fc1 = tf.nn.relu(tf.matmul(eye2_h_pool4_flat, eye2_W_fc1) + eye2_b_fc1)
        eye2_h_fc1_norm = tf.layers.batch_normalization(eye2_h_fc1, training = isTraining, scale=False, renorm=True,
                                                        name="e2_fc1")
        eye2_h_fc1_drop = tf.nn.dropout(eye2_h_fc1_norm, keep_prob)
    # combine both eyes
    num_comb = (num_face[-1]+2*num_cls1[-1], 256)
    with tf.variable_scope("combine"):
        
        cls1_W_fc2 = weight_variable([num_comb[0], num_comb[1]], std = 0.07)
        cls1_b_fc2 = bias_variable([num_comb[1]], std = 0.001)
            
        cls1_W_fc3 = weight_variable([num_comb[1], 2], std = 0.125)
        cls1_b_fc3 = bias_variable([2], std = 0.001)
                
        cls1_h_fc1_drop = tf.concat([face_h_fc2_drop, eye1_h_fc1_drop, eye2_h_fc1_drop], axis = 1)
        cls1_h_fc2 = tf.nn.relu(tf.matmul(cls1_h_fc1_drop, cls1_W_fc2) + cls1_b_fc2)
        cls1_h_fc2_norm = tf.layers.batch_normalization(cls1_h_fc2, training = isTraining, scale=False, renorm=True,
                                                        name="c_fc2")
        cls1_h_fc2_drop = tf.nn.dropout(cls1_h_fc2_norm, keep_prob)
        
        y_train = tf.matmul(cls1_h_fc2_drop, cls1_W_fc3) + cls1_b_fc3
        y_conv = y_train / 10.
    
    num_batch = 1
    face_h_trans = face+mu
    face2 = tf.image.resize_images(face, [64, 64])
    h_trans = tf.concat([face2+mu, tf.ones([num_batch, 64, 1, 3]), 
                         left_eye+mu, tf.ones([num_batch, 64, 1, 3]),
                         right_eye+mu], axis = 2)
    
    return y_conv, face_h_trans, h_trans