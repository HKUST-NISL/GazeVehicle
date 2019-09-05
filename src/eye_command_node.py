from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

import rospy
import tensorflow as tf

from tf_utils import conv2d, dilated2d, max_pool_2x2, weight_variable, bias_variable, put_kernels_on_grid, dense_to_one_hot,dilatedNet


import cv2
import numpy as np


if __name__ == '__main__':
    global cap_region_x_begin,cap_region_y_end,threshold,blurValue,bgSubThreshold,learningRate,isBgCaptured,triggerSwitch,skinkernel
    
    print('hola')
    # parameters
    cap_region_x_begin=0.5  # start point/total width
    cap_region_y_end=0.8  # start point/total width

    # parameter 1
    threshold = 40  #  BINARY threshold


    blurValue = 41  # GaussianBlur parameter
    bgSubThreshold = 50
    learningRate = 0

    # variables
    isBgCaptured = 0   # bool, whether the background captured
    triggerSwitch = False  # if true, keyborad simulator works
    skinkernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    
    
    #added from ZC's code 
    parser = argparse.ArgumentParser()

    parser.add_argument('--vgg_dir', type=str,
                        default='./asserts/vgg16_weights.npz',
                        help='Directory for pretrained vgg16')
    
    parser.add_argument("--shape-predictor", type=str,
                        default='./asserts/shape_predictor_68_face_landmarks.dat',
                            help="Path to facial landmark predictor")
    
    parser.add_argument("--camera_mat", type=str,
                        default='./asserts/camera_matrix.mat',
                            help="Path to camera matrix")
    FLAGS, unparsed = parser.parse_known_args()
    # tf.app.run()
    # tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
    
    #added from ZC's code

#     try:
#         talker()
#     except rospy.ROSInterruptException:
#         pass
# ``