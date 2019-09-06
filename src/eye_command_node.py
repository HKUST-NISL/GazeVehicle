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
import dlib

import face_utils

# parameters setting
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



if __name__ == '__main__':
    
    print('Starting...')
    
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

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(FLAGS.shape_predictor)

    scale = 0.5
    video_capture = cv2.VideoCapture(0)
    video_capture.set(3, 1920)
    video_capture.set(4, 1080)

    success, frame = video_capture.read()

    while(success):
        frame = frame[:,::-1,:].copy()
        frame_small = cv2.resize(frame, None, fx=scale, fy=scale,interpolation = cv2.INTER_CUBIC)
        gray_small = cv2.cvtColor(frame_small, cv2.COLOR_BGR2GRAY)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        rects_small = detector(gray_small, 1)

        for (ii, rect_s) in enumerate(rects_small):
            
            tmp = np.array([rect_s.left(), rect_s.top(), rect_s.right(), rect_s.bottom()]) / scale
            tmp = tmp.astype(np.long)

            # get face rect
            rect = dlib.rectangle(tmp[0], tmp[1], tmp[2], tmp[3])
            (x, y, w, h) = face_utils.rect_to_bb(rect)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # get face landmarks
            shape = predictor(frame, rect)
            shape = face_utils.shape_to_np(shape)
            for (x, y) in shape:
		        cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)
            
        cv2.imshow("frame", frame)
        cv2.waitKey(10)

        success, frame = video_capture.read()

        

    
