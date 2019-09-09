from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

import rospy
import tensorflow as tf

from eye_model import dilatedNet

import cv2
import numpy as np
import dlib
import scipy.io as spio

import face_utils
import preprocess_eye as pre_eye
from geometry_msgs.msg import Twist

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


def encode_msg(status, direction):
    msg = Twist()
    msg.linear.x = 0
    msg.linear.y = 0
    msg.linear.z = 0

    msg.angular.x = 0
    msg.angular.y = 0
    msg.angular.z = 0

    speed = 0.02
    ang_sped = 0.05
    
    if status == 'open' and direction == 'forward':
        msg.linear.x = speed

    if status == 'open' and direction == 'left':
        msg.angular.z = ang_sped

    if status == 'open' and direction == 'right':
        msg.angular.z = -ang_sped

    if status == 'open' and direction == 'backward':
        msg.linear.x = -speed

    rospy.loginfo(msg)
    
    return msg
    

if __name__ == '__main__':

    pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
    rospy.init_node('talker', anonymous=True)
    rate = rospy.Rate(10) # 10hz
    
    print('Starting...')
    
    #added from ZC's code 
    parser = argparse.ArgumentParser()

    parser.add_argument('--vgg_dir', type=str,
                        default='./models/vgg16_weights.npz',
                        help='Directory for pretrained vgg16')
    
    parser.add_argument("--shape-predictor", type=str,
                        default='./models/shape_predictor_68_face_landmarks.dat',
                            help="Path to facial landmark predictor")
    
    parser.add_argument("--camera_mat", type=str,
                        default='./models/camera_matrix.mat',
                            help="Path to camera matrix")

    parser.add_argument("--gaze_model", type=str,
                        default='./models/model21.ckpt',
                            help="Path to eye gaze model")

    parser.add_argument("--camera_ind", type=str,
                        default=0,
                            help="camera index")

    FLAGS, unparsed = parser.parse_known_args()

    scale = 0.25
    input_size = (64, 96)
    gaze_lock = np.zeros(6, np.float64)
    gaze_unlock = np.zeros(15, np.float64)
    gaze_cursor = np.zeros(1, np.int_)
    shape = None
    face_backup = np.zeros((input_size[1], input_size[1], 3))
    left_backup = np.zeros((input_size[1], input_size[1], 3))
    rigt_backup = np.zeros((input_size[1], input_size[1], 3))
    print('define video capturer')

    # define model
    mu = np.array([123.68, 116.779, 103.939], dtype = \
        np.float32).reshape((1, 1, 3))
    print('defined model')

    dataset = spio.loadmat(FLAGS.camera_mat)
    
    cameraMat = dataset['camera_matrix']
    inv_cameraMat = np.linalg.inv(cameraMat)
    cam_new = np.mat([[1536., 0., 960.],[0., 1536., 540.],[0., 0., 1.]])
    cam_face = np.mat([[1536., 0., 48.],[0., 1536., 48.],[0., 0., 1.]])
    inv_cam_face = np.linalg.inv(cam_face)
    print('got camera matrix')   

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(FLAGS.shape_predictor)

    # define network
    # input image
    x_f = tf.placeholder(tf.float32, [1, input_size[1], input_size[1], 3])
    x_l = tf.placeholder(tf.float32, [1, input_size[0], input_size[1], 3])
    x_r = tf.placeholder(tf.float32, [1, input_size[0], input_size[1], 3])
    
    y_conv, face_h_trans, h_trans = dilatedNet(FLAGS, mu, x_f, x_l, x_r)

    saver = tf.train.Saver()

    print("camera index: ", FLAGS.camera_ind, type(FLAGS.camera_ind))
    
    video_capture = cv2.VideoCapture(int(FLAGS.camera_ind))
    video_capture.set(3, 1920)
    video_capture.set(4, 1080)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, FLAGS.gaze_model)

        success, frame = video_capture.read()
        

        while(success and (not rospy.is_shutdown())):
            frame = frame[:,::-1,:].copy()
            frame = cv2.resize(frame, (1920, 1080))
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
                cv2.rectangle(frame_small, (int(x*scale), int(y*scale)), (int((x+w)*scale), int((y+h)*scale)), (0, 255, 0), 2)

                # get face landmarks
                shape = predictor(frame, rect)
                shape = face_utils.shape_to_np(shape)
                status = face_utils.get_mouth_status(shape)

                for i, (x, y) in enumerate(shape):
                    cv2.circle(frame_small, (int(x*scale), int(y*scale)), 1, (0, 0, 255), -1)

                # eye gaze estimation
                face_img, left_img, rigt_img, eye_lm, fc_c_world = \
                    pre_eye.WarpNCrop(frame[:,:,::-1], shape, inv_cameraMat, cam_new)

                y_result, eye_tensor, face_tensor = sess.run([y_conv, h_trans, face_h_trans], feed_dict={
                                                       x_f: face_img[None, :],
                                                       x_l: left_img[None, :],
                                                       x_r: rigt_img[None, :]})

                direction = face_utils.angle_to_direction(y_result[0])

                print('mouth: %s eye: %s' % (status, direction))

                msg = encode_msg(status, direction)
                pub.publish(msg)

                break
            
            cv2.imshow("frame", frame_small)
            cv2.imshow("face_img", face_img)
            cv2.imshow("left_img", left_img)
            cv2.imshow("rigt_img", rigt_img)
            cv2.waitKey(10)

            success, frame = video_capture.read()

        

    
