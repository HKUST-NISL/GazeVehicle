#! /usr/bin/python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

import rospy
import tensorflow as tf

from utils.eye_model import dilatedNet

import cv2
import numpy as np
import dlib
import scipy.io as spio

from utils import face_utils
import utils.preprocess_eye as pre_eye

from Tkinter import *
import tkMessageBox
import Tkinter as tk
from threading import Thread
from geometry_msgs.msg import Twist

import time
import pyautogui
import keyboard


from utils.gaze_projection import gaze_to_screen

# Dimensions of Isamu's laptop in centimeters
# xps 17
# SCREEN_W = 37.0
# SCREEN_H = 23.0
# resolution_H = 2400
# resolution_W = 3840

# hd monitor
SCREEN_W = 34.5
SCREEN_H = 19.8
resolution_H = 1080
resolution_W = 1920
res = (resolution_W, resolution_H)

# pixel to physical size ratio (pixel/cm)
pixelr_H = 1.0 * resolution_H / SCREEN_H
pixelr_W = 1.0 * resolution_W / SCREEN_W

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

def dwell_direction(x, y, resolution_H, resolution_W):
    unit_x = resolution_W / 5
    unit_y = resolution_H / 5
    
    if (x > 2 * unit_x) and (x < resolution_W - 2 * unit_x) and (y > unit_y) and (y < resolution_H - unit_y):
        return "stop"

    elif (x < 2 * unit_x) and (y > unit_y) and (y < resolution_H - 2* unit_y):
        return "left"
    
    elif (x > 2* unit_x) and (x < resolution_W - 2 * unit_x) and (y < 2 * unit_y):
        return "forward"

    elif (x > resolution_W - 2 * unit_x) and (y > unit_y) and (y < resolution_H - unit_y):
        return "right"

    else:
        return "backward"

# multithread code (I need this)
class DrawingThread(Thread):

    def __init__(self):
        Thread.__init__(self)
    
    def run(self):

        #=================================================================#
        #============= setting up the buttons for the thread =============#
        #=================================================================#
        root = tk.Tk()
        canvas1 = Canvas(root, width=130, height=130, background='blue')
        canvas2 = Canvas(root, width=130, height=130, background='blue')
        canvas3 = Canvas(root, width=130, height=130, background='blue')
        canvas4 = Canvas(root, width=130, height=130, background='blue')
        canvas5 = Canvas(root, width=130, height=130, background='blue')

        canvas6 = Canvas(root, width=130, height=130, background='blue')
        canvas7 = Canvas(root, width=130, height=130, background='blue')
        canvas8 = Canvas(root, width=130, height=130, background='blue')
        canvas9 = Canvas(root, width=130, height=130, background='blue')

        canvas1.grid(row=0, column = 1)
        canvas2.grid(row=2, column = 1)
        canvas3.grid(row=1, column = 0)
        canvas4.grid(row=1, column = 2)
        canvas5.grid(row=1, column = 1)

        canvas6.grid(row=0, column = 0)
        canvas7.grid(row=0, column = 2)
        canvas8.grid(row=2, column = 0)
        canvas9.grid(row=2, column = 2)

        root.update_idletasks()
        root.update()
        
        #root.mainloop()
            
  
def is_moving(msg):

    if msg is None:
        return False

    if msg.linear.x == 0 and \
        msg.linear.y == 0 and \
        msg.linear.z == 0 and \
        msg.angular.x == 0 and \
        msg.angular.y == 0 and \
        msg.angular.z == 0:
        return False

    return True

def encode_msg(status, direction, spacekey, last_msg):

    if (status == 'open' or spacekey) and is_moving(last_msg):
        return last_msg 


    msg = Twist()
    msg.linear.x = 0
    msg.linear.y = 0
    msg.linear.z = 0

    msg.angular.x = 0
    msg.angular.y = 0
    msg.angular.z = 0

    speed = 0.1 # originally 0.05
    ang_sped = 0.15
    cur_moving = False


    # rospy.loginfo((spacekey, status, direction))
    
    # centres are spaced by 50 pixels
    """
    pyautogui.FAILSAFE = False
    pyautogui.moveTo(mouse_centreX, mouse_centreY)
    """
    
    #=================================================================#
    #============= send move signals (to revert back) ================#
    #=================================================================#

    if direction == 'forward':
     
        msg.linear.x = speed
        cur_moving = True
        # print("YEEEtttttttt")
            

    elif direction == 'left':
       
        msg.angular.z = ang_sped
        cur_moving = True
        # print("YEEEtttttttt")

    elif direction == 'right':

        msg.angular.z = -ang_sped

        cur_moving = True 
        # print("YEEEtttttttt")

    elif direction == 'backward':

        msg.linear.x = -speed
        cur_moving = True
        # print("YEEEtttttttt")

    else:
        # print("excuseme, I'm w a i t i n g")
        pass

    # rospy.loginfo(msg)
    
    return msg



if __name__ == '__main__':
    print("start GUI")
   
    gui_thread = DrawingThread()

    # start the drawing thread
    # gui_thread.start()
    
    get_input = True

    root1 = tk.Tk()
    root2 = tk.Tk()
    root3 = tk.Tk()
    root4 = tk.Tk()

    root1.title("UP")
    root2.title("DOWN")
    root3.title("LEFT")
    root4.title("RIGHT")

    canvas1 = Canvas(root1, width=130, height=130, background='blue')
    canvas2 = Canvas(root2, width=130, height=130, background='blue')
    canvas3 = Canvas(root3, width=130, height=130, background='blue')
    canvas4 = Canvas(root4, width=130, height=130, background='blue')
    
    canvas1.grid(row=0, column = 0)
    canvas2.grid(row=0, column = 0)
    canvas3.grid(row=0, column = 0)
    canvas4.grid(row=0, column = 0)

    # =================================================================================== #
    pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
    rospy.init_node('talker', anonymous=True)
    rate = rospy.Rate(10) # 10hz

    import rospkg

    rospack = rospkg.RosPack()
    model_dir = rospack.get_path('interfaces')
    
    print('Starting...')
    
    #added from ZC's code 
    parser = argparse.ArgumentParser()

    parser.add_argument('--vgg_dir', type=str,
                        default=model_dir+'/../../models/vgg16_weights.npz',
                        help='Directory for pretrained vgg16')
    
    parser.add_argument("--shape-predictor", type=str,
                        default=model_dir+'/../../models/shape_predictor_68_face_landmarks.dat',
                            help="Path to facial landmark predictor")
    
    parser.add_argument("--camera_mat", type=str,
                        default=model_dir+'/../../models/camera_matrix.mat',
                            help="Path to camera matrix")

    parser.add_argument("--gaze_model", type=str,
                        default=model_dir+'/../../models/model21.ckpt',
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

    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.3

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, FLAGS.gaze_model)

        success, frame = video_capture.read()
        
        status = None
        direction = None
        spacekey = False
        last_msg = None


        while(success and (not rospy.is_shutdown())):
            frame = frame[:,::-1,:].copy()
            frame = cv2.resize(frame, (1920, 1080))
            frame_small = cv2.resize(frame, None, fx=scale, fy=scale,interpolation = cv2.INTER_CUBIC)
            gray_small = cv2.cvtColor(frame_small, cv2.COLOR_BGR2GRAY)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            rects_small = detector(gray_small, 1)

            face_img = np.zeros((64, 96)).astype(np.uint8)
            left_img = np.zeros((64, 96)).astype(np.uint8)
            rigt_img = np.zeros((64, 96)).astype(np.uint8)

            cur_status = None
            cur_direction = None
            cur_spacekey = None

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
                cur_status = face_utils.get_mouth_status(shape)

                for i, (x, y) in enumerate(shape):
                    cv2.circle(frame_small, (int(x*scale), int(y*scale)), 1, (0, 0, 255), -1)

                # eye gaze estimation
                face_img, left_img, rigt_img, eye_lm, fc_c_world = \
                    pre_eye.WarpNCrop(frame[:,:,::-1], shape, inv_cameraMat, cam_new)

                if face_img.shape[0] != 96 or face_img.shape[1] != 96 or \
                    left_img.shape[0] != 64 or left_img.shape[1] != 96 or \
                    rigt_img.shape[0] != 64 or rigt_img.shape[1] != 96 :
                    break

                y_result, eye_tensor, face_tensor = sess.run([y_conv, h_trans, face_h_trans], feed_dict={
                                                    x_f: face_img[None, :],
                                                    x_l: left_img[None, :],
                                                    x_r: rigt_img[None, :]})

                gaze_p, face_p = gaze_to_screen(y_result[0], rect_s, scale)

                mock_direction = dwell_direction((gaze_p[0] - SCREEN_W / 2) * pixelr_W, gaze_p[1] * pixelr_H, resolution_H, resolution_W)
                # print("scaled dimensions: W: %d H: %d Direction: %s" %((gaze_p[0] - SCREEN_W / 2) * pixelr_W, gaze_p[1] * pixelr_H, mock_direction))
                X = (gaze_p[0] + SCREEN_W / 2) * pixelr_W
                Y = gaze_p[1] * pixelr_H 

                # cur_direction = face_utils.angle_to_direction(y_result[0])
                cur_direction = dwell_direction(X, Y, resolution_H, resolution_W)

                # print('mouth: %s eye: %s' % (cur_status, cur_direction))

                break
            
            cv2.imshow("frame", frame_small)
            cv2.imshow("face_img", face_img)
            # cv2.imshow("left_img", left_img)
            # cv2.imshow("rigt_img", rigt_img)
            c = cv2.waitKey(10)
            
            if c == 32:
                spacekey = (not spacekey)

            # rospy.loginfo(spacekey)

            status = cur_status
            direction = cur_direction

            

            msg = encode_msg(status, direction, spacekey, last_msg)
            last_msg = msg
            pub.publish(msg)

            success, frame = video_capture.read()

            #=================================================================#
            #============= setting up the buttons for the thread =============#
            #=================================================================#
            

            # UP
            if get_input and direction == "forward" :
               
                canvas1.create_rectangle(0, 0, 130, 130, outline="#fb0", fill="green")
                canvas2.create_rectangle(0, 0, 130, 130, outline="#fb0", fill="blue")
                canvas3.create_rectangle(0, 0, 130, 130, outline="#fb0", fill="blue")
                canvas4.create_rectangle(0, 0, 130, 130, outline="#fb0", fill="blue")
                get_input = True


            # DOWN
            elif get_input and direction == "backward":
                
                canvas1.create_rectangle(0, 0, 130, 130, outline="#fb0", fill="blue")
                canvas2.create_rectangle(0, 0, 130, 130, outline="#fb0", fill="green")
                canvas3.create_rectangle(0, 0, 130, 130, outline="#fb0", fill="blue")
                canvas4.create_rectangle(0, 0, 130, 130, outline="#fb0", fill="blue")
                get_input = True
                

            # LEFT
            elif get_input and direction == "left":
                
                canvas1.create_rectangle(0, 0, 130, 130, outline="#fb0", fill="blue")
                canvas2.create_rectangle(0, 0, 130, 130, outline="#fb0", fill="blue")
                canvas3.create_rectangle(0, 0, 130, 130, outline="#fb0", fill="green")
                canvas4.create_rectangle(0, 0, 130, 130, outline="#fb0", fill="blue")
                get_input = True


            # RIGHT
            elif get_input and direction == "right":

                canvas1.create_rectangle(0, 0, 130, 130, outline="#fb0", fill="blue")
                canvas2.create_rectangle(0, 0, 130, 130, outline="#fb0", fill="blue")
                canvas3.create_rectangle(0, 0, 130, 130, outline="#fb0", fill="blue")
                canvas4.create_rectangle(0, 0, 130, 130, outline="#fb0", fill="green")
                get_input = True


            # STOP
            elif direction == "stop":
                
                canvas1.create_rectangle(0, 0, 130, 130, outline="#fb0", fill="orange")
                canvas2.create_rectangle(0, 0, 130, 130, outline="#fb0", fill="orange")
                canvas3.create_rectangle(0, 0, 130, 130, outline="#fb0", fill="orange")
                canvas4.create_rectangle(0, 0, 130, 130, outline="#fb0", fill="orange")
                get_input = True

                if direction == "forward" :
               
                    canvas1.create_rectangle(0, 0, 130, 130, outline="#fb0", fill="green")
                    canvas2.create_rectangle(0, 0, 130, 130, outline="#fb0", fill="orange")
                    canvas3.create_rectangle(0, 0, 130, 130, outline="#fb0", fill="orange")
                    canvas4.create_rectangle(0, 0, 130, 130, outline="#fb0", fill="orange")


                # DOWN
                elif direction == "backward":
                    
                    canvas1.create_rectangle(0, 0, 130, 130, outline="#fb0", fill="orange")
                    canvas2.create_rectangle(0, 0, 130, 130, outline="#fb0", fill="green")
                    canvas3.create_rectangle(0, 0, 130, 130, outline="#fb0", fill="orange")
                    canvas4.create_rectangle(0, 0, 130, 130, outline="#fb0", fill="orange")

                # LEFT
                elif direction == "left":
                    
                    canvas1.create_rectangle(0, 0, 130, 130, outline="#fb0", fill="orange")
                    canvas2.create_rectangle(0, 0, 130, 130, outline="#fb0", fill="orange")
                    canvas3.create_rectangle(0, 0, 130, 130, outline="#fb0", fill="green")
                    canvas4.create_rectangle(0, 0, 130, 130, outline="#fb0", fill="orange")


                # RIGHT
                elif direction == "right":

                    canvas1.create_rectangle(0, 0, 130, 130, outline="#fb0", fill="orange")
                    canvas2.create_rectangle(0, 0, 130, 130, outline="#fb0", fill="orange")
                    canvas3.create_rectangle(0, 0, 130, 130, outline="#fb0", fill="orange")
                    canvas4.create_rectangle(0, 0, 130, 130, outline="#fb0", fill="green")

            
            root1.update_idletasks()
            root1.update()

            root2.update_idletasks()
            root2.update()

            root3.update_idletasks()
            root3.update()

            root4.update_idletasks()
            root4.update()
