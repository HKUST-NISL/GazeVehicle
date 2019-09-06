#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 19 16:12:58 2017

@author: zchenbc
"""
import random
import numpy as np
import cv2
import math

def randomFlip(input_image, flip_prob = 0.5):
    """ flip the single image horizontally with probability flip_prob"""
    tmp = random.random()
    if tmp <= flip_prob:
        fliped_image = input_image[:, ::-1,:];
        output_image = fliped_image
        FLAG = 1
    else:
        output_image = input_image
        FLAG = 0
  
    return output_image, FLAG

"""
def randomCrop(input_image, out_size, cropOffset):
    # random crop the single input image based on size
    y_max = int(input_image.shape[0] - out_size[0] + 1)
    x_max = int(input_image.shape[1] - out_size[1] + 1)
    
    x = random.randrange(math.floor(x_max/2.-cropOffset*0.4), math.ceil(x_max/2.+cropOffset*0.4))
    y = random.randrange(math.floor(-cropOffset), math.ceil(+1.2*cropOffset))
    #x = x_max / 2
    #y = y_max / 2
    padding_image = np.zeros((input_image.shape[0]+2*out_size[0], input_image.shape[1]+2*out_size[1],
                              input_image.shape[2]), dtype=np.float32)
    padding_image[out_size[0]: out_size[0]+input_image.shape[0], 
                  out_size[1]: out_size[1]+input_image.shape[1],
                  :] = input_image
    output_image = padding_image[y+out_size[0]: y+2*out_size[0], x+out_size[1]: x+2*out_size[1], :]
    dx = x - x_max/2.
    dy = y - y_max/2.
    return output_image, dx, dy
"""
def randomCrop(input_image, x_c, y_c, out_size, cropOffset):
    #random crop the single input image based on size
    
    padding_image = np.zeros((input_image.shape[0]+2*out_size[0], input_image.shape[1]+2*out_size[1],
                              input_image.shape[2]), dtype=np.float32)
    padding_image[out_size[0]: out_size[0]+input_image.shape[0], 
                  out_size[1]: out_size[1]+input_image.shape[1],
                  :] = input_image
    
    if cropOffset == 0:
        x = x_c.astype(np.int_)
        y = y_c.astype(np.int_)
    else:
        #y_max = int(input_image.shape[0] - out_size[0] + 1)
        #x_max = int(input_image.shape[1] - out_size[1] + 1)
        x_cropOffset = np.min((cropOffset, np.absolute(x_c - out_size[1]/2), np.absolute(x_c - out_size[1])))
        y_cropOffset = np.min((cropOffset, np.absolute(y_c - out_size[0]/2), np.absolute(y_c - out_size[0])))
    
        x_cropOffset = np.max((x_cropOffset, 1.))
        y_cropOffset = np.max((y_cropOffset, 1.))

        #print(y_cropOffset)
        #print((math.floor(y_c - y_cropOffset), math.ceil(y_c + y_cropOffset)))
        x = random.randrange(math.floor(x_c - x_cropOffset), math.ceil(x_c + x_cropOffset))
        y = random.randrange(math.floor(y_c - y_cropOffset), math.ceil(y_c + y_cropOffset))
        #print([x-x_c, y-y_c])
        #x = x_max / 2
        #y = y_max / 2

    output_image = padding_image[y+out_size[0]/2: y+3*out_size[0]/2, x+out_size[1]/2: x+3*out_size[1]/2, :]
    dx = x - x_c
    dy = y - y_c
    return output_image, dx, dy

def grayNhist(input_image):
    gray_img = cv2.cvtColor(input_image, cv2.COLOR_RGB2GRAY)
    gray_img = gray_img.astype(np.uint8)
    equ = cv2.equalizeHist(gray_img)
    output_image = np.stack((equ,equ,equ), axis = 2)
    return output_image

def randomNoise(input_image, std = 0.025):
    """add random Gaussian noise to image"""
    #gauss = np.random.normal(0, std, input_image.shape)
    noise = np.zeros(input_image.shape, dtype = np.float32)
    cv2.randn(noise, np.zeros(3), std * np.ones(3))
    output_image = input_image + noise

    return output_image

def point_to_matrix(points, desiredDist = 224 * 0.6):
    # obtain the affine matrix given two points
    points = points.astype(np.int)
    dX = points[2] - points[0]
    dY = points[3] - points[1]
    angle = np.degrees(np.arctan2(dY, dX))
    dist = np.sqrt((dX ** 2) + (dY ** 2))
    scale = desiredDist / dist
    eyesCenter = ((points[0] + points[2]) // 2,
                  (points[1] + points[3]) // 2)
    # grab the rotation matrix for rotating and scaling the face
    M = cv2.getRotationMatrix2D((79.5, 79.5), angle, scale)
    return M, scale

def eye_location_affine(eyeLoc, M):
    current_eye = np.array([], dtype=np.float32).reshape((0,2))
    for ii in range(eyeLoc.shape[0]):
        x = eyeLoc[ii, 0]
        y = eyeLoc[ii, 1]
        eye_tmp = np.array([[x, y, 1.]], dtype=np.float32).transpose()
        eye_tmp = np.matmul(M, eye_tmp)
        eye_tmp = eye_tmp.transpose()
        current_eye = np.vstack((current_eye, eye_tmp[:, :2]))

    return current_eye

def Crop(input_image, landmarks, out_size):
    """ change the resolution of input_imag from min_factor to 1"""

    input_size = input_image.shape[0:2]
        
    current_image = np.array(input_image, dtype=np.float32)
    
    current_landmarks = landmarks.astype(np.float32)
    current_landmarks += input_image.shape[0] / 2.
    eyeCrop = 0
    
    left_eye = np.array(current_landmarks[36: 42, :])
    left_eye_mean = np.mean(left_eye[[0,3], :], axis=0, keepdims=True)
    left_eye_top = np.mean(left_eye[[1,2], :], axis=0, keepdims=True)
    left_eye_down = np.mean(left_eye[[4,5], :], axis=0, keepdims=True)
            
    right_eye = np.array(current_landmarks[42: 48, :])
    right_eye_mean = np.mean(right_eye[[0,3],:], axis=0, keepdims=True)
    right_eye_top = np.mean(right_eye[[1,2],:], axis=0, keepdims=True)
    right_eye_down = np.mean(right_eye[[4,5],:], axis=0, keepdims=True)
                
    # rezie the whole image based on the center of the eyes first
    s_x = 120. / (right_eye_mean[0,0]-left_eye_mean[0,0])
            #print(right_eye_mean[0,0]-left_eye_mean[0,0])
    s_y = s_x
    
    face_image = cv2.resize(current_image, (0, 0), 
                           fx=s_x, fy=s_y,interpolation = cv2.INTER_CUBIC)
    
    current_eye = np.mean(np.concatenate((left_eye_mean*s_x, right_eye_mean*s_x), axis = 0),
                                  axis=0, keepdims=True)
                
    M_left, s_left = point_to_matrix(left_eye[[0,3], :].reshape(-1), desiredDist = 96*0.7)
    M_right, s_right = point_to_matrix(right_eye[[0,3], :].reshape(-1), desiredDist = 96*0.7)
    
    # left eye
    left_image = cv2.warpAffine(current_image, M_left, input_size)
    top = eye_location_affine(left_eye_top, M_left)
    down = eye_location_affine(left_eye_down, M_left)
    tmp = eye_location_affine(left_eye_mean, M_left)
    y_c = 0.5*(down[0,1] + top[0,1])
    x_c = tmp[0,0]
    left_image, dx, dy = randomCrop(left_image, x_c, y_c, out_size, eyeCrop)
    
    # right eye
    right_image = cv2.warpAffine(current_image, M_right, input_size)
    top = eye_location_affine(right_eye_top, M_right)
    down = eye_location_affine(right_eye_down, M_right)
    tmp = eye_location_affine(right_eye_mean, M_right)
    y_c = 0.5*(down[0,1] + top[0,1])
    x_c = tmp[0,0]
    right_image, dx, dy = randomCrop(right_image, x_c, y_c,out_size, eyeCrop)
    
    # face
    #current_eye = np.mean(np.vstack((left_eye_mean, right_eye_mean)), axis = 0)
    
    current_eye = current_eye.astype(np.int)
    current_eye = current_eye.reshape(-1)
    y_c = current_eye[1] + 40
    x_c = current_eye[0]
    
    face_image, dx, dy = randomCrop(face_image, x_c, y_c, 
                                       (320,320), 0)
        
    left_image = grayNhist(left_image)
    right_image = grayNhist(right_image)
    face_image = grayNhist(face_image)
    face_image = cv2.resize(face_image, (out_size[0],out_size[0]))
    
    #face_image -= face_image        
    current_image = np.hstack((face_image, left_image, right_image))
    current_image = np.minimum(current_image, 255)
    current_image = np.maximum(current_image, 0)
    
    output_image = np.expand_dims(current_image, axis = 0)
    
    return output_image

def WarpNCrop(input_image, landmarks, inv_CameraMat, cam_new):
    """ crop images by perspective warping """
    eye_lm = np.vstack([landmarks[36:48, :].transpose(), np.ones((1,12))]) # eye landmarks
    eye_lm_w = np.dot(inv_CameraMat, eye_lm) # in world coordinate
    
    rigtEye_w = eye_lm_w[:,0:6]
    leftEye_w = eye_lm_w[:,6:]
    rigtEye_wc = np.mean(rigtEye_w, axis=1, keepdims=True)
    leftEye_wc = np.mean(leftEye_w, axis=1, keepdims=True)
    # warp the images to make the face at the center
    
    fc_c_world = np.mean(eye_lm_w, axis=1, keepdims=True)
    # calculate the rotation matrix
    forward = fc_c_world / np.linalg.norm(fc_c_world)
    hRx = leftEye_wc - rigtEye_wc
    down = np.cross(forward.reshape(-1), hRx.reshape(-1))
    down = down / np.linalg.norm(down)
    down.shape = (3,-1)
    right = np.cross(down.reshape(-1), forward.reshape(-1))
    right = right / np.linalg.norm(right)
    right.shape = (3,-1)
    
    rotMat = np.hstack([right, down, forward])
    #rotMat = np.eye(3)
    warpMat = cam_new * np.mat(rotMat.transpose()) * inv_CameraMat
    warped_image = cv2.warpPerspective(input_image, warpMat, input_image.shape[1::-1])
        
    # obtain the warped landmarks
    eye_lm_warped = warpMat * eye_lm
    eye_lm_warped = np.array(eye_lm_warped[:2] / eye_lm_warped[2]).astype(np.int_)
    eye_lm_store = np.mean(eye_lm_warped, axis=1)[1]
    
    #img_size = warped_image.shape
    """ plot landmarks
    cv2.circle(warped_image, (img_size[1]//2, img_size[0]//2), 10, (0, 0, 255), -1)
    for ii in range(eye_lm_warped.shape[1]):
        cv2.circle(warped_image, (eye_lm_warped[0,ii], eye_lm_warped[1,ii]), 10, (255, 0, 0), -1)
    """
    rigtEye_warped = eye_lm_warped[:,0:6]
    leftEye_warped = eye_lm_warped[:,6:]
    rigtEye_warpedc = np.mean(rigtEye_warped, axis=1)
    leftEye_warpedc = np.mean(leftEye_warped, axis=1)
    # obtain face image
    scale = 32. / np.abs(leftEye_warpedc[0]-rigtEye_warpedc[0])
    face_image = cv2.resize(warped_image, (0, 0), 
                           fx=scale, fy=scale,interpolation = cv2.INTER_CUBIC)
    f_size = face_image.shape
    face_image = face_image[(f_size[0]//2-60+5):(f_size[0]//2+60+5), (f_size[1]//2-60):(f_size[1]//2+60)]
    face_image = grayNhist(face_image)
    face_image = face_image[12:-12, 12:-12]
    eye_lm_store = eye_lm_store * scale - (f_size[0]//2-43)
    #cv2.circle(face_image, (48, eye_lm_store), 5, (255, 0, 0), -1)
    # obtain left eye
    x_c = np.mean(leftEye_warped[0,[0,3]])
    x_c = x_c.astype(np.int_)
    y_c = np.mean(leftEye_warped[1,[0,3]])
    y_c = y_c.astype(np.int_)
    left_image = warped_image[y_c-80:y_c+80, x_c-80:x_c+80]
    M_left, s_left = point_to_matrix(leftEye_warped[:,[0,3]].transpose().reshape(-1), desiredDist = 96*0.7)

    left_image = cv2.warpAffine(left_image, M_left, (160,160))
    left_image = left_image[80-40-5:80+40-5, 80-60:80+60]
    left_image = grayNhist(left_image)
    left_image = left_image[8:-8, 12:-12]
    # obtain right eye
    x_c = np.mean(rigtEye_warped[0,[0,3]])
    x_c = x_c.astype(np.int_)
    y_c = np.mean(rigtEye_warped[1,[0,3]])
    y_c = y_c.astype(np.int_)
    rigt_image = warped_image[y_c-80:y_c+80, x_c-80:x_c+80]
    M_rigt, s_rigt = point_to_matrix(rigtEye_warped[:,[0,3]].transpose().reshape(-1), desiredDist = 96*0.7)
    rigt_image = cv2.warpAffine(rigt_image, M_rigt, (160,160))
    rigt_image = rigt_image[80-40-5:80+40-5, 80-60:80+60]    
    rigt_image = grayNhist(rigt_image)
    rigt_image = rigt_image[8:-8, 12:-12]
    
    return face_image, left_image, rigt_image, eye_lm_store, fc_c_world / np.linalg.norm(fc_c_world)

def WarpNDraw(input_image, eye_lm, gaze_vec, cam_new, inv_cam_new):
    """ crop images by perspective warping """
    
    #eye_lm = 48
    output_image = np.array(input_image)
    eye_lm_2d = np.array([[48], [eye_lm], [1.]])
    eye_lm_3d = np.matmul(inv_cam_new, eye_lm_2d)

    g_end = np.matmul(cam_new, eye_lm_3d - 0.05*gaze_vec.transpose())
    #print(g_end[1,0])
    g_end = g_end[:2,0] / g_end[2,0]
    
    cv2.line(output_image, (48, int(eye_lm)), tuple(g_end), (0, 255, 0), 2)
    
    return output_image
