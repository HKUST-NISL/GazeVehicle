import sys, os
import matplotlib.pyplot as plt
import numpy as np
import rosbag
import cv2
import pandas as pd

from analyze_bags import get_cmd_pose_gaze

def get_acc_dist(poses):
    acc_dist = [0]
    dist_motion = 0
    last_p = poses[0][1].position
    
    for i in range(1, len(poses)):
        cur_p = poses[i][1].position

        dist_motion += np.sqrt((cur_p.x - last_p.x)**2 + (cur_p.y - last_p.y)**2)

        last_p = cur_p
        acc_dist.append(dist_motion)
    
    return acc_dist

def get_dist(poses):
    acc_dist = get_acc_dist(poses)
    left_d = [acc_dist[-1]- d for d in acc_dist]
    t = [(t-poses[0][0]).to_nsec()/1e+9 for (t, p) in poses]

    return t, left_d

def draw_gaze1(img_path, gazes, color=(0, 0, 255)):
    img = cv2.imread(img_path)
    h, w, _ = img.shape
    px = []
    py = []
    for gaze in gazes:
        g = gaze[1]
        
        px.append(int(g.x * w / 34) + w/2)
        py.append(int(g.y * h / 19.5))

        if len(px) > 1:
            cv2.line(img, (px[-2], py[-2]), (px[-1], py[-1]), color, 2)

    return img

def draw_gaze2(img_path, gazes, color=(255, 0, 0)):
    img = cv2.imread(img_path)
    h, w, _ = img.shape
    px = []
    py = []
    for gaze in gazes:
        g = gaze[1]
        
        if g.y < 0:
            g.y *= 1.5

        px.append(int(g.x * w / 34) + w/2)
        py.append(int(g.y * (h/2) / 19.5) + h/2)

        if len(px) > 1:
            cv2.line(img, (px[-2], py[-2]), (px[-1], py[-1]), color, 2)

    return img

if __name__ == "__main__":
        
    g_img_path = './assets/gaze_key.png'
    v_img_path = './assets/virt_key.png'

    g_bag_path = './bags/sub_01/obst_gaze_key_2021-01-18-13-18-36.bag'
    k_bag_path = './bags/sub_01/obst_only_key_2021-01-18-13-15-40.bag'
    v_bag_path = './bags/sub_01/obst_virtual_key_2021-01-18-13-21-57.bag'

    # g_bag_path = './bags/sub_02/obst_gaze_key_2021-01-20-13-51-25.bag'
    # v_bag_path = './bags/sub_02/obst_virtual_key_2021-01-20-13-46-48.bag'

    # g_bag_path = './bags/sub_05/obst_gaze_key_2021-01-18-13-46-47.bag'
    # v_bag_path = './bags/sub_05/obst_virtual_key_2021-01-18-13-50-16.bag'

    g_bag = rosbag.Bag(g_bag_path)
    g_cmds, g_poses, g_gazes = get_cmd_pose_gaze(g_bag)
    g_t, g_d = get_dist(g_poses)
    v_bag = rosbag.Bag(v_bag_path)
    v_cmds, v_poses, v_gazes = get_cmd_pose_gaze(v_bag)
    v_t, v_d = get_dist(v_poses)

    k_bag = rosbag.Bag(k_bag_path)
    k_cmds, k_poses, k_gazes = get_cmd_pose_gaze(k_bag)
    k_t, k_d = get_dist(k_poses)

    plot = plt.figure()
    plt.plot(g_t, g_d, color='blue', label='Gaze Key')
    plt.plot(v_t, v_d, color='red', label='Virtual Key')
    plt.plot(k_t, k_d, color='green', label='Only Key')
    plt.legend()
    plt.xlabel('time (s)')
    plt.ylabel('distance to target (m)')
    ax = plt.gca()
    plt.show()


    g_img = draw_gaze1(g_img_path, g_gazes)
    cv2.imshow('gaze key', g_img)

    v_img = draw_gaze2(v_img_path, v_gazes)
    cv2.imshow('virturl key', v_img)
    cv2.imwrite('./gaze_key_gaze.png', g_img)
    cv2.imwrite('./virt_key_gaze.png', v_img)
    cv2.waitKey(0)





