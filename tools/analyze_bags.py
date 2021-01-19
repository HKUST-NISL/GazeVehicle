import sys, os
import matplotlib.pyplot as plt
import numpy as np
import rosbag
import cv2
import pandas as pd

from enum import Enum, unique

@unique
class Motion(Enum):
    still = 0 
    forward = 1
    backward = 2
    left = 3
    right = 4


def motion_type(msg):
    if msg.linear.x > 0:
        return Motion.forward
    if msg.linear.x < 0:
        return Motion.backward
    if msg.angular.z < 0:
        return Motion.left
    if msg.angular.z > 0:
        return Motion.right
    return Motion.still


if __name__ == "__main__":
        
    root_dir = './bags/'
    bag_list = './bags/bag.lst'

    with open(bag_list) as f:
        lines = f.read().splitlines()

    for line in lines:
        
        bag_path = os.path.join(root_dir, line)
        dir_name, bag_name = os.path.split(bag_path)
        dir_splits = dir_name.split('/')
        bag_splits = bag_name.split('_')

        env_name = bag_splits[0]
        int_name = bag_splits[1] + '_' + bag_splits[2]
        sub_name = dir_splits[-1]

        print(sub_name, env_name, int_name)

        # bag = rosbag.Bag(bag_path)
        bag = rosbag.Bag('/data/ros/bags/hongjiang/detour_gaze_key_2021-01-18-13-38-11.bag')
        
        cmds = []
        gazes = []
        poses = []
        t_start = 0
        t_end = 0
        Moving = False
        for topic, msg, t in bag.read_messages(topics=['/cmd_vel', '/gaze_to_camera', '/vehicle_pose']):

            if not Moving:
                # remove the begining stills
                if topic == '/cmd_vel':
                    if motion_type(msg) == Motion.still:
                        continue
                    else:
                        Moving = True
            if Moving:
                if topic == '/cmd_vel':
                    cmds.append((t, msg))
                if topic == '/gaze_to_camera':
                    gazes.append((t, msg))
                if topic == '/vehicle_pose':
                    poses.append((t, msg))

        # remove the last stills 
        for ind, cmd in enumerate(cmds[::-1]):
            if motion_type(cmd[1]) != Motion.still:
                cmds = cmds[:len(cmds)-1-ind]
                break 

        for ind, gaze in enumerate(gazes[::-1]):
            if gaze[0] < cmds[-1][0]:
                gazes = gazes[:len(gazes)-1-ind]
                break

        for ind, pose in enumerate(poses[::-1]):
            if pose[0] < cmds[-1][0]:
                poses = poses[:len(poses)-1-ind]
                break
    
        print(len(cmds))
        print(len(gazes))
        print(len(poses))
        print(cmds[-1][0] - cmds[0][0])
        print(gazes[-1][0] - gazes[0][0])
        print(poses[-1][0] - poses[0][0])
        break


