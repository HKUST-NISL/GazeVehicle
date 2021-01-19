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
    if msg.angular.z > 0:
        return Motion.left
    if msg.angular.z < 0:
        return Motion.right
    return Motion.still

def analyze_motion_time(cmds):
    t_motions = np.zeros(5)
    
    last_m = motion_type(cmds[0][1])
    last_t = cmds[0][0]
    for i in range(1, len(cmds)):
        cur_m = motion_type(cmds[i][1])
        cur_t = cmds[i][0]

        t_motions[cur_m.value] += cur_t.to_nsec() - last_t.to_nsec()

        last_m = cur_m
        last_t = cur_t
    
    return t_motions / 1e+9

def analyze_pose(poses):

    dist_motion = 0
    last_p = poses[0][1].position
    
    for i in range(1, len(poses)):
        cur_p = poses[i][1].position

        dist_motion += np.sqrt((cur_p.x - last_p.x)**2 + (cur_p.y - last_p.y)**2)

        last_p = cur_p
    
    return dist_motion * 5

def analyze_gaze(gazes):
    dist_gaze = 0
    last_p = gazes[0][1]
    
    for i in range(1, len(gazes)):
        cur_p = gazes[i][1]

        dist_gaze += np.sqrt((cur_p.x - last_p.x)**2 + (cur_p.y - last_p.y)**2)

        last_p = cur_p
    
    return dist_gaze




if __name__ == "__main__":
        
    root_dir = './bags/'
    bag_list = './bags/bag.lst'

    with open(bag_list) as f:
        lines = f.read().splitlines()

    df = pd.DataFrame()

    for line in lines:
        
        bag_path = os.path.join(root_dir, line)
        dir_name, bag_name = os.path.split(bag_path)
        dir_splits = dir_name.split('/')
        bag_splits = bag_name.split('_')

        env_name = bag_splits[0]
        int_name = bag_splits[1] + '_' + bag_splits[2]
        sub_name = dir_splits[-1]

        bag = rosbag.Bag(bag_path)
        # bag = rosbag.Bag('/data/ros/bags/hongjiang/detour_gaze_key_2021-01-18-13-38-11.bag')
        
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
    

        t_tol = (cmds[-1][0] - cmds[0][0]).to_nsec() / 1e+9

        t_motions = analyze_motion_time(cmds)
        t_still = t_motions[0]
        t_linear = t_motions[1] + t_motions[2]
        t_angler = t_motions[3] + t_motions[4]
        dist_motion = analyze_pose(poses)
        vel_motion = dist_motion / t_tol
        dis_gaze = analyze_gaze(gazes)
        # print(t_tol, t_still, t_linear, t_angler, dist_motion, vel_motion, dis_gaze)

        new = pd.DataFrame({'env': env_name,
                        'interface': int_name,
                        'subject': sub_name,
                        't_total': t_tol, 
                        't_still': t_still,
                        't_linear':t_linear,
                        't_angler': t_angler,
                        'diststance': dist_motion,
                        'speed_avg': vel_motion,
                        'gaze distance': dis_gaze}, index=[1])
                
        df = df.append(new, ignore_index=True) 
        
        # break
    cols=['env', 'interface', 'subject', 't_total', 't_still', 't_linear', 't_angler', 'diststance', 'speed_avg', 'gaze distance']
    df=df[cols]
    # df.set_index(['env', 'interface', 'subject'])
    print(df)
