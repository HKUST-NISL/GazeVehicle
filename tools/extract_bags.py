import sys, os
import matplotlib.pyplot as plt
import numpy as np
import rosbag
import cv2
import pandas as pd

from geometry_msgs.msg import Pose
from gazebo_msgs.msg import LinkStates


root_dir = './bags/'
bag_list = './bags/bag.lst'
out_dir = './bags/new_bags/'

with open(bag_list) as f:
    lines = f.read().splitlines()

for i, line in enumerate(lines):
    print(i+1, line)
    bag_path = os.path.join(root_dir, line)
    dir_name, bag_name = os.path.split(bag_path)
    dir_splits = dir_name.split('/')
    bag_splits = bag_name.split('_')

    env_name = bag_splits[0]
    int_name = bag_splits[1] + '_' + bag_splits[2]
    sub_name = dir_splits[-1]

    out_bag_dir = os.path.join(out_dir, sub_name)
    if not os.path.exists(out_bag_dir):
        os.makedirs(out_bag_dir)

    out_bag_path = os.path.join(out_bag_dir, bag_name)

    # print(sub_name, env_name, int_name)

    out_bag = rosbag.Bag(out_bag_path, 'w')
    out_bag.reindex()
    bag = rosbag.Bag(bag_path)

    pose_c = 0
    for topic, msg, t in bag.read_messages(topics=['/cmd_vel', '/gaze_to_camera', '/gazebo/link_states']):
        
        if topic == '/gazebo/link_states':
            
            if pose_c % 100 == 0:
                ind = msg.name.index('mybot::chassis')
                pose = msg.pose[ind]
                out_bag.write('/vehicle_pose', pose, t)
            pose_c += 1

        else:
            out_bag.write(topic, msg, t)

    out_bag.close()


