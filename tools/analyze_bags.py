import sys, os
import matplotlib.pyplot as plt
import numpy as np
import rosbag
import cv2
import pandas as pd


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

    bag = rosbag.Bag(bag_path)
    bag = rosbag.Bag('/data/ros/bags/obst_only_key_2021-01-19-14-43-39.bag')
    x = []
    ts = []
    for topic, msg, t in bag.read_messages(topics=['/gaze_to_camera']):
        # print(msg)
        x.append(msg)
        ts.append(t)

        # if abs(msg.x) > 16:
        #     continue

        # x.append(msg.x)
        # y.append(msg.y)
    print(len(x))
    print(ts[-1] - ts[0])
    break


