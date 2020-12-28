import matplotlib.pyplot as plt
import numpy as np
import rosbag


axis_detour = [9.0, 9.0]
# bag = rosbag.Bag('bags/detour_gaze_key_2020-12-28-11-53-49.bag')
bag = rosbag.Bag('bags/detour_virtual_key_2020-12-28-12-08-37.bag')

x = []
y = []
skip_num = 1000
skip = skip_num
for topic, msg, t in bag.read_messages(topics=['/gazebo/link_states']):
   
    if skip != 0:
        skip -= 1
        continue
    skip = skip_num
    print(msg)
    ind = msg.name.index('mybot::chassis')
    pose = msg.pose[ind].position

    x.append(pose.x)
    y.append(pose.y)

plot = plt.figure()
plt.plot(x, y)

plt.xlim([0, 9.0])
plt.ylim([0, 9.0])

ax = plt.gca()
ax.set_aspect(1)
plt.show()


# # cmd_vel
# x = []
# y = []
# for topic, msg, t in bag.read_messages(topics=['/cmd_vel']):
#     print(msg)
#     # print(t)
#     # ind = msg.name.index('mybot::chassis')
#     # pose = msg.pose[ind].position

#     # x.append(pose.x)
#     # y.append(pose.y)



# gaze_to_camera
x = []
y = []
for topic, msg, t in bag.read_messages(topics=['/gaze_to_camera']):
    print(msg)

    x.append(msg.x)
    y.append(msg.y)

plot = plt.figure()
plt.plot(x, y)

# plt.xlim([-18.0, 18.0])
# plt.ylim([0, 18.0])

ax = plt.gca()
ax.set_aspect(1)
ax.xaxis.set_ticks_position('top') 
ax.invert_yaxis()  
plt.show()



# close bag
bag.close()



