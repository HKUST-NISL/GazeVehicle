import matplotlib.pyplot as plt
import numpy as np
import rosbag


axis_detour = [9.0, 9.0]
bag = rosbag.Bag('./bags/detour_only_key_2020-12-17-22-45-20.bag')

x = []
y = []

skip_num = 1000
skip = skip_num
for topic, msg, t in bag.read_messages(topics=['/gazebo/link_states']):
    if skip != 0:
        skip -= 1
        continue
    skip = skip_num
    print(t)
    ind = msg.name.index('mybot::chassis')
    pose = msg.pose[ind].position

    x.append(pose.x)
    y.append(pose.y)

bag.close()


plot = plt.figure()
plt.plot(x, y)

plt.xlim([0, 9.0])
plt.ylim([0, 9.0])

ax = plt.gca()
ax.set_aspect(1)
plt.show()

