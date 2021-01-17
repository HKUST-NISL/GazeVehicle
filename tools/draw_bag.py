import matplotlib.pyplot as plt
import numpy as np
import rosbag
import cv2


img_path = '/home/liang/Documents/Academic/GazeVehicle/detour_0.png'
axis_detour = [9.0, 9.0]
bag = rosbag.Bag('bags/detour_gaze_key_2020-12-28-11-53-49.bag')
# bag = rosbag.Bag('bags/detour_gaze_key_2021-01-06-20-16-13.bag')
bag1 = rosbag.Bag('bags/detour_virtual_key_2020-12-28-12-39-52.bag')
# bag = rosbag.Bag('bags/detour_virtual_key_2021-01-06-16-25-59.bag')

# x = []
# y = []
# skip_num = 1000
# skip = skip_num
# t_list = []
# for topic, msg, t in bag.read_messages(topics=['/gazebo/link_states']):
   
#     if skip != 0:
#         skip -= 1
#         continue
#     skip = skip_num
#     ind = msg.name.index('mybot::chassis')
#     pose = msg.pose[ind].position

#     x.append(pose.x)
#     y.append(pose.y)
#     t_list.append(t.to_nsec())

# print(t_list[0], t_list[-1], t_list[-1] - t_list[0])

# plot = plt.figure()
# plt.plot(x, y)
# plt.xlim([0, 9.0])
# plt.ylim([0, 9.0])
# ax = plt.gca()
# ax.set_aspect(1)
# plt.show()

# z = []
# start_i = 0
# for i in range(len(x)):
#     z.append(np.sqrt((x[i]-x[-1])**2 + (y[i]-y[-1])**2))

#     if i > 0 and start_i == 0 and z[i]-z[i-1]>0.00001:
#         start_i = i

# tt = (np.array(t_list).astype(np.float32) - t_list[start_i]) / 1000000000
# z = list(z[start_i:])
# tt = list(tt[start_i:])

# x1 = []
# y1 = []
# skip_num = 1000
# skip = skip_num
# t_list1 = []
# for topic, msg, t in bag1.read_messages(topics=['/gazebo/link_states']):
   
#     if skip != 0:
#         skip -= 1
#         continue
#     skip = skip_num
#     # print(msg)
#     ind = msg.name.index('mybot::chassis')
#     pose = msg.pose[ind].position

#     x1.append(pose.x)
#     y1.append(pose.y)
#     t_list1.append(t.to_nsec())

# print(t_list1[0], t_list1[-1], t_list1[-1] - t_list1[0])

# plot = plt.figure()
# plt.plot(x1, y1)
# plt.xlim([0, 9.0])
# plt.ylim([0, 9.0])
# ax = plt.gca()
# ax.set_aspect(1)
# plt.show()

# z1 = []
# start_i = 0
# for i in range(len(x1)):
#     z1.append(np.sqrt((x1[i]-x1[-1])**2 + (y1[i]-y1[-1])**2))

#     if i > 0 and start_i == 0 and z1[i]-z1[i-1]>0.00001:
#         start_i = i

# tt1 = (np.array(t_list1).astype(np.float32) - t_list1[start_i]) / 1000000000
# z1 = list(z1[start_i:])
# tt1 = list(tt1[start_i:])


# plot = plt.figure()
# plt.plot(tt, z, color='blue', label='Gaze Key')
# plt.plot(tt1, z1, color='red', label='Virtual Key')
# plt.legend()
# plt.xlabel('time (s)')
# plt.ylabel('distance to target (m)')
# ax = plt.gca()
# plt.show()


# img = cv2.imread(img_path)


# def draw_line(img, x, y, color):
#     h, w, _ = img.shape
#     px = []
#     py = []
#     for i in range(len(x)):
#         px.append(int(1.0*(x[i]) * h / 9))
#         py.append(int(1.0*(9-y[i]) * w / 9))

#         if i > 0:
#             cv2.line(img, (px[i-1], py[i-1]), (px[i], py[i]), color, 2)

# draw_line(img, x, y, color=(255, 0, 0))
# draw_line(img, x1, y1, color=(0, 0, 255))

# cv2.imshow('img', img)
# cv2.imwrite('./trajs.png', img)
# cv2.waitKey(0)


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
for topic, msg, t in bag1.read_messages(topics=['/gaze_to_camera']):
    # print(msg)

    if abs(msg.x) > 16:
        continue

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

# img_path = '/home/liang/Documents/Academic/GazeVehicle/gaze_key_screen.png'
img_path = '/home/liang/Documents/Academic/GazeVehicle/virt_key_interface1.png'
img = cv2.imread(img_path)

def draw_gaze_line(img, x, y, color=(0,0,255)):
    h, w, _ = img.shape
    px = []
    py = []
    for i in range(len(x)):

        # if x[i] > -5:
        #     y[i] -= 1
        # if x[i] > 5:
        #     x[i] += 2

        if y[i] < 0:
            y[i] -= 5 * (-y[i] / 5)
        y[i] += 17.5
            
        py.append(int(1.0* y[i] * h / 36))
        # py.append(int(1.0* y[i] * h / 17.5))
        px.append(int(1.0* x[i] * w / 34) + w/2)

        if len(px) > 1:
            cv2.line(img, (px[-2], py[-2]), (px[-1], py[-1]), color, 2)

draw_gaze_line(img, x, y, (255, 0, 0) )
cv2.imshow('img', img)
cv2.imwrite('virt_key_traj_gaze.png', img)
# cv2.imwrite('gaze_key_traj_gaze.png', img)
cv2.waitKey(0)

# close bag
bag.close()



