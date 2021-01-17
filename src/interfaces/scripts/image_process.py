#!/usr/bin/env python
import rospy
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

bridge = CvBridge()

# cv_image = bridge.imgmsg_to_cv2(image_message, desired_encoding='passthrough')
# image_message = bridge.cv2_to_imgmsg(cv_image, encoding="passthrough")

pub = rospy.Publisher('//mybot/camera1/image_pro', Image)

def callback(data):
    # rospy.loginfo("img %d %d %d", data.height, data.width, len(data.data))
    

    cv_image = bridge.imgmsg_to_cv2(data, desired_encoding='bgr8')

    h_n = int(1.0 * data.width / 1920 * 1080)

    start_h = int((data.height - h_n) / 2)

    cv_image = cv_image[start_h:start_h+h_n, :, :]

    res_image = cv2.resize(cv_image, (1920, 1080))

    

    msg_data = bridge.cv2_to_imgmsg(res_image, encoding="bgr8")

    pub.publish(msg_data)
    
def listener():

    # In ROS, nodes are uniquely named. If two nodes with the same
    # name are launched, the previous one is kicked off. The
    # anonymous=True flag means that rospy will choose a unique
    # name for our 'listener' node so that multiple listeners can
    # run simultaneously.
    rospy.init_node('listener', anonymous=True)
    rospy.Subscriber("/mybot/camera1/image_raw", Image, callback)
    

    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()

if __name__ == '__main__':
    listener()