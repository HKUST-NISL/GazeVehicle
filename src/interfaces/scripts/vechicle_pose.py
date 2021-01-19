#!/usr/bin/env python
import rospy
from std_msgs.msg import String
from geometry_msgs.msg import Pose
from gazebo_msgs.msg import LinkStates


rospy.init_node('listener', anonymous=True)
pub = rospy.Publisher('/vehicle_pose', Pose)
rate = rospy.Rate(100) # 100hz

def callback(msg):
    # rospy.loginfo("img %d %d %d", data.height, data.width, len(data.data))
    
    ind = msg.name.index('mybot::chassis')
    pose = msg.pose[ind]
    pub.publish(pose)
    rate.sleep()
    
def listener():

    # In ROS, nodes are uniquely named. If two nodes with the same
    # name are launched, the previous one is kicked off. The
    # anonymous=True flag means that rospy will choose a unique
    # name for our 'listener' node so that multiple listeners can
    # run simultaneously.
    
    rospy.Subscriber("/gazebo/link_states", LinkStates, callback)
    

    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()

if __name__ == '__main__':
    listener()