#!/usr/bin/python3

import rospy
import math
import numpy as np

from geometry_msgs.msg import PoseStamped
import tf.transformations as t

class HumanController():
    def __init__(self):
        rospy.init_node("human_teleop")

        # self.pose = np.zeros(3)                # [x, y, th]

        self.width = rospy.get_param("~area_width", 19)
        self.humans_num = rospy.get_param("~humans_num", 3)
        self.seed = rospy.get_param("~seed", 42)
        self.vel = rospy.get_param("~vel", 0.025)
        self.pubs = []
        np.random.seed(self.seed)
        for i in range(self.humans_num):
            self.pubs.append(rospy.Publisher(f"/human{i}/actor_pose", PoseStamped, queue_size=10))
        self.poses = -0.5*self.width + self.width * np.random.rand(self.humans_num, 3)
        self.poses[:, 2] = 2 * np.pi * np.random.rand(self.humans_num) 

        self.timer = rospy.Timer(
            rospy.Duration(0.1),
            self.timer_cb
        )

    def timer_cb(self, e):
        for i in range(self.humans_num):
            x_prev, y_prev, theta_prev = self.poses[i]
            xn = x_prev + self.vel * np.cos(theta_prev) #+ np.random.normal(0, 0.1)
            yn = y_prev + self.vel * np.sin(theta_prev) #+ np.random.normal(0, 0.1)
            theta_new = theta_prev
            if xn < -0.5*self.width or xn > 0.5*self.width:
                xn = np.clip(xn, -0.5*self.width, 0.5*self.width)
                theta_new = np.pi - theta_prev
            if yn < -0.5*self.width or yn > 0.5*self.width:
                yn = np.clip(yn, -0.5*self.width, 0.5*self.width)
                theta_new = -theta_new
            
            if -0.5*self.width < xn < 0.5*self.width and -0.5*self.width < yn < 0.5*self.width:  
                theta_new += np.random.normal(0, 0.1)

            self.poses[i] = np.array([xn, yn, theta_new])

            # publish pose
            pose_msg = PoseStamped()
            pose_msg.header.frame_id = "odom"
            pose_msg.header.stamp = rospy.Time.now()
            pose_msg.pose.position.x = self.poses[i, 0]
            pose_msg.pose.position.y = self.poses[i, 1]
            quat = t.quaternion_from_euler(0, 0, self.poses[i, 2]+np.pi/2)
            pose_msg.pose.orientation.x = quat[0]
            pose_msg.pose.orientation.y = quat[1]
            pose_msg.pose.orientation.z = quat[2]
            pose_msg.pose.orientation.w = quat[3]
            self.pubs[i].publish(pose_msg)


if __name__ == "__main__":
    node = HumanController()
    rospy.spin()



