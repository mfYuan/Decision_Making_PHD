#!/usr/bin/env python

import os
import numpy as np
from math import cos, sin

from geometry_msgs.msg import PoseStamped

import tf
import rospy


class collision_node:
    def __init__(self):
        rospy.init_node('collision_node')
        # Initialize variables
        (
            self.c1_x,
            self.c1_y,
            self.c2_x,
            self.c2_y,
            self.c3_x,
            self.c3_y,
            self.c4_x,
            self.c4_y,
        ) = [0] * 8
        
        rospy.Subscriber('/qcar1/center_pose', PoseStamped,
                         self.center1, queue_size=1)
        rospy.Subscriber('/qcar2/center_pose', PoseStamped,
                         self.center2, queue_size=1)
        rospy.Subscriber('/qcar3/center_pose', PoseStamped,
                         self.center3, queue_size=1)
        rospy.Subscriber('/qcar4/center_pose', PoseStamped,
                         self.center4, queue_size=1)

        self.qcar1_distance_pub = rospy.Publisher(
            '/qcar1/distances', PoseStamped, queue_size=1)

        self.qcar2_distance_pub = rospy.Publisher(
            '/qcar2/distances', PoseStamped, queue_size=1)

        self.qcar3_distance_pub = rospy.Publisher(
            '/qcar3/distances', PoseStamped, queue_size=1)

        self.qcar4_distance_pub = rospy.Publisher(
            '/qcar4/distances', PoseStamped, queue_size=1)
        
        self.p1 = [-1.04, 0.18]
        self.p2 = [-0.22, 0.18]   # [-0.22, -0.21]
        self.p3 = [-0.22, -0.21]  # [0.18, 0.18]
        self.p4 = [0.18, 0.18]   # [0.18, -0.18]
        self.p5 = [0.18, -0.18]  # [1.04, -0.18]
        self.p6 = [1.04, -0.18]
        
        while True:
            self.spin()
            
    def center1(self, data):
        self.c1_x = data.pose.position.x
        self.c1_y = data.pose.position.y

    def center2(self, data):
        self.c2_x = data.pose.position.x
        self.c2_y = data.pose.position.y

    def center3(self, data):
        self.c3_x = data.pose.position.x
        self.c3_y = data.pose.position.y

    def center4(self, data):
        self.c4_x = data.pose.position.x
        self.c4_y = data.pose.position.y

    def spin(self):
        # make arrays
        qcar1 = [self.c1_x, self.c1_y]
        qcar2 = [self.c2_x, self.c2_y]
        qcar3 = [self.c3_x, self.c3_y]
        qcar4 = [self.c4_x, self.c4_y]

       	qcar1_position = [self.p1, self.p3, self.p4, self.p5, self.p6]
        qcar1_dist = []
        for x,y in qcar1_position:
            x_pos = qcar1[1][0]-x
            y_pos = qcar1[1][1]-y
            qcar1_dist.append(np.sqrt(x_pos**2 + y_pos**2))
            
        qcar1_distances = PoseStamped()
        qcar1_distances.pose.position.x = qcar1_dist[0]
        qcar1_distances.pose.position.y = qcar1_dist[1]
        qcar1_distances.pose.position.z = qcar1_dist[2]
        qcar1_distances.pose.orientation.x = qcar1_dist[3]
        qcar1_distances.pose.orientation.y = qcar1_dist[4]
        
        self.qcar1_distance_pub.publish(qcar1_distances)
            
        qcar2_position = [self.p3, self.p5, self.p6]
        qcar2_dist = []

        for x, y in qcar2_position:
            x_pos = qcar2[0]-x
            y_pos = qcar2[1]-y
            qcar2_dist.append(np.sqrt(x_pos**2 + y_pos**2))

        qcar2_distances = PoseStamped()
        qcar2_distances.pose.position.x = qcar2_dist[0]
        qcar2_distances.pose.position.y = qcar2_dist[1]
        qcar2_distances.pose.position.z = qcar2_dist[2]

        self.qcar2_distance_pub.publish(qcar2_distances)
        
        qcar3_position = [self.p1, self.p2, self.p4]
        qcar3_dist = []

        for x, y in qcar3_position:
            x_pos = qcar3[0]-x
            y_pos = qcar3[1]-y
            qcar3_dist.append(np.sqrt(x_pos**2 + y_pos**2))

        qcar3_distances = PoseStamped()
        qcar3_distances.pose.position.x = qcar3_dist[0]
        qcar3_distances.pose.position.y = qcar3_dist[1]
        qcar3_distances.pose.position.z = qcar3_dist[2]

        self.qcar3_distance_pub.publish(qcar3_distances)
        
        qcar4_position = [self.p2, self.p3]
        qcar4_dist = []

        for x, y in qcar4_position:
            x_pos = qcar4[0]-x
            y_pos = qcar4[1]-y
            qcar4_dist.append(np.sqrt(x_pos**2 + y_pos**2))

        qcar4_distances = PoseStamped()
        qcar4_distances.pose.position.x = qcar4_dist[0]
        qcar4_distances.pose.position.y = qcar4_dist[1]

        self.qcar4_distance_pub.publish(qcar4_distances)


if __name__ == "__main__":
    try:
        collision_node()
    except:
        rospy.logwarn("cannot start collision node")

