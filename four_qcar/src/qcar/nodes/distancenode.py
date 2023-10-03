#!/usr/bin/env python

import os
import numpy as np
from math import cos, sin

from geometry_msgs.msg import PoseStamped, TwistStamped

import tf
import rospy


class distance_node:
    def __init__(self):
        rospy.init_node('distance_node')
        # Initialize variables
        (
            self.r1_x,
            self.r1_y,
            self.c1_x,
            self.c1_y,
            self.f1_x,
            self.f1_y,
            self.r2_x,
            self.r2_y,
            self.c2_x,
            self.c2_y,
            self.f2_x,
            self.f2_y,
            self.r3_x,
            self.r3_y,
            self.c3_x,
            self.c3_y,
            self.f3_x,
            self.f3_y,
            self.r4_x,
            self.r4_y,
            self.c4_x,
            self.c4_y,
            self.f4_x,
            self.f4_y,
        ) = [0] * 24
        rospy.Subscriber('/qcar1/rear_pose', PoseStamped,
                         self.rear1, queue_size=1)
        rospy.Subscriber('/qcar1/center_pose', PoseStamped,
                         self.center1, queue_size=1)
        rospy.Subscriber('/qcar1/front_pose', PoseStamped,
                         self.front1, queue_size=1)
        rospy.Subscriber('/qcar2/rear_pose', PoseStamped,
                         self.rear2, queue_size=1)
        rospy.Subscriber('/qcar2/center_pose', PoseStamped,
                         self.center2, queue_size=1)
        rospy.Subscriber('/qcar2/front_pose', PoseStamped,
                         self.front2, queue_size=1)
        rospy.Subscriber('/qcar3/rear_pose', PoseStamped,
                         self.rear3, queue_size=1)
        rospy.Subscriber('/qcar3/center_pose', PoseStamped,
                         self.center3, queue_size=1)
        rospy.Subscriber('/qcar3/front_pose', PoseStamped,
                         self.front3, queue_size=1)
        rospy.Subscriber('/qcar4/rear_pose', PoseStamped,
                         self.rear4, queue_size=1)
        rospy.Subscriber('/qcar4/center_pose', PoseStamped,
                         self.center4, queue_size=1)
        rospy.Subscriber('/qcar4/front_pose', PoseStamped,
                         self.front4, queue_size=1)

        self.distance_pub = rospy.Publisher(
            '/distances', PoseStamped, queue_size=1)

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

    def rear1(self, data):
        self.r1_x = data.pose.position.x
        self.r1_y = data.pose.position.y

    def center1(self, data):
        self.c1_x = data.pose.position.x
        self.c1_y = data.pose.position.y

    def front1(self, data):
        self.f1_x = data.pose.position.x
        self.f1_y = data.pose.position.y

    def rear2(self, data):
        self.r2_x = data.pose.position.x
        self.r2_y = data.pose.position.y

    def center2(self, data):
        self.c2_x = data.pose.position.x
        self.c2_y = data.pose.position.y

    def front2(self, data):
        self.f2_x = data.pose.position.x
        self.f2_y = data.pose.position.y

    def rear3(self, data):
        self.r3_x = data.pose.position.x
        self.r3_y = data.pose.position.y

    def center3(self, data):
        self.c3_x = data.pose.position.x
        self.c3_y = data.pose.position.y

    def front3(self, data):
        self.f3_x = data.pose.position.x
        self.f3_y = data.pose.position.y

    def rear4(self, data):
        self.r4_x = data.pose.position.x
        self.r4_y = data.pose.position.y

    def center4(self, data):
        self.c4_x = data.pose.position.x
        self.c4_y = data.pose.position.y

    def front4(self, data):
        self.f4_x = data.pose.position.x
        self.f4_y = data.pose.position.y

    def spin(self):
        # make arrays
        qcar1 = np.array(
            [[self.f1_x, self.f1_y], [self.c1_x, self.c1_y], [self.r1_x, self.r1_y]])
        qcar2 = np.array(
            [[self.f2_x, self.f2_y], [self.c2_x, self.c2_y], [self.r2_x, self.r2_y]])
        qcar3 = np.array(
            [[self.f3_x, self.f3_y], [self.c3_x, self.c3_y], [self.r3_x, self.r3_y]])
        qcar4 = np.array(
            [[self.f4_x, self.f4_y], [self.c4_x, self.c4_y], [self.r4_x, self.r4_y]])

        # qcar1 and qcar2
        self.d_1_2 = 10000000
        for i in range(3):
            for j in range(3):
                x = np.sqrt((qcar1[i][0] - qcar2[j][0])**2 +
                            (qcar1[i][1] - qcar2[j][1])**2)
                if x < self.d_1_2:
                    self.d_1_2 = x

        # qcar1 and qcar3
        self.d_1_3 = 100000000
        for i in range(3):
            for j in range(3):
                x = np.sqrt((qcar1[i][0] - qcar3[j][0])**2 +
                            (qcar1[i][1] - qcar3[j][1])**2)
                if x < self.d_1_3:
                    self.d_1_3 = x

        # qcar2 and qcar3
        self.d_2_3 = 100000000
        for i in range(3):
            for j in range(3):
                x = np.sqrt((qcar2[i][0] - qcar3[j][0])**2 +
                            (qcar2[i][1] - qcar3[j][1])**2)
                if x < self.d_2_3:
                    self.d_2_3 = x

        # qcar1 and qcar4
        self.d_1_4 = 100000000
        for i in range(3):
            for j in range(3):
                x = np.sqrt((qcar1[i][0] - qcar4[j][0])**2 +
                            (qcar1[i][1] - qcar4[j][1])**2)

                if x < self.d_1_4:
                    self.d_1_4 = x

        # qcar2 and qcar4
        self.d_2_4 = 100000000
        for i in range(3):
            for j in range(3):
                x = np.sqrt((qcar2[i][0] - qcar4[j][0])**2 +
                            (qcar2[i][1] - qcar4[j][1])**2)
                if x < self.d_2_4:
                    self.d_2_4 = x

        # qcar3 and qcar4
        self.d_3_4 = 100000000
        for i in range(3):
            for j in range(3):
                x = np.sqrt((qcar3[i][0] - qcar4[j][0])**2 +
                            (qcar3[i][1] - qcar4[j][1])**2)
                if x < self.d_3_4:
                    self.d_3_4 = x

        distances = PoseStamped()
        distances.pose.position.x = np.round(self.d_1_2, 4)
        distances.pose.position.y = np.round(self.d_1_3, 4)
        distances.pose.position.z = np.round(self.d_1_4, 4)
        distances.pose.orientation.x = np.round(self.d_2_3, 4)
        distances.pose.orientation.y = np.round(self.d_2_4, 4)
        distances.pose.orientation.z = np.round(self.d_3_4, 4)
        self.distance_pub.publish(distances)

        qcar1_position = [self.p1, self.p3, self.p4, self.p5, self.p6]
        qcar1_dist = []
        for x, y in qcar1_position:
            x_pos = qcar1[1][0]-x
            y_pos = qcar1[1][1]-y
            qcar1_dist.append(np.sqrt(x_pos**2 + y_pos**2))

        qcar1_distances = PoseStamped()
        qcar1_distances.pose.position.x = np.round(qcar1_dist[0], 4)
        qcar1_distances.pose.position.y = np.round(qcar1_dist[1], 4)
        qcar1_distances.pose.position.z = np.round(qcar1_dist[2], 4)
        qcar1_distances.pose.orientation.x = np.round(qcar1_dist[3], 4)
        qcar1_distances.pose.orientation.y = np.round(qcar1_dist[4], 4)

        self.qcar1_distance_pub.publish(qcar1_distances)

        qcar2_position = [self.p3, self.p5, self.p6]
        qcar2_dist = []

        for x, y in qcar2_position:
            x_pos = qcar2[1][0]-x
            y_pos = qcar2[1][1]-y
            qcar2_dist.append(np.sqrt(x_pos**2 + y_pos**2))

        qcar2_distances = PoseStamped()
        qcar2_distances.pose.position.x = np.round(qcar2_dist[0], 4)
        qcar2_distances.pose.position.y = np.round(qcar2_dist[1], 4)
        qcar2_distances.pose.position.z = np.round(qcar2_dist[2], 4)

        self.qcar2_distance_pub.publish(qcar2_distances)

        qcar3_position = [self.p1, self.p2, self.p4]
        qcar3_dist = []

        for x, y in qcar3_position:
            x_pos = qcar3[1][0]-x
            y_pos = qcar3[1][1]-y
            qcar3_dist.append(np.sqrt(x_pos**2 + y_pos**2))

        qcar3_distances = PoseStamped()
        qcar3_distances.pose.position.x = np.round(qcar3_dist[0], 4)
        qcar3_distances.pose.position.y = np.round(qcar3_dist[1], 4)
        qcar3_distances.pose.position.z = np.round(qcar3_dist[2], 4)

        self.qcar3_distance_pub.publish(qcar3_distances)

        qcar4_position = [self.p2, self.p3]
        qcar4_dist = []

        for x, y in qcar4_position:
            x_pos = qcar4[1][0]-x
            y_pos = qcar4[1][1]-y
            qcar4_dist.append(np.sqrt(x_pos**2 + y_pos**2))

        qcar4_distances = PoseStamped()
        qcar4_distances.pose.position.x = np.round(qcar4_dist[0], 4)
        qcar4_distances.pose.position.y = np.round(qcar4_dist[1], 4)

        self.qcar4_distance_pub.publish(qcar4_distances)


if __name__ == "__main__":
    try:
        distance_node()
    except:
        rospy.logwarn("cannot start distance node")

