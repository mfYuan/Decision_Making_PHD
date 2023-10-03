#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Author: Mingfeng
Date: 27/2021
"""

import rospy
import tf as tf1
import csv
from std_msgs.msg import Float64
from std_msgs.msg import String
from gazebo_msgs.msg import ModelStates
from gazebo_msgs.msg import ModelState
from geometry_msgs.msg import Twist
from geometry_msgs.msg import Pose
from geometry_msgs.msg import Quaternion, PoseStamped, TwistStamped
from gazebo_msgs.srv import SetModelState
from std_srvs.srv import Empty
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped
from styx_msgs.msg import Lane, Waypoint

import matplotlib.pyplot as plt
import os
import shutil
import math
import numpy as np
import threading
import time
import random
# import tensorflow as tf
import datetime
import cv2
from cv_bridge import CvBridge, CvBridgeError

MAXENVSIZE = 30.0  # 边长为30的正方形作为环境的大小
MAXLASERDIS = 3.0  # 雷达最大的探测距离
Image_matrix = []
HORIZON = 0.3


class envmodel():
    def __init__(self):
        '''
        # 保存每次生成的map信息
        self.count_map = 1
        self.foldername_map='map'
        if os.path.exists(self.foldername_map):
            shutil.rmtree(self.foldername_map)
        os.mkdir(self.foldername_map)
        '''

        # agent列表
        self.agentrobot1 = 'qcar1'
        self.agentrobot2 = 'qcar2'
        self.agentrobot3 = 'qcar3'
        self.agentrobot4 = 'qcar4'
        self.agentrobot5 = 'qcar5'
        self.agentrobot6 = 'qcar6'
        self.agentrobot7 = 'qcar7'
        self.agentrobot8 = 'qcar8'
        

        self.img_size = 80

        # 障碍数量
        self.num_obs = 10

        # 位置精度-->判断是否到达目标的距离 (Position accuracy-->Judge whether to reach the target distance)
        self.dis = 1.0

        self.obs_pos = []  # 障碍物的位置信息

        self.gazebo_model_states = ModelStates()

        self.resetval()

        # Subscribers and publishers
        self.sub = rospy.Subscriber(
            '/gazebo/model_states', ModelStates, self.gazebo_states_callback)
        self.pub1 = rospy.Publisher(
            '/' + self.agentrobot1 + '/ackermann_cmd', AckermannDriveStamped, queue_size=10)
        self.pub2 = rospy.Publisher(
            '/' + self.agentrobot2 + '/ackermann_cmd', AckermannDriveStamped, queue_size=10)
        self.pub3 = rospy.Publisher(
            '/' + self.agentrobot3 + '/ackermann_cmd', AckermannDriveStamped, queue_size=10)
        self.pub4 = rospy.Publisher(
            '/' + self.agentrobot4 + '/ackermann_cmd', AckermannDriveStamped, queue_size=10)
        self.pub5 = rospy.Publisher(
            '/' + self.agentrobot5 + '/ackermann_cmd', AckermannDriveStamped, queue_size=10)
        self.pub6 = rospy.Publisher(
            '/' + self.agentrobot6 + '/ackermann_cmd', AckermannDriveStamped, queue_size=10)
        self.pub7 = rospy.Publisher(
            '/' + self.agentrobot7 + '/ackermann_cmd', AckermannDriveStamped, queue_size=10)
        self.pub8 = rospy.Publisher(
            '/' + self.agentrobot8 + '/ackermann_cmd', AckermannDriveStamped, queue_size=10)

        time.sleep(1.0)

    def resetval(self):
        self.robotstate1 = [0.0, 0.0, 0.0, 0.0,
                            0.0, 0.0, 0.0]  # x,y,v,w,yaw,vx,vy
        self.robotstate2 = [0.0, 0.0, 0.0, 0.0,
                            0.0, 0.0, 0.0]  # x,y,v,w,yaw,vx,vy
        self.robotstate3 = [0.0, 0.0, 0.0, 0.0,
                            0.0, 0.0, 0.0]  # x,y,v,w,yaw,vx,vy
        self.robotstate4 = [0.0, 0.0, 0.0, 0.0,
                            0.0, 0.0, 0.0]  # x,y,v,w,yaw,vx,vy
        self.robotstate5 = [0.0, 0.0, 0.0, 0.0,
                            0.0, 0.0, 0.0]  # x,y,v,w,yaw,vx,vy
        self.robotstate6 = [0.0, 0.0, 0.0, 0.0,
                            0.0, 0.0, 0.0]  # x,y,v,w,yaw,vx,vy
        self.robotstate7 = [0.0, 0.0, 0.0, 0.0,
                            0.0, 0.0, 0.0]  # x,y,v,w,yaw,vx,vy
        self.robotstate8 = [0.0, 0.0, 0.0, 0.0,
                            0.0, 0.0, 0.0]  # x,y,v,w,yaw,vx,vy

        self.d = 0.0                                  # distance between qcar1 and qcar2
        self.d_last = 0.0                                  # 前一时刻到目标的距离
        self.v_last = 0.0                                  # 前一时刻的速度
        self.w_last = 0.0                                  # 前一时刻的角速度
        self.r = 0.0                                  # 奖励
        self.cmd = [0.0, 0.0]                           # agent robot的控制指令
        self.done_list = False                                # episode是否结束的标志

    def gazebo_states_callback(self, data):
        self.gazebo_model_states = data
        # name: ['ground_plane', 'jackal1', 'jackal2', 'jackal0',...]
        for i in range(len(data.name)):
            # qcar1
            if data.name[i] == self.agentrobot1:
                # robotstate--->x,y,v,w,yaw,vx,vy
                self.robotstate1[0] = data.pose[i].position.x
                self.robotstate1[1] = data.pose[i].position.y
                v = math.sqrt(data.twist[i].linear.x **
                              2 + data.twist[i].linear.y**2)
                self.robotstate1[2] = v
                self.robotstate1[3] = data.twist[i].angular.z
                rpy = self.euler_from_quaternion(data.pose[i].orientation.x, data.pose[i].orientation.y,
                                                 data.pose[i].orientation.z, data.pose[i].orientation.w)
                self.robotstate1[4] = rpy[2]
                self.robotstate1[5] = data.twist[i].linear.x
                self.robotstate1[6] = data.twist[i].linear.y

            # qcar2
            if data.name[i] == self.agentrobot2:
                # robotstate--->x,y,v,w,yaw,vx,vy
                self.robotstate2[0] = data.pose[i].position.x
                self.robotstate2[1] = data.pose[i].position.y
                v = math.sqrt(data.twist[i].linear.x **
                              2 + data.twist[i].linear.y**2)
                self.robotstate2[2] = v
                self.robotstate2[3] = data.twist[i].angular.z
                rpy = self.euler_from_quaternion(data.pose[i].orientation.x, data.pose[i].orientation.y,
                                                 data.pose[i].orientation.z, data.pose[i].orientation.w)
                self.robotstate2[4] = rpy[2]
                self.robotstate2[5] = data.twist[i].linear.x
                self.robotstate2[6] = data.twist[i].linear.y

            # qcar3
            if data.name[i] == self.agentrobot3:
                # robotstate--->x,y,v,w,yaw,vx,vy
                self.robotstate3[0] = data.pose[i].position.x
                self.robotstate3[1] = data.pose[i].position.y
                v = math.sqrt(data.twist[i].linear.x **
                              2 + data.twist[i].linear.y**2)
                self.robotstate3[2] = v
                self.robotstate3[3] = data.twist[i].angular.z
                rpy = self.euler_from_quaternion(data.pose[i].orientation.x, data.pose[i].orientation.y,
                                                 data.pose[i].orientation.z, data.pose[i].orientation.w)
                self.robotstate3[4] = rpy[2]
                self.robotstate3[5] = data.twist[i].linear.x
                self.robotstate3[6] = data.twist[i].linear.y

            # qcar4
            if data.name[i] == self.agentrobot4:
                # robotstate--->x,y,v,w,yaw,vx,vy
                self.robotstate4[0] = data.pose[i].position.x
                self.robotstate4[1] = data.pose[i].position.y
                v = math.sqrt(data.twist[i].linear.x **
                              2 + data.twist[i].linear.y**2)
                self.robotstate4[2] = v
                self.robotstate4[3] = data.twist[i].angular.z
                rpy = self.euler_from_quaternion(data.pose[i].orientation.x, data.pose[i].orientation.y,
                                                 data.pose[i].orientation.z, data.pose[i].orientation.w)
                self.robotstate4[4] = rpy[2]
                self.robotstate4[5] = data.twist[i].linear.x
                self.robotstate4[6] = data.twist[i].linear.y
            # qcar5
            if data.name[i] == self.agentrobot5:
                # robotstate--->x,y,v,w,yaw,vx,vy
                self.robotstate5[0] = data.pose[i].position.x
                self.robotstate5[1] = data.pose[i].position.y
                v = math.sqrt(data.twist[i].linear.x **
                              2 + data.twist[i].linear.y**2)
                self.robotstate5[2] = v
                self.robotstate5[3] = data.twist[i].angular.z
                rpy = self.euler_from_quaternion(data.pose[i].orientation.x, data.pose[i].orientation.y,
                                                 data.pose[i].orientation.z, data.pose[i].orientation.w)
                self.robotstate5[4] = rpy[2]
                self.robotstate5[5] = data.twist[i].linear.x
                self.robotstate5[6] = data.twist[i].linear.y
            # qcar6
            if data.name[i] == self.agentrobot6:
                # robotstate--->x,y,v,w,yaw,vx,vy
                self.robotstate6[0] = data.pose[i].position.x
                self.robotstate6[1] = data.pose[i].position.y
                v = math.sqrt(data.twist[i].linear.x **
                              2 + data.twist[i].linear.y**2)
                self.robotstate6[2] = v
                self.robotstate6[3] = data.twist[i].angular.z
                rpy = self.euler_from_quaternion(data.pose[i].orientation.x, data.pose[i].orientation.y,
                                                 data.pose[i].orientation.z, data.pose[i].orientation.w)
                self.robotstate6[4] = rpy[2]
                self.robotstate6[5] = data.twist[i].linear.x
                self.robotstate6[6] = data.twist[i].linear.y
            # qcar7
            if data.name[i] == self.agentrobot7:
                # robotstate--->x,y,v,w,yaw,vx,vy
                self.robotstate7[0] = data.pose[i].position.x
                self.robotstate7[1] = data.pose[i].position.y
                v = math.sqrt(data.twist[i].linear.x **
                              2 + data.twist[i].linear.y**2)
                self.robotstate7[2] = v
                self.robotstate7[3] = data.twist[i].angular.z
                rpy = self.euler_from_quaternion(data.pose[i].orientation.x, data.pose[i].orientation.y,
                                                 data.pose[i].orientation.z, data.pose[i].orientation.w)
                self.robotstate7[4] = rpy[2]
                self.robotstate7[5] = data.twist[i].linear.x
                self.robotstate7[6] = data.twist[i].linear.y
            # qcar8
            if data.name[i] == self.agentrobot8:
                # robotstate--->x,y,v,w,yaw,vx,vy
                self.robotstate8[0] = data.pose[i].position.x
                self.robotstate8[1] = data.pose[i].position.y
                v = math.sqrt(data.twist[i].linear.x **
                              2 + data.twist[i].linear.y**2)
                self.robotstate8[2] = v
                self.robotstate8[3] = data.twist[i].angular.z
                rpy = self.euler_from_quaternion(data.pose[i].orientation.x, data.pose[i].orientation.y,
                                                 data.pose[i].orientation.z, data.pose[i].orientation.w)
                self.robotstate8[4] = rpy[2]
                self.robotstate8[5] = data.twist[i].linear.x
                self.robotstate8[6] = data.twist[i].linear.y

    def quaternion_from_euler(self, r, p, y):
        q = [0, 0, 0, 0]
        q[3] = math.cos(r / 2) * math.cos(p / 2) * math.cos(y / 2) + \
            math.sin(r / 2) * math.sin(p / 2) * math.sin(y / 2)
        q[0] = math.sin(r / 2) * math.cos(p / 2) * math.cos(y / 2) - \
            math.cos(r / 2) * math.sin(p / 2) * math.sin(y / 2)
        q[1] = math.cos(r / 2) * math.sin(p / 2) * math.cos(y / 2) + \
            math.sin(r / 2) * math.cos(p / 2) * math.sin(y / 2)
        q[2] = math.cos(r / 2) * math.cos(p / 2) * math.sin(y / 2) - \
            math.sin(r / 2) * math.sin(p / 2) * math.cos(y / 2)
        return q

    def euler_from_quaternion(self, x, y, z, w):
        euler = [0, 0, 0]
        Epsilon = 0.0009765625
        Threshold = 0.5 - Epsilon
        TEST = w * y - x * z
        if TEST < -Threshold or TEST > Threshold:
            if TEST > 0:
                sign = 1
            elif TEST < 0:
                sign = -1
            euler[2] = -2 * sign * math.atan2(x, w)
            euler[1] = sign * (math.pi / 2.0)
            euler[0] = 0
        else:
            euler[0] = math.atan2(2 * (y * z + w * x),
                                  w * w - x * x - y * y + z * z)
            euler[1] = math.asin(-2 * (x * z - w * y))
            euler[2] = math.atan2(2 * (x * y + w * z),
                                  w * w + x * x - y * y - z * z)

        return euler

    def reset_env(self, spawn_ls):
        # '''self.path = path
        #     if path == '5':
        #         self.start1 = [0.18, -1.16, 0.5*np.pi]
        #         self.start2 = [-1.16, -0.18, 0.0]

        #     if path == '6':
        #         self.start1 = [1.16, 0.18, np.pi]
        #         self.start2 = [-0.18, 1.16, 1.5*np.pi]

        #     if path == '7':
        #         self.start1 = [-0.18, 1.16, 1.5*np.pi]
        #         self.start2 = [1.16, 0.18, np.pi]

        #     elif path == '8':
        #         self.start1 = [-1.16, -0.18, 0.0]
        #         self.start2 = [0.18, -1.16, 0.5*np.pi]
        #     '''
        init_pos = 1.4
        follower_dis = 0.6
        ego_mov = 0.05
        #Ego car
        self.start1 = [0.18, -init_pos-ego_mov, 0.5*np.pi]  # -1.16
       
        if spawn_ls[7] == '1':
            self.start8 = [0.18, -init_pos-ego_mov-follower_dis, 0.5*np.pi]

        else:
            self.start8 = [0.18, -init_pos-0.3-10, 0.5*np.pi]
            
        #car 1
        if spawn_ls[1] == '1':
            self.start2 = [-init_pos, -0.18, 0.0]

        else:
            self.start2 = [-100, -100, 0.0]
        #car 5
        if spawn_ls[4] == '1':
            self.start5 = [-init_pos-follower_dis, -0.18, 0.0]

        else:
            self.start5 = [-init_pos-10, -0.18, 0.0]
            
        #car 2
        if spawn_ls[2] == '1':
            self.start3 = [init_pos, 0.18, np.pi]

        else:
            self.start3 = [100, 100, np.pi]
        #car 6
        if spawn_ls[5] == '1':
            self.start6 = [init_pos+follower_dis, 0.18, np.pi]

        else:
            self.start6 = [init_pos-10, 0.18, np.pi]
            
        #car 3
        if spawn_ls[3] == '1':
            self.start4 = [-0.18, init_pos, -np.pi*0.5]

        else:
            self.start4 = [-100, 100, np.pi]
        #car 7
        if spawn_ls[6] == '1':
            self.start7 = [-0.18, init_pos+follower_dis, -np.pi*0.5]

        else:
            self.start7 = [-100, -100, 0.0]
        
            
            
        self.resetval()
        rospy.wait_for_service('/gazebo/set_model_state')
        val = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)

        randomposition = 2 * self.dis * \
            np.random.random_sample((1, 2)) - self.dis
        # agent robot生成一个随机的角度
        randangle = 2 * math.pi * np.random.random_sample(1) - math.pi
        # 根据model name对每个物体的位置初始化
        state = ModelState()
        for i in range(len(self.gazebo_model_states.name)):
            if self.gazebo_model_states.name[i] == "point_start":
                state.reference_frame = 'world'
                state.pose.position.z = 0.0
                state.model_name = self.gazebo_model_states.name[i]
                state.pose.position.x = self.start1[0]
                state.pose.position.y = self.start1[1]
                val(state)
            if self.gazebo_model_states.name[i] == self.agentrobot1:
                state.reference_frame = 'world'
                state.pose.position.z = 0.0
                state.model_name = self.gazebo_model_states.name[i]
                rpy = [0.0, 0.0, self.start1[2]]
                q = self.quaternion_from_euler(rpy[0], rpy[1], rpy[2])
                state.pose.orientation.x = q[0]
                state.pose.orientation.y = q[1]
                state.pose.orientation.z = q[2]
                state.pose.orientation.w = q[3]
                state.pose.position.x = self.start1[0]
                state.pose.position.y = self.start1[1]
                val(state)
            if self.gazebo_model_states.name[i] == self.agentrobot2:
                state.reference_frame = 'world'
                state.pose.position.z = 0.0
                state.model_name = self.gazebo_model_states.name[i]
                rpy = [0.0, 0.0, self.start2[2]]
                q = self.quaternion_from_euler(rpy[0], rpy[1], rpy[2])
                state.pose.orientation.x = q[0]
                state.pose.orientation.y = q[1]
                state.pose.orientation.z = q[2]
                state.pose.orientation.w = q[3]
                state.pose.position.x = self.start2[0]
                state.pose.position.y = self.start2[1]
                val(state)
            if self.gazebo_model_states.name[i] == self.agentrobot3:
                state.reference_frame = 'world'
                state.pose.position.z = 0.0
                state.model_name = self.gazebo_model_states.name[i]
                rpy = [0.0, 0.0, self.start3[2]]
                q = self.quaternion_from_euler(rpy[0], rpy[1], rpy[2])
                state.pose.orientation.x = q[0]
                state.pose.orientation.y = q[1]
                state.pose.orientation.z = q[2]
                state.pose.orientation.w = q[3]
                state.pose.position.x = self.start3[0]
                state.pose.position.y = self.start3[1]
                val(state)
            if self.gazebo_model_states.name[i] == self.agentrobot4:
                state.reference_frame = 'world'
                state.pose.position.z = 0.0
                state.model_name = self.gazebo_model_states.name[i]
                rpy = [0.0, 0.0, self.start4[2]]
                q = self.quaternion_from_euler(rpy[0], rpy[1], rpy[2])
                state.pose.orientation.x = q[0]
                state.pose.orientation.y = q[1]
                state.pose.orientation.z = q[2]
                state.pose.orientation.w = q[3]
                state.pose.position.x = self.start4[0]
                state.pose.position.y = self.start4[1]
                val(state)
            if self.gazebo_model_states.name[i] == self.agentrobot5:
                state.reference_frame = 'world'
                state.pose.position.z = 0.0
                state.model_name = self.gazebo_model_states.name[i]
                rpy = [0.0, 0.0, self.start5[2]]
                q = self.quaternion_from_euler(rpy[0], rpy[1], rpy[2])
                state.pose.orientation.x = q[0]
                state.pose.orientation.y = q[1]
                state.pose.orientation.z = q[2]
                state.pose.orientation.w = q[3]
                state.pose.position.x = self.start5[0]
                state.pose.position.y = self.start5[1]
                val(state)
            if self.gazebo_model_states.name[i] == self.agentrobot6:
                state.reference_frame = 'world'
                state.pose.position.z = 0.0
                state.model_name = self.gazebo_model_states.name[i]
                rpy = [0.0, 0.0, self.start6[2]]
                q = self.quaternion_from_euler(rpy[0], rpy[1], rpy[2])
                state.pose.orientation.x = q[0]
                state.pose.orientation.y = q[1]
                state.pose.orientation.z = q[2]
                state.pose.orientation.w = q[3]
                state.pose.position.x = self.start6[0]
                state.pose.position.y = self.start6[1]
                val(state)
            if self.gazebo_model_states.name[i] == self.agentrobot7:
                state.reference_frame = 'world'
                state.pose.position.z = 0.0
                state.model_name = self.gazebo_model_states.name[i]
                rpy = [0.0, 0.0, self.start7[2]]
                q = self.quaternion_from_euler(rpy[0], rpy[1], rpy[2])
                state.pose.orientation.x = q[0]
                state.pose.orientation.y = q[1]
                state.pose.orientation.z = q[2]
                state.pose.orientation.w = q[3]
                state.pose.position.x = self.start7[0]
                state.pose.position.y = self.start7[1]
                val(state)
            if self.gazebo_model_states.name[i] == self.agentrobot8:
                state.reference_frame = 'world'
                state.pose.position.z = 0.0
                state.model_name = self.gazebo_model_states.name[i]
                rpy = [0.0, 0.0, self.start8[2]]
                q = self.quaternion_from_euler(rpy[0], rpy[1], rpy[2])
                state.pose.orientation.x = q[0]
                state.pose.orientation.y = q[1]
                state.pose.orientation.z = q[2]
                state.pose.orientation.w = q[3]
                state.pose.position.x = self.start8[0]
                state.pose.position.y = self.start8[1]
                val(state)

        print("The environment has been reset!\n")
        time.sleep(1.0)
        
    def render_env(self, pos):

        start2 = [pos[0][0], pos[1][0], pos[2][0]] 
        start1 = [pos[0][1], pos[1][1], pos[2][1]]
        start3 = [pos[0][2], pos[1][2], pos[2][2]]
        start4 = [pos[0][3], pos[1][3], pos[2][3]]
      
        self.resetval()
        rospy.wait_for_service('/gazebo/set_model_state')
        val = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)

        # 根据model name对每个物体的位置初始化
        state = ModelState()
        for i in range(len(self.gazebo_model_states.name)):
            if self.gazebo_model_states.name[i] == "point_start":
                state.reference_frame = 'world'
                state.pose.position.z = 0.0
                state.model_name = self.gazebo_model_states.name[i]
                state.pose.position.x = start1[0]
                state.pose.position.y = start1[1]
                val(state)
            if self.gazebo_model_states.name[i] == self.agentrobot1:
                state.reference_frame = 'world'
                state.pose.position.z = 0.0
                state.model_name = self.gazebo_model_states.name[i]
                rpy = [0.0, 0.0, start1[2]]
                q = self.quaternion_from_euler(rpy[0], rpy[1], rpy[2])
                state.pose.orientation.x = q[0]
                state.pose.orientation.y = q[1]
                state.pose.orientation.z = q[2]
                state.pose.orientation.w = q[3]
                state.pose.position.x = start1[0]
                state.pose.position.y = start1[1]
                val(state)
            if self.gazebo_model_states.name[i] == self.agentrobot2:
                state.reference_frame = 'world'
                state.pose.position.z = 0.0
                state.model_name = self.gazebo_model_states.name[i]
                rpy = [0.0, 0.0, start2[2]]
                q = self.quaternion_from_euler(rpy[0], rpy[1], rpy[2])
                state.pose.orientation.x = q[0]
                state.pose.orientation.y = q[1]
                state.pose.orientation.z = q[2]
                state.pose.orientation.w = q[3]
                state.pose.position.x = start2[0]
                state.pose.position.y = start2[1]
                val(state)
            if self.gazebo_model_states.name[i] == self.agentrobot3:
                state.reference_frame = 'world'
                state.pose.position.z = 0.0
                state.model_name = self.gazebo_model_states.name[i]
                rpy = [0.0, 0.0, start3[2]]
                q = self.quaternion_from_euler(rpy[0], rpy[1], rpy[2])
                state.pose.orientation.x = q[0]
                state.pose.orientation.y = q[1]
                state.pose.orientation.z = q[2]
                state.pose.orientation.w = q[3]
                state.pose.position.x = start3[0]
                state.pose.position.y = start3[1]
                val(state)
            if self.gazebo_model_states.name[i] == self.agentrobot4:
                state.reference_frame = 'world'
                state.pose.position.z = 0.0
                state.model_name = self.gazebo_model_states.name[i]
                rpy = [0.0, 0.0, start4[2]]
                q = self.quaternion_from_euler(rpy[0], rpy[1], rpy[2])
                state.pose.orientation.x = q[0]
                state.pose.orientation.y = q[1]
                state.pose.orientation.z = q[2]
                state.pose.orientation.w = q[3]
                state.pose.position.x = start4[0]
                state.pose.position.y = start4[1]
                val(state)

        #print("The environment has been rendered!\n")
        time.sleep(0.01)

    def stop_all_cars(self):
        zero = AckermannDriveStamped()
        self.pub1.publish(zero)
        self.pub2.publish(zero)
        self.pub3.publish(zero)
        self.pub4.publish(zero)
        self.pub5.publish(zero)
        self.pub6.publish(zero)
        self.pub7.publish(zero)
        self.pub8.publish(zero)
    
    def get_collision_status(self, s2, s3, s4):
        qcar2_passed = False
        qcar3_passed = False
        qcar4_passed_1 = False
        qcar4_passed_2 = False
        qcar4_check_1 = False
        if (self.robotstate3[0] < -0.34 or s3 != str(1)):
            qcar3_passed = True
        if (self.robotstate2[0] > -0.13 or s2 != str(1)):
            qcar2_passed = True
        if (self.robotstate4[1] < 0.07 or s4 != str(1)):
            qcar4_passed_1 = True
        if (self.robotstate4[1] < -0.32 or s4 != str(1)):
            qcar4_passed_2 = True
        if (self.robotstate4[1] < 0.40 or s4 != str(1)):
            qcar4_check_1 = True

        return qcar2_passed, qcar3_passed, qcar4_passed_1, qcar4_passed_2,qcar4_check_1
             



if __name__ == '__main__':
    pass

