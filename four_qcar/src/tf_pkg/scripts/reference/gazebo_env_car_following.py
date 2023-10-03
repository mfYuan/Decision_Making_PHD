#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Author: Wangcai
Date: 06/2019
"""

import rospy
from std_msgs.msg import String
from gazebo_msgs.msg import ModelStates
from gazebo_msgs.msg import ModelState
from geometry_msgs.msg import Twist
from geometry_msgs.msg import Pose
from gazebo_msgs.srv import SetModelState
from std_srvs.srv import Empty
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped

import matplotlib.pyplot as plt
import os
import shutil
import math
import numpy as np
import time
import random
import tensorflow as tf
import datetime
import cv2
from cv_bridge import CvBridge, CvBridgeError

MAXENVSIZE  = 30.0  # 边长为30的正方形作为环境的大小 (default world size = 30 by 30)
MAXLASERDIS = 10.0  # 雷达最大的探测距离 (max laser detection range)
Image_matrix = []

class envmodel():
    def __init__(self):
        rospy.init_node('control_node', anonymous=True)
        '''
        # 保存每次生成的map信息
        self.count_map = 1
        self.foldername_map='map'
        if os.path.exists(self.foldername_map):
            shutil.rmtree(self.foldername_map)
        os.mkdir(self.foldername_map)
        '''

        # agent列表
        self.ego = 'qcar2'
        self.target = 'qcar1'
	self.target_distance = 2.0 # target distance between target and ego vehicle
	self.initial_distance = 5.0 # initial distance between target and ego vehicle
	self.initial_target_speed = 2.0 # initial speed of target vehicle
        
        self.img_size = 80

        # 障碍数量

        self.gazebo_model_states = ModelStates()
        
        self.bridge       = CvBridge()
        self.image_matrix = []
        self.image_matrix_callback = []

        self.resetval()

        # 接收gazebo的modelstate消息
        self.sub = rospy.Subscriber('/gazebo/model_states', ModelStates, self.gazebo_states_callback)
        
        # ego vehicle lidar
        self.subEgoLaser = rospy.Subscriber('/' + self.ego + '/lidar', LaserScan, self.laser_states_callback)

        # 发布控制指令给agent robot !!!
	self.pub = rospy.Publisher('/' + self.ego + '/ackermann_cmd', AckermannDriveStamped, queue_size=10)
        
        time.sleep(1.0)
        
    def resetval(self):
        self.ego_robotstate    = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # x,y,v,w,yaw,vx,vy
	self.target_robotstate = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # x,y,v,w,yaw,vx,vy
        self.d                 = 0.0                                  # relative distance
        self.d_last   	       = 0.0                                  # last relative distance
	self.v	 	       = 0.0				      # relative velocity
        self.v_last            = 0.0                                  # last relative velocity
        self.w_last            = 0.0                                  # last relative angular velocity
        self.r 		       = 0.0                                  # reward (based on all 3 factors)
        self.cmd               = 0.0	                              # agent robot cmd
        self.done_list         = False                                # whether episode is over

    def sign(self, x):
        if x > 0:
            return 1
        elif x < 0:
            return -1
        else:
            return 0

    def gazebo_states_callback(self, data):
        self.gazebo_model_states = data
        # name: ['ground_plane', 'qcar1', 'qcar2', ...]
        for i in range(len(data.name)):
            if data.name[i] == self.ego:
                # robotstate--->x,y,v,w,yaw,vx,vy
                self.ego_robotstate[0] = data.pose[i].position.x
                self.ego_robotstate[1] = data.pose[i].position.y
                v = math.sqrt(data.twist[i].linear.x**2 + data.twist[i].linear.y**2)
                self.ego_robotstate[2] = v
                self.ego_robotstate[3] = data.twist[i].angular.z
                rpy = self.euler_from_quaternion(data.pose[i].orientation.x,data.pose[i].orientation.y,
                data.pose[i].orientation.z,data.pose[i].orientation.w)
                self.ego_robotstate[4] = rpy[2]
                self.ego_robotstate[5] = data.twist[i].linear.x
                self.ego_robotstate[6] = data.twist[i].linear.y

	   if data.name[i] == self.target:
                # robotstate--->x,y,v,w,yaw,vx,vy
                self.target_robotstate[0] = data.pose[i].position.x
                self.target_robotstate[1] = data.pose[i].position.y
                v = math.sqrt(data.twist[i].linear.x**2 + data.twist[i].linear.y**2)
                self.target_robotstate[2] = v
                self.target_robotstate[3] = data.twist[i].angular.z
                rpy = self.euler_from_quaternion(data.pose[i].orientation.x,data.pose[i].orientation.y,
                data.pose[i].orientation.z,data.pose[i].orientation.w)
                self.target_robotstate[4] = rpy[2]
                self.target_robotstate[5] = data.twist[i].linear.x
                self.target_robotstate[6] = data.twist[i].linear.y
    
    '''
    def image_callback(self, data):
        try:
            self.image_matrix_callback = self.bridge.imgmsg_to_cv2(data, desired_encoding="passthrough")
        except CvBridgeError as e:
            print(e)
    '''

    def laser_states_callback(self, data):
        self.laser = data

    def quaternion_from_euler(self, r, p, y):
        q = [0, 0, 0, 0]
        q[3] = math.cos(r / 2) * math.cos(p / 2) * math.cos(y / 2) + math.sin(r / 2) * math.sin(p / 2) * math.sin(y / 2)
        q[0] = math.sin(r / 2) * math.cos(p / 2) * math.cos(y / 2) - math.cos(r / 2) * math.sin(p / 2) * math.sin(y / 2)
        q[1] = math.cos(r / 2) * math.sin(p / 2) * math.cos(y / 2) + math.sin(r / 2) * math.cos(p / 2) * math.sin(y / 2)
        q[2] = math.cos(r / 2) * math.cos(p / 2) * math.sin(y / 2) - math.sin(r / 2) * math.sin(p / 2) * math.cos(y / 2)
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
            euler[0] = math.atan2(2 * (y * z + w * x), w * w - x * x - y * y + z * z)
            euler[1] = math.asin(-2 * (x * z - w * y))
            euler[2] = math.atan2(2 * (x * y + w * z), w * w + x * x - y * y - z * z)
        
        return euler
    
    def update_variables(self):
	self.d_last = self.d
	self.v_last = self.v
	
	# calculate relative distance
	x_ego    = self.ego_robotstate[0]
	y_ego    = self.ego_robotstate[1]
	x_target = self.target_robotstate[0]
	y_target = self.target_robotstate[1]
	self.d   = math.sqrt((x_ego - x_target)**2 + (y_ego - y_target)**2)

	# calculate relative velocity
	v_ego    = self.ego_robotstate[2]
	v_target = self.target_robotstate[2]
	self.v   = abs(v_ego - v_target)

    # 获取agent robot的回报值
    def getreward(self):
	# 3 kinds of rewards        
	collision_reward = 0
	distance_reward = 0
	velocity_reward = 0

	# calculate collision reward (from obstacle script)
	
	# calculate distance reward

	# calculate velocity reward

        reward = abs(d_ego - (x_leader - x_ego)) + abs(v_ego - v_leader)
        
        return reward

    # 重置environment
    def reset_env(self, start=[0.0, 0.0], goal=[10.0, 10.0]):
        self.sp = start
        self.gp = goal
        # 初始点到目标点的距离
        self.d_sg = ((self.sp[0]-self.gp[0])**2 + (self.sp[1]-self.gp[1])**2)**0.5
        # 重新初始化各参数
        self.resetval()
        
        rospy.wait_for_service('/gazebo/set_model_state')
        val = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)

        randomposition = 2 * self.dis * np.random.random_sample((1, 2)) - self.dis
        # agent robot生成一个随机的角度
        randangle = 2 * math.pi * np.random.random_sample(1) - math.pi
        # 根据model name对每个物体的位置初始化 
        state = ModelState()
        for i in range(len(self.gazebo_model_states.name)):

            if self.gazebo_model_states.name[i] == self.target:
                state.reference_frame = 'world'
                state.pose.position.x = 2.0
                state.pose.position.y = 0.0
                state.pose.position.z = 0.0
                state.model_name = self.gazebo_model_states.name[i]
                val(state)
            if self.gazebo_model_states.name[i] == self.ego:
                state.reference_frame = 'world'
                state.pose.position.x = 0.0
                state.pose.position.y = 0.0
                state.pose.position.z = 0.0
                state.model_name = self.gazebo_model_states.name[i]
                val(state)
                
		# 到目标点的距离
                self.d = math.sqrt((state.pose.position.x - self.gp[0])**2 + (state.pose.position.y - self.gp[1])**2)
            
        self.done_list = False  # episode结束的标志
        print("The environment has been reset!")     
        time.sleep(2.0)
    
    # 随机初始化障碍的位置 (Randomly initialize the location of obstacles)
   '''
   def random_square(self, a):
        obstacle_robot_position = []
        x1, y1 = -a, -a
        x2, y2 = a, -a
        x3, y3 = -a, a
        x4, y4 = a, a

        # 初始化 (Initialization)
        for i in range(self.num_obs):
            obstacle_robot_position.append([0.0, 0.0]) 

        random_list = np.random.random_sample(self.num_obs)
        for i in range(self.num_obs):
            if random_list[i] >= 0.5:
                rnd11 = np.random.random()
                rnd21 = np.random.random()
                rnd21 = np.sqrt(rnd21)
                obstacle_robot_position[i][0]=rnd21 * (rnd11 * x1 + (1 - rnd11) * x2) + (1 - rnd21) * x3
                obstacle_robot_position[i][1]=rnd21 * (rnd11 * y1 + (1 - rnd11) * y2) + (1 - rnd21) * y3
            else:
                rnd12 = np.random.random()
                rnd22 = np.random.random()
                rnd22 = np.sqrt(rnd22)
                obstacle_robot_position[i][0]=rnd22 * (rnd12 * x3 + (1 - rnd12) * x4) + (1 - rnd22) * x2
                obstacle_robot_position[i][1]=rnd22 * (rnd12 * y3 + (1 - rnd12) * y4) + (1 - rnd22) * y2
        return obstacle_robot_position
    '''
    def get_env(self):
        env_info=[]
        # input2-->agent robot的v,w,d,theta
        selfstate = [0.0, 0.0, 0.0, 0.0]
        # robotstate--->x,y,v,w,yaw,vx,vy
        selfstate[0] = self.ego_robotstate[2]  # v
        selfstate[1] = self.ego_robotstate[3]  # w
        # d代表agent机器人距离目标的位置-->归一化[0,1]
        selfstate[2] = self.d/MAXENVSIZE
        dx = -(self.ego_robotstate[0]-self.gp[0])
        dy = -(self.ego_robotstate[1]-self.gp[1])
        xp = dx*math.cos(self.ego_robotstate[4]) + dy*math.sin(self.ego_robotstate[4])
        yp = -dx*math.sin(self.ego_robotstate[4]) + dy*math.cos(self.ego_robotstate[4])
        thet = math.atan2(yp, xp)
        selfstate[3] = thet/math.pi

        # input1-->雷达信息
        laser = []
        temp = []
        sensor_info = []
        for j in range(len(self.laser.ranges)):
            tempval = self.laser.ranges[j]
            # 归一化处理
            if tempval > MAXLASERDIS:
                tempval = MAXLASERDIS
            temp.append(tempval/MAXLASERDIS)
        laser = temp
        # 将agent robot的input2和input1合并成为一个vector:[input2 input1]

        # env_info.append(laser)
        # env_info.append(selfstate)
        for i in range(len(laser)+len(selfstate)):
            if i<len(laser):
                sensor_info.append(laser[i])
            else:
                sensor_info.append(selfstate[i-len(laser)])
        
        env_info.append(sensor_info)
        #print("The state is:{}".format(state))

        # input1-->相机
        # shape of image_matrix [768,1024,3]
        self.image_matrix = np.uint8(self.image_matrix_callback)
        self.image_matrix = cv2.resize(self.image_matrix, (self.img_size, self.img_size))
        # shape of image_matrix [80,80,3]
        self.image_matrix = cv2.cvtColor(self.image_matrix, cv2.COLOR_RGB2GRAY)
        # shape of image_matrix [80,80]
        self.image_matrix = np.reshape(self.image_matrix, (self.img_size, self.img_size))
        # shape of image_matrix [80,80]
        # cv2.imshow("Image window", self.image_matrix)
        # cv2.waitKey(2)
        # (rows,cols,channels) = self.image_matrix.shape
        # print("image matrix rows:{}".format(rows))
        # print("image matrix cols:{}".format(cols))
        # print("image matrix channels:{}".format(channels))
        env_info.append(self.image_matrix)
        # print("shape of image matrix={}".format(self.image_matrix.shape))

        # 判断是否终止
        self.done_list = True
        # 是否到达目标点判断
        if self.d > self.dis:
            self.done_list = False  # 不终止
        else:
            self.done_list = True  # 终止
        
        # 是否与障碍物发生碰撞判断
        if self.done_list == False:
            for i in range(len(self.obs_pos)):
                # 障碍物的边长为1.0m
                if math.sqrt((self.ego_robotstate[0]-self.obs_pos[i][0])**2 + (self.ego_robotstate[1]-self.obs_pos[i][1])**2) >= 1.5:
                    self.done_list = False  # 不终止
                else:
                    # print("Obstacle!")
                    self.done_list = True  # 终止
                    break
        
        env_info.append(self.done_list)

        self.r = self.getreward()
        
        env_info.append(self.r)

        self.v_last = self.cmd[0]
        self.w_last = self.cmd[1]

        return env_info
    
    def step(self, cmd=[1.0, 0.0]):
        self.d_last = math.sqrt((self.ego_robotstate[0] - self.gp[0])**2 + (self.ego_robotstate[1] - self.gp[1])**2)
        self.cmd[0] = cmd[0]
        self.cmd[1] = cmd[1]
	cmd_vel = AckermannDriveStamped()
        cmd_vel.drive.speed = cmd[0]
	cmd_vel.drive.steering_angle = cmd[1]
	self.pub.publish(cmd_vel)
        
        time.sleep(0.05)

        self.d = math.sqrt((self.ego_robotstate[0] - self.gp[0])**2 + (self.ego_robotstate[1] - self.gp[1])**2)
        self.v_last = cmd[0]
        self.w_last = cmd[1]

if __name__ == '__main__':
    pass
