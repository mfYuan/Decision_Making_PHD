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
import tensorflow as tf
import datetime
import cv2
from cv_bridge import CvBridge, CvBridgeError

MAXENVSIZE = 30.0  # 边长为30的正方形作为环境的大小
MAXLASERDIS = 3.0  # 雷达最大的探测距离
Image_matrix = []
HORIZON = 0.4
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
        self.agentrobot1 = 'qcar1'
	self.agentrobot2 = 'qcar2'
        
        self.img_size = 80

        # 障碍数量
        self.num_obs = 10

        self.dis = 1.0  # 位置精度-->判断是否到达目标的距离 (Position accuracy-->Judge whether to reach the target distance)

        self.obs_pos = []  # 障碍物的位置信息

        self.gazebo_model_states = ModelStates()

        self.bridge       = CvBridge()
        self.image_matrix = []
        self.image_matrix_callback = []

        self.resetval()

        # 接收gazebo的modelstate消息
        self.sub = rospy.Subscriber('/gazebo/model_states', ModelStates, self.gazebo_states_callback)

        # subscribers
        self.subimage = rospy.Subscriber('/' + self.agentrobot1 + '/csi_front/image_raw', Image, self.image_callback)
        self.subLaser = rospy.Subscriber('/' + self.agentrobot1 + '/lidar', LaserScan, self.laser_states_callback)

        self.rearPose = rospy.Subscriber('/' + self.agentrobot1 + '/rear_pose', PoseStamped, self.pose_cb, queue_size=1)
	self.Waypoints = rospy.Subscriber('/final_waypoints5', Lane, self.lane_cb, queue_size=1)

        self.pub1 = rospy.Publisher('/' + self.agentrobot1 + '/ackermann_cmd', AckermannDriveStamped, queue_size=10)
	self.pub2 = rospy.Publisher('/' + self.agentrobot2 + '/ackermann_cmd', AckermannDriveStamped, queue_size=10)
        self.currentPose = None
        self.currentVelocity = None
        self.currentWaypoints = None
        #self.loop()

        time.sleep(1.0)

    def pose_cb(self, data):
        self.currentPose = data

    def loop(self):
        twistCommand = self.calculateTwistCommand()
        #print(twistCommand)
        return twistCommand

    def vel_cb(self, data):
        self.currentVelocity = data

    def lane_cb(self, data):
        self.currentWaypoints = data

    def calculateTwistCommand(self):
        lad = 0.0  # look ahead distance accumulator
        k = 1
        #ld = k * self.robotstate1[5]
        #print("value of vx is", self.robotstate1[2],"value of vx is", self.robotstate1[5], "value of ld is", ld)
        targetIndex = len(self.currentWaypoints.waypoints) - 1
        for i in range(len(self.currentWaypoints.waypoints)):
            if ((i + 1) < len(self.currentWaypoints.waypoints)):
                this_x = self.currentWaypoints.waypoints[i].pose.pose.position.x
                this_y = self.currentWaypoints.waypoints[i].pose.pose.position.y
                next_x = self.currentWaypoints.waypoints[i + 1].pose.pose.position.x
                next_y = self.currentWaypoints.waypoints[i + 1].pose.pose.position.y
                lad = lad + math.hypot(next_x - this_x, next_y - this_y)
                if (lad > HORIZON):
                    targetIndex = i + 1
                    break

        targetWaypoint = self.currentWaypoints.waypoints[targetIndex]
        targetX = targetWaypoint.pose.pose.position.x
        targetY = targetWaypoint.pose.pose.position.y
        currentX = self.currentPose.pose.position.x
        currentY = self.currentPose.pose.position.y
        # get vehicle yaw angle
        quanternion = (self.currentPose.pose.orientation.x, self.currentPose.pose.orientation.y, self.currentPose.pose.orientation.z, self.currentPose.pose.orientation.w)
        #euler = self.euler_from_quaternion(quanternion)
        euler = tf1.transformations.euler_from_quaternion(quanternion)
        yaw = euler[2]
        # get angle difference
        alpha = math.atan2(targetY - currentY, targetX - currentX) - yaw
        l = math.sqrt(math.pow(currentX - targetX, 2) + math.pow(currentY - targetY, 2))
        theta = math.atan(2 * 0.256 * math.sin(alpha) / l)
        #print(self.robotstate1[5])
        '''if (l > 0.5):
            theta = math.atan(2 * 0.256 * math.sin(alpha) / l)
            # #get twist command
            twistCmd = AckermannDriveStamped()
            twistCmd.drive.steering_angle = theta
            twistCmd = Twist()
            twistCmd.linear.x = targetSpeed
            twistCmd.angular.z = theta 
        else:
            twistCmd = AckermannDriveStamped()
            twistCmd.drive.speed = 0
            twistCmd.drive.steering_angle = 0
            theta=0
            twistCmd = Twist()
            twistCmd.linear.x = 0
            twistCmd.angular.z = 0'''
        return theta

    def resetval(self):
        self.robotstate1 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # x,y,v,w,yaw,vx,vy
	self.robotstate2 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # x,y,v,w,yaw,vx,vy

        self.d           = 0.0                                  # distance between qcar1 and qcar2
        self.d_last      = 0.0                                  # 前一时刻到目标的距离
        self.v_last      = 0.0                                  # 前一时刻的速度
        self.w_last      = 0.0                                  # 前一时刻的角速度
        self.r           = 0.0                                  # 奖励
        self.cmd         = [0.0, 0.0]                           # agent robot的控制指令
        self.done_list   = False                                # episode是否结束的标志

    def sign(self, x):
        if x > 0:
            return 1
        elif x < 0:
            return -1
        else:
            return 0

    def gazebo_states_callback(self, data):
        self.gazebo_model_states = data
        # name: ['ground_plane', 'jackal1', 'jackal2', 'jackal0',...]
        for i in range(len(data.name)):
	    # qcar1
            if data.name[i] == self.agentrobot1:
                # robotstate--->x,y,v,w,yaw,vx,vy
                self.robotstate1[0] = data.pose[i].position.x
                self.robotstate1[1] = data.pose[i].position.y
                v = math.sqrt(data.twist[i].linear.x**2 + data.twist[i].linear.y**2)
                self.robotstate1[2] = v
                self.robotstate1[3] = data.twist[i].angular.z
                rpy = self.euler_from_quaternion(data.pose[i].orientation.x,data.pose[i].orientation.y,
                data.pose[i].orientation.z,data.pose[i].orientation.w)
                self.robotstate1[4] = rpy[2]
                self.robotstate1[5] = data.twist[i].linear.x
                self.robotstate1[6] = data.twist[i].linear.y

   	    # qcar2
            if data.name[i] == self.agentrobot2:
                # robotstate--->x,y,v,w,yaw,vx,vy
                self.robotstate2[0] = data.pose[i].position.x
                self.robotstate2[1] = data.pose[i].position.y
                v = math.sqrt(data.twist[i].linear.x**2 + data.twist[i].linear.y**2)
                self.robotstate2[2] = v
                self.robotstate2[3] = data.twist[i].angular.z
                rpy = self.euler_from_quaternion(data.pose[i].orientation.x,data.pose[i].orientation.y,
                data.pose[i].orientation.z,data.pose[i].orientation.w)
                self.robotstate2[4] = rpy[2]
                self.robotstate2[5] = data.twist[i].linear.x
                self.robotstate2[6] = data.twist[i].linear.y    

    def image_callback(self, data):
        try:
            self.image_matrix_callback = self.bridge.imgmsg_to_cv2(data, desired_encoding="passthrough")
        except CvBridgeError as e:
            print(e)
    
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
    
    # 获取agent robot的回报值
    def getreward(self):
        
        reward = 0

	# Lose reward for each step... (weight TBD)
	reward = reward - 0.005

        # 速度发生变化就会有负的奖励 (Avoid unnecessary speed changes)
        reward = reward - 0.005*(abs(self.w_last - self.cmd[1]) + abs(self.v_last - self.cmd[0])) 

        # Collision (-1 reward)
        if math.sqrt((self.robotstate1[0] - self.robotstate2[0])**2 + (self.robotstate1[1] - self.robotstate2[1])**2) < 0.5:
            reward = reward - 5
            print("Vehicle Collision: Lose 5 reward.")
                       
        # 到达目标点有正的奖励
        '''# if self.d < self.dis and not self.done_list:
        if self.d < self.dis:
            reward = reward + 20
            print("Get 20 reward------goal point!!!!!!")
            # self.done_list = True'''

        return reward

    # 重置environment

    def reset_env(self, path='5'):
	if path == '5':
	    self.path = path
	    self.Waypoints.unregister()
	    self.Waypoints = rospy.Subscriber('/final_waypoints5', Lane, self.lane_cb, queue_size=1)
	    self.start1 = [0.18, -1.16, 0.5*np.pi]
	    self.start2 = [-1.16, -0.18, 0.0]

	if path == '6':
	    self.path = path
	    self.Waypoints.unregister()
	    self.Waypoints = rospy.Subscriber('/final_waypoints6', Lane, self.lane_cb, queue_size=1)
	    self.start1 = [1.16, 0.18, np.pi]
	    self.start2 = [-0.18, 1.16, 1.5*np.pi]

	if path == '7':
	    self.path = path
	    self.Waypoints.unregister()
	    self.Waypoints = rospy.Subscriber('/final_waypoints7', Lane, self.lane_cb, queue_size=1)
	    self.start1 = [-0.18, 1.16, 1.5*np.pi]
	    self.start2 = [1.16, 0.18, np.pi]

	elif path == '8':
	    self.path = path
	    self.Waypoints.unregister()
	    self.Waypoints = rospy.Subscriber('/final_waypoints8', Lane, self.lane_cb, queue_size=1)
	    self.start1 = [-1.16, -0.18, 0.0]
	    self.start2 = [0.18, -1.16, 0.5*np.pi]
	    
        # 初始点到目标点的距离
        self.d_sg = ((self.start1[0]-self.start2[0])**2 + (self.start1[1]-self.start2[1])**2)**0.5
        # 重新初始化各参数
        self.resetval()

        rospy.wait_for_service('/gazebo/set_model_state')
        val = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)

	# stop all vehicles
	self.step([0.0, 0.0])
	self.move(0.0)

        randomposition = 2 * self.dis * np.random.random_sample((1, 2)) - self.dis
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
                state.pose.position.x = self.start1[0] #+ randomposition[0][0]
                state.pose.position.y = self.start1[1] #+ randomposition[0][1]
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
                state.pose.position.x = self.start2[0] #+ randomposition[0][0]
                state.pose.position.y = self.start2[1] #+ randomposition[0][1]
                val(state)		
                # 到目标点的距离
        self.d = math.sqrt((self.robotstate1[0] - self.robotstate2[0])**2 + (self.robotstate1[1] - self.robotstate2[1])**2)
        self.done_list = False  # episode结束的标志
        print("The environment has been reset!")     
        time.sleep(1.0)
    
    def get_env(self):
        env_info=[]
        # input2-->agent robot的v,w,d,theta
        selfstate = [0.0, 0.0, 0.0, 0.0]
        # robotstate--->x,y,v,w,yaw,vx,vy
        selfstate[0] = self.robotstate1[2]  # v
        selfstate[1] = self.robotstate1[3]  # w
        # d代表agent机器人距离目标的位置-->归一化[0,1]
        # selfstate[2] = self.d/MAXENVSIZE
        # 第1 2次训练
        # ----------------------------------------
        # d代表agent机器人距离目标的位置-->归一化[0,1]
        # selfstate[2] = self.d/MAXENVSIZE
        # ----------------------------------------
        # 第3次训练
        # ----------------------------------------
        if self.d >= 5.0:
            selfstate[2] = 1.0
        else:
            selfstate[2] = self.d/5.0
        # ----------------------------------------
        dx = -(self.robotstate1[0]-self.robotstate2[0]) #######
        dy = -(self.robotstate1[1]-self.robotstate2[1])
        xp = dx*math.cos(self.robotstate1[4]) + dy*math.sin(self.robotstate1[4])
        yp = -dx*math.sin(self.robotstate1[4]) + dy*math.cos(self.robotstate1[4])
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
        # If qcar1 reaches end of path 5 (0.18, 0.94)
        if (self.robotstate1[0] > 0.18 and self.robotstate1[1] > 0.94 and self.path=='5'):
            self.done_list = True  # 终止 (terminated due to episode completed)
	    print("Reached other side!")

        # If qcar1 reaches end of path 6 (-0.94, 0.18)
        elif (self.robotstate1[0] < -0.94 and self.robotstate1[1] > 0.18 and self.path=='6'):
            self.done_list = True  # 终止 (terminated due to episode completed)
	    print("Reached other side!")

        # If qcar1 reaches end of path 7 (-0.18, -0.94)
        elif (self.robotstate1[0] < -0.18 and self.robotstate1[1] < -0.94 and self.path=='7'):
            self.done_list = True  # 终止 (terminated due to episode completed)
	    print("Reached other side!")

        # If qcar1 reaches end of path 8 (0.94, -0.18)
        elif (self.robotstate1[0] > 0.94 and self.robotstate1[1] < -0.18 and self.path=='8'):
            self.done_list = True  # 终止 (terminated due to episode completed)
	    print("Reached other side!")
       
	else:
            self.done_list = False  # 不终止

	if self.done_list == False:
            if math.sqrt((self.robotstate1[0] - self.robotstate2[0])**2 + (self.robotstate1[1] - self.robotstate2[1])**2) >= 0.5:
                self.done_list = False  # 不终止
            else:
                self.done_list = True  # 终止 (terminated due to collision)

        env_info.append(self.done_list)

        self.r = self.getreward()

        env_info.append(self.r)

        self.v_last = self.cmd[0]
        self.w_last = self.cmd[1]

        return env_info

    def kmph2mps(self, velocity_kmph):
        return (velocity_kmph * 1000.) / (60. * 60.)

    # Publish velocity and steering cmd for qcar1
    def step(self, cmd=[0.0, 0.0]):
        #self.d_last = math.sqrt((self.robotstate1[0] - self.robotstate2[0])**2 + (self.robotstate1[1] - self.robotstate2[1])**2)
        self.cmd[0] = cmd[0]#self.kmph2mps(cmd[0])
        self.cmd[1] = cmd[1]
        cmd_vel = AckermannDriveStamped()
        cmd_vel.drive.speed = cmd[0]
        cmd_vel.drive.steering_angle = cmd[1]
        #print("speed=",cmd_vel.drive.speed, "steering=",cmd_vel.drive.steering_angle)
        self.pub1.publish(cmd_vel)

        #time.sleep(0.05)

        self.d = math.sqrt((self.robotstate1[0] - self.robotstate2[0])**2 + (self.robotstate1[1] - self.robotstate2[1])**2)
        #self.v_last = cmd[0]
        #self.w_last = cmd[1]

    # Publish velocity and steering cmd for qcar2
    def move(self, v_qcar2):   
	cmd_vel = AckermannDriveStamped()
	cmd_vel.drive.speed = v_qcar2
	#cmd_vel.drive.steering_angle = 0.0
	self.pub2.publish(cmd_vel)

if __name__ == '__main__':
    pass
