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
from styx_msgs.msg import Lane


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


import sys
sys.path.append('/home/sdcnlab025/ROS_test/four_qcar/src/tf_pkg/safe_RL/MPC_GT_RL/GT_MPC')
import get_params
import decisiontree_l01_DiGraph
import DecisionTree_L11_DiGraph
import environment_multi_DiGraph
import Environment_Multi_Sim_DiGraph
import plot_sim_DiGraph
# import plot_level_ratio_DiGraph
import decisiontree_l01
import DecisionTree_L11
import environment_multi
import Environment_Multi_Sim
import plot_sim
import plot_level_ratio
import switching_tp

import traff
import numpy.matlib
import matplotlib.pyplot as plt
from shapely.geometry import Polygon

MAXENVSIZE = 30.0  # 边长为30的正方形作为环境的大小
MAXLASERDIS = 3.0  # 雷达最大的探测距离
Image_matrix = []
HORIZON = 1.0

debug=1
num_cars =4
action_space = np.array([[0, 1, 2, 3, 4]])


class envmodel1():
    def __init__(self):
        #rospy.init_node('control_node1', anonymous=True)
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
        # self.qcar_dict = {0: 0.0, 1: 0.3, 2: -0.3}
        k = 1
        self.qcar_dict = {0: 0.0, 1: 0.25/k, 2: -0.25/k, 3: 0.5/k, 4: -0.5/k}
        self.previous = 0

        self.obs_pos = []  # 障碍物的位置信息

        self.gazebo_model_states = ModelStates()
        self.debug = True

        # self.bridge = CvBridge()
        # self.image_matrix = []
        # self.image_matrix_callback = []

        self.resetval()

        # 接收gazebo的modelstate消息
        self.sub = rospy.Subscriber(
            '/gazebo/model_states', ModelStates, self.gazebo_states_callback)
        # self.sub1 = rospy.Subscriber(
        #     '/distances', PoseStamped, self.distance_callback)
        # self.dist1 = rospy.Subscriber(
        #     '/qcar1/distances', PoseStamped, self.dist1_callback)
        # self.dist2 = rospy.Subscriber(
        #     '/qcar2/distances', PoseStamped, self.dist2_callback)
        # self.dist3 = rospy.Subscriber(
        #     '/qcar3/distances', PoseStamped, self.dist3_callback)
        # self.dist4 = rospy.Subscriber(
        #     '/qcar4/distances', PoseStamped, self.dist4_callback)
        self.idx = rospy.Subscriber('idx5', PoseStamped, self.idx_cb)
        # subscribers
    #    # self.subimage = rospy.Subscriber(
    #         '/' + self.agentrobot1 + '/csi_front/image_raw', Image, self.image_callback)
        # self.subLaser = rospy.Subscriber(
        #     '/' + self.agentrobot1 + '/lidar', LaserScan, self.laser_states_callback)

        self.rearPose = rospy.Subscriber(
            '/' + self.agentrobot1 + '/rear_pose', PoseStamped, self.pose_cb, queue_size=1)
        self.Waypoints = rospy.Subscriber(
            '/final_waypoints5', Lane, self.lane_cb, queue_size=1)

        self.pub = rospy.Publisher(
            '/' + self.agentrobot1 + '/ackermann_cmd', AckermannDriveStamped, queue_size=10)

        self.currentPose = None
        self.currentVelocity = None
        self.currentWaypoints = None
        self.params = get_params.get_params()
            # Figure size
        
        # self.loop()
        
        time.sleep(0.01)

    def distance_callback(self, data):
        self.d_1_2 = data.pose.position.x
        self.d_1_3 = data.pose.position.y
        self.d_1_4 = data.pose.position.z

    def dist1_callback(self, data):
        self.qcar1_p1 = data.pose.position.x
        self.qcar1_p2 = data.pose.position.y
        self.qcar1_p3 = data.pose.position.z
        self.qcar1_p4 = data.pose.orientation.x
        self.qcar1_p5 = data.pose.orientation.y

    def dist2_callback(self, data):
        self.qcar2_p1 = data.pose.position.x
        self.qcar2_p2 = data.pose.position.y
        self.qcar2_p3 = data.pose.position.z

    def dist3_callback(self, data):
        self.qcar3_p1 = data.pose.position.x
        self.qcar3_p2 = data.pose.position.y
        self.qcar3_p3 = data.pose.position.z

    def dist4_callback(self, data):
        self.qcar4_p1 = data.pose.position.x
        self.qcar4_p2 = data.pose.position.y

    def pose_cb(self, data):
        self.currentPose = data
        
    def LevelK_action_DiGraph_subprocess(self, state_info, Matrix_buffer, command_carls, Level_ratio, params, c_id):
        Action_ID = [-1.0]*self.num_cars
        L_matrix_all = Matrix_buffer[c_id]['matrix']
        if c_id == 6:
            print(Matrix_buffer[c_id]['matrix'])
        ego_interact_list = Matrix_buffer[c_id]['interact_list']
               
        # L-0  decision tree
        L0_action_id =  [None]*self.num_cars  # set the action sizes according to car numbers
        L0_Q_value =  [None]*self.num_cars    # set the Q value sizes according to car numbers
        
        for car_id in range(0, self.num_cars): # find the action for level 0 decision 
            if L_matrix_all[car_id][car_id] > 0 or car_id == c_id:
                L0_Q_value[car_id], L0_action_id[car_id] = decisiontree_l01_DiGraph.decisiontree_l0(state_info, car_id, action_space, params, Level_ratio, L_matrix_all)
            else:
                L0_Q_value[car_id], L0_action_id[car_id] = [0, 0, 0], [0, 0]
        
        #predict the state for each car with level-0 policy in next 2 steps 
        X_pseudo_L0 = environment_multi_DiGraph.environment_multi(state_info, L0_action_id, params.t_step_DT, params, L_matrix_all, c_id)

        X_pseudo_L0_Id = [None]*self.num_cars
        for car_id in range(0, self.num_cars):
            X_pseudo_L0_Id[car_id] = X_pseudo_L0.copy() # copy the state for each car
            for pre_step in range(0, len(L0_action_id[0])): # 
                X_pseudo_L0_Id[car_id][pre_step, :, car_id] = state_info[:, car_id]
        #print(len(X_pseudo_L0_Id))
        # L-1
        L1_action_id = [None]*self.num_cars
        L1_Q_value = [None]*self.num_cars
        for car_id in range(0, self.num_cars):
            start = time.time()
            if L_matrix_all[car_id][car_id] > 0 or car_id == c_id:
                L1_Q_value[car_id], L1_action_id[car_id] = DecisionTree_L11_DiGraph.DecisionTree_L1(X_pseudo_L0_Id[car_id], car_id, action_space,  params, Level_ratio, L_matrix_all) # Call the decision tree function
            else:
                L1_Q_value[car_id], L1_action_id[car_id] = [0, 0, 0], [0, 0]
            end = time.time()
            # print("L1 time: ", end - start)
        X_pseudo_L1 = environment_multi_DiGraph.environment_multi(state_info, L1_action_id, params.t_step_DT, params, L_matrix_all, c_id)

        
        X_pseudo_L1_Id = [None]*self.num_cars
        for car_id in range(0, self.num_cars):
            X_pseudo_L1_Id[car_id] = X_pseudo_L1.copy()
            for pre_step in range(0, len(L1_action_id[0])):
                X_pseudo_L1_Id[car_id][pre_step, :, car_id] = state_info[:, car_id]
                
        # D-1
        if c_id == 1:
            D1_action_id =[-1]*self.num_cars
            D1_Q_value = [-1e6]*self.num_cars
            D1_Q_value_opt = [-1e6]*(self.num_cars-1)
            D1_action_id_opt = [-1]*(self.num_cars-1)
            # for car_id in range(0, self.num_cars):
            car_id = 1
            for add in range(0, self.num_cars-1):
                if (add ==0 and add in ego_interact_list) or (add !=0 and (add+1) in ego_interact_list):
                    # print('----------->', add, ego_interact_list)
                    D1_Q_value[add] = np.dot(Level_ratio[car_id*(self.num_cars-1)+add, 0], L0_Q_value[car_id]) + np.dot(Level_ratio[car_id*(self.num_cars-1)+add, 1],L1_Q_value[car_id])
                    D1_Q_value_opt[add] = np.max(D1_Q_value[add])
                    D1_action_id_opt[add] = np.argmax(D1_Q_value[add])
                if len(ego_interact_list) == 0:
                    D1_action_id[car_id] = L0_action_id[car_id][0]
                else:
                    D1_action_id[car_id] = D1_action_id_opt[np.argmax(D1_Q_value_opt)]
        
        for car_id in range(0, self.num_cars):
            if car_id in command_carls[c_id]:
                # print('c_id:{}, existing_list:{}, car_id:{}, ego_interact_list:{}, Action_ID:{}'.format(c_id, existing_list, car_id, ego_interact_list, Action_ID))
                if state_info[9][car_id]==0:
                    Action_ID [car_id] = L0_action_id[car_id][0]
                elif state_info[9][car_id]==2:
                    Action_ID [car_id] = L1_action_id[car_id][0]
                else:
                    Action_ID [car_id] = D1_action_id[car_id]


        # Level estimation update
        if c_id == 1:
            # for car_id in range(0, num_cars):
            count = 0 
            car_id = c_id
            for inter_car in range(0, self.num_cars):
                if inter_car != car_id: 
                    if inter_car in ego_interact_list:
                        if (L0_action_id[inter_car][0]==L1_action_id[inter_car][0]):
                            count = count +1
                        else:                                                          
                            if Action_ID[inter_car] == L0_action_id[inter_car][0]:
                                Level_ratio[car_id*(self.num_cars-1)+count, 0] = Level_ratio[car_id*(self.num_cars-1) + count, 0] + 0.1
                            if Action_ID[inter_car] == L1_action_id[inter_car][0]:
                                Level_ratio[car_id*(self.num_cars-1)+count, 1] = Level_ratio[car_id*(self.num_cars-1) + count, 1] + 0.1

                            Level_ratio[car_id*(self.num_cars-1)+count,:] = Level_ratio[car_id*(self.num_cars-1)+count,:]/ sum(Level_ratio[car_id*(self.num_cars-1)+count,:])   # normalizing
                            count = count +1
                    else:
                        count = count +1
            
        else:
            Level_ratio = []
            
            # print('Level_ratio', self.Level_ratio)

        # Completion check
        complete_flag = state_info
        # if sum(params.complete_flag[1, :]) == num_cars:
        #     pass
        return Action_ID, Level_ratio
        
    def globle_update(self, Action_ID, L_matrix_all=0, c_id=0):
        # plot_sim_DiGraph.plot_sim(self.X_old, self.params, self.step, self.Level_ratio, self.fig_sim)
        # plot_level_ratio.plot_level_ratio(self.Level_ratio_history, ego_car_id, opp_car_id, self.params, self.step, self.episode, self.max_step, self.fig_0)
        X_new, R = Environment_Multi_Sim_DiGraph.Environment_Multi_Sim(self.X_old, Action_ID, 0.15, self.params, L_matrix_all, c_id)#self.params.t_step_Sim
        self.X_old = X_new#

    def loop(self):
        twistCommand = self.calculateTwistCommand()
        # print(twistCommand)
        return twistCommand

    def vel_cb(self, data):
        self.currentVelocity = data

    def lane_cb(self, data):
        self.currentWaypoints = data

    def idx_cb(self, data):
        self.current_idx = data.pose.position.x
        
    def laplacian_metrix(self, X_old, params, Level_ratio, car_id):
        L_matrix_all = np.zeros((self.num_cars, self.num_cars))
        collision_list = []
        cross_conflict_list = []
        
        for i in range(0, self.num_cars):
            collision_ID, cross_conflict_ID = switching_tp.get_graph_tp(X_old, i, params, Level_ratio)
            collision_list.append(collision_ID)
            cross_conflict_list.append(cross_conflict_ID)
        
        ego_interact_list = []
        if len(cross_conflict_list[car_id])>0: 
            for i in cross_conflict_list[car_id]:
                if i not in ego_interact_list:
                    ego_interact_list.append(i)
                if len(cross_conflict_list[i])>0: 
                    for j in cross_conflict_list[i]:
                        if j not in ego_interact_list:
                            ego_interact_list.append(j)
            resulting_list = []
            for k in ego_interact_list:
                resulting_list = cross_conflict_list[k]
                resulting_list.extend(x for x in collision_list[k] if x not in cross_conflict_list[k])
                
                L_matrix_all[k][k] = len(resulting_list)
                for l in cross_conflict_list[k]:
                    L_matrix_all[k][l] = -1
                for m in collision_list[k]:
                    L_matrix_all[k][m] = -1
            
        else:
            L_matrix_all = np.zeros((self.num_cars, self.num_cars))
        
        return L_matrix_all, ego_interact_list
        
        # for i in range(0, L_matrix.shape[0]):
        #     for j in range(0, L_matrix.shape[1]):
        #         if i == j:
        #             L_matrix[i][j] = len(collision_list[i]) + len(cross_conflict_list[i])
        #         else:
        #             L_matrix[i][j] = -len(set(collision_list[i]).intersection(set(cross_conflict_list[j])))
        # print('cross_conflict_list', cross_conflict_list)    
                
    def Get_LevelK_action(self):
        t0 = time.time()
        # Animation plots
        # print(self.X_old)
        # plot_sim.plot_sim(self.X_old, self.params, self.step, self.Level_ratio, self.fig_sim)
        # Plot level history
        self.X_old = self.veh_pos_realtime(self.X_old)
        num_cars = np.shape(self.X_old)[1]
        # print(self.X_old)
        # car_id = 1
        # L_matrix_all, ego_interact_list = self.laplacian_metrix(self.X_old, self.params, self.Level_ratio, car_id)
      
        #self.Level_ratio_history[self.episode, self.step, :, :] = self.Level_ratio
        
        # L-0
        L0_action_id =  [None]*num_cars    # set the action sizes according to car numbers
        L0_Q_value =  [None]*num_cars      # set the Q value sizes according to car numbers
        
        for car_id in range(0, num_cars): # find the action for level 0 decision 
            start = time.time()
            L0_Q_value[car_id], L0_action_id[car_id] = decisiontree_l01.decisiontree_l0(self.X_old, car_id, self.action_space, self.params, self.Level_ratio)
            #print(L0_Q_value[car_id], L0_action_id[car_id])
            #L0_Q_value[car_id], L0_action_id[car_id] = [0, 0, 0], [0, 0]
            end = time.time()
            # if car_id == 1:
            #     print("L0 time: ", end - start)
        # print('L0_action_id', L0_action_id)
        # print('L0_Q_value', L0_Q_value)
        #predict the state for each car with level-0 policy in next 2 steps 
        X_pseudo_L0 = environment_multi.environment_multi(self.X_old, L0_action_id, self.t_step_DT, self.params)
     
        X_pseudo_L0_Id = [None]*num_cars
        
        for car_id in range(0, num_cars):
            X_pseudo_L0_Id[car_id] = X_pseudo_L0.copy() # copy the state for each car
            for pre_step in range(0, len(L0_action_id[0])): # 
                X_pseudo_L0_Id[car_id][pre_step, :, car_id] = self.X_old[:, car_id]
        #print(len(X_pseudo_L0_Id))
        # L-1
        L1_action_id = [None]*num_cars
        L1_Q_value = [None]*num_cars
        for car_id in range(0, num_cars):
            start = time.time()
            L1_Q_value[car_id], L1_action_id[car_id] = DecisionTree_L11.DecisionTree_L1(X_pseudo_L0_Id[car_id], car_id, self.action_space,  self.params, self.Level_ratio) # Call the decision tree function
            end = time.time()
            # print("L1 time: ", end - start)
        X_pseudo_L1 = environment_multi.environment_multi(self.X_old, L1_action_id, self.t_step_DT, self.params)
       
        
        X_pseudo_L1_Id = [None]*num_cars
        for car_id in range(0, num_cars):
            X_pseudo_L1_Id[car_id] = X_pseudo_L1.copy()
            for pre_step in range(0, len(L1_action_id[0])):
                X_pseudo_L1_Id[car_id][pre_step, :, car_id] = self.X_old[:, car_id]
                
        # D-1
        D1_action_id =[None]*num_cars
        D1_Q_value = [None]*num_cars
        D1_Q_value_opt = [None]*(num_cars-1)
        D1_action_id_opt = [None]*(num_cars-1)
        for car_id in range(0, num_cars):
           for add in range(0, num_cars-1):
                D1_Q_value[add] = np.dot(self.Level_ratio[car_id*(num_cars-1)+add, 1], L0_Q_value[car_id]) + np.dot(self.Level_ratio[car_id*(num_cars-1)+add, 0],L1_Q_value[car_id])

                D1_Q_value_opt[add] = np.max(D1_Q_value[add])
                D1_action_id_opt[add] = np.argmax(D1_Q_value[add])
           D1_action_id[car_id] = D1_action_id_opt[np.argmax(D1_Q_value_opt)]
           
           #print(D1_action_id[car_id])
                

        # Controller selection for each cars
        # AV: Auto controller, HV: Level-K contoller
        Action_id = [None]*num_cars
        for car_id in range(0, num_cars):
            if self.X_old[4][car_id]==1:
                # Action_id[car_id] = L1_action_id[car_id][0]
                Action_id[car_id] = D1_action_id[car_id]
                #print(Action_id[car_id])
                # Action_id[car_id] = L1_action_id[car_id][0]
            elif self.X_old[4][car_id]==2:
                # Action_id[car_id] = L0_action_id[car_id][0]
                Action_id[car_id] = L1_action_id[car_id][0]
                # Action_id[car_id] = D1_action_id[car_id]
            else:
                Action_id[car_id] = L0_action_id[car_id][0]
        #print(Action_id)

        # Level estimation update
        for car_id in range(0, num_cars):
            count = 0 
            for inter_car in range(0, num_cars):
                if inter_car != car_id:
                   if (L0_action_id[inter_car][0]==L1_action_id[inter_car][0]):
                       count = count +1
                   else:                                                          
                        if Action_id[inter_car] == L0_action_id[inter_car][0]:
                           self.Level_ratio[car_id*(num_cars-1)+count, 0] = self.Level_ratio[car_id*(num_cars-1) + count, 0] + 0.5
                        if Action_id[inter_car] == L1_action_id[inter_car][0]:
                           self.Level_ratio[car_id*(num_cars-1)+count, 1] = self.Level_ratio[car_id*(num_cars-1) + count, 1] + 0.5

                        self.Level_ratio[car_id*(num_cars-1)+count,:] = self.Level_ratio[car_id*(num_cars-1)+count,:]/ sum(self.Level_ratio[car_id*(num_cars-1)+count,:])   # normalizing
                        count = count+1
        
        # print(self.Level_ratio)

        # State update
        # X_new, R = Environment_Multi_Sim.Environment_Multi_Sim(self.X_old, Action_id, self.params.t_step_Sim, self.params)
        # self.X_old = X_new


        # Reward plot
        color = ['b','r','m','g']
        # self.R_history[self.episode, :, self.step] = np.transpose(R)
        
        # plt.figure(2)
        # plt.plot(step,R[0],color='b',marker='.',markersize=16)
        # plt.plot(step,R[1],color='r',marker='.',markersize=16)


        # Plot the level_history
        ego_car_id = 1  # AV
        opp_car_id = 2  # Car 4
        # plot_level_ratio.plot_level_ratio(self.Level_ratio_history, ego_car_id, 1, self.params, self.step, self.episode, self.max_step, self.fig_0)
        # plot_level_ratio.plot_level_ratio(self.Level_ratio_history, ego_car_id, 2, self.params, self.step, self.episode, self.max_step, self.fig_2)
        # plot_level_ratio.plot_level_ratio(self.Level_ratio_history, ego_car_id, 3, self.params, self.step, self.episode, self.max_step, self.fig_3)
        # # Timer
        # t1 = time.time()
        # time_per_step = t1 - t0
        # print('total', time_per_step)


        # Completion check
        if sum(self.complete_flag[self.episode, :]) == num_cars:
            pass
        if self.step+1 <300:
            self.step += 1
    
        return Action_id, self.X_old
    
    def Get_LevelK_action_DiGraph(self, c_id, others_list, existing_list, Action_ID):
        t0 = time.time()
        # Animation plots
        
        # Plot level history
        self.state_info = self.veh_pos_realtime(self.X_old)
        # print(self.X_old)
        # self.state_info = self.X_old
        L_matrix_all, ego_interact_list = self.laplacian_metrix(self.state_info, self.params, self.Level_ratio, c_id)
        # if c_id == 3:
        #     print(L_matrix_all)
        #     print(ego_interact_list)
        # print(ego_interact_list)
        # print('Car_ID:{}|||ego_interact_list:{}'.format(c_id, ego_interact_list))
        # print('-----------------------------')
        # time.sleep(1)
        # if c_id == 1:
        #     self.Level_ratio_history[self.episode, self.step, :, :] = self.Level_ratio
        # else:
        #     self.Level_ratio_history[self.episode, self.step, :, :]
        
        # L-0  decision tree
        L0_action_id =  [None]*self.num_cars    # set the action sizes according to car numbers
        L0_Q_value =  [None]*self.num_cars      # set the Q value sizes according to car numbers
        
        for car_id in range(0, self.num_cars): # find the action for level 0 decision 
            start = time.time()
            if L_matrix_all[car_id][car_id] > 0 or car_id == c_id:
                # print(L_matrix_all)
                # print(c_id, car_id)
                # print('------')
                L0_Q_value[car_id], L0_action_id[car_id] = decisiontree_l01_DiGraph.decisiontree_l0(self.state_info, car_id, self.action_space, self.params, self.Level_ratio, L_matrix_all)
            else:
                L0_Q_value[car_id], L0_action_id[car_id] = [0, 0, 0], [0, 0]
            end = time.time()
        
        #predict the state for each car with level-0 policy in next 2 steps 
        X_pseudo_L0 = environment_multi_DiGraph.environment_multi(self.state_info, L0_action_id, self.t_step_DT, self.params, L_matrix_all, c_id)
     
        X_pseudo_L0_Id = [None]*self.num_cars
        for car_id in range(0, self.num_cars):
            X_pseudo_L0_Id[car_id] = X_pseudo_L0.copy() # copy the state for each car
            for pre_step in range(0, len(L0_action_id[0])): # 
                X_pseudo_L0_Id[car_id][pre_step, :, car_id] = self.state_info[:, car_id]
        #print(len(X_pseudo_L0_Id))
        # L-1
        L1_action_id = [None]*self.num_cars
        L1_Q_value = [None]*self.num_cars
        for car_id in range(0, self.num_cars):
            start = time.time()
            if L_matrix_all[car_id][car_id] > 0 or car_id == c_id:
                L1_Q_value[car_id], L1_action_id[car_id] = DecisionTree_L11_DiGraph.DecisionTree_L1(X_pseudo_L0_Id[car_id], car_id, self.action_space,  self.params, self.Level_ratio, L_matrix_all) # Call the decision tree function
            else:
                L1_Q_value[car_id], L1_action_id[car_id] = [0, 0, 0], [0, 0]
            end = time.time()
            # print("L1 time: ", end - start)
        X_pseudo_L1 = environment_multi_DiGraph.environment_multi(self.state_info, L1_action_id, self.t_step_DT, self.params, L_matrix_all, c_id)
       
        
        X_pseudo_L1_Id = [None]*self.num_cars
        for car_id in range(0, self.num_cars):
            X_pseudo_L1_Id[car_id] = X_pseudo_L1.copy()
            for pre_step in range(0, len(L1_action_id[0])):
                X_pseudo_L1_Id[car_id][pre_step, :, car_id] = self.state_info[:, car_id]
                
        # D-1
        if c_id == 1:
            D1_action_id =[None]*self.num_cars
            D1_Q_value = [None]*self.num_cars
            D1_Q_value_opt = [None]*(self.num_cars-1)
            D1_action_id_opt = [None]*(self.num_cars-1)
            # for car_id in range(0, self.num_cars):
            car_id = 1
            for add in range(0, self.num_cars-1):
                if (add ==0 and add in ego_interact_list) or (add !=0 and (add+1) in ego_interact_list):
                    # print('----------->', add, ego_interact_list)
                    D1_Q_value[add] = np.dot(self.Level_ratio[car_id*(self.num_cars-1)+add, 0], L0_Q_value[car_id]) + np.dot(self.Level_ratio[car_id*(self.num_cars-1)+add, 1],L1_Q_value[car_id])
                    D1_Q_value_opt[add] = np.max(D1_Q_value[add])
                    D1_action_id_opt[add] = np.argmax(D1_Q_value[add])
                if len(ego_interact_list) == 0:
                    D1_action_id[car_id] = L0_action_id[car_id][0]
                else:
                    # print('adaptive')
                    D1_action_id[car_id] = D1_action_id_opt[np.argmax(D1_Q_value_opt)]
        # car_id = 1
        # for add in range(0, self.num_cars-1):
        #     if L_matrix_all[add][add] < 0:
        #     D1_Q_value[add] = np.dot(self.Level_ratio[car_id*(self.num_cars-1)+add, 0], L0_Q_value[car_id]) + np.dot(self.Level_ratio[car_id*(self.num_cars-1)+add, 1],L1_Q_value[car_id])
        #     D1_Q_value_opt[add] = np.max(D1_Q_value[add])
        #     D1_action_id_opt[add] = np.argmax(D1_Q_value[add])
        # D1_action_id[car_id] = D1_action_id_opt[np.argmax(D1_Q_value_opt)]

        # Controller selection for each cars
        # AV: Auto controller, HV: Level-K contoller
        # print('existing_list', existing_list)
        
        for car_id in range(0, self.num_cars):
            if car_id in ego_interact_list or car_id == c_id:
                if car_id not in existing_list:
                    # print('c_id:{}, existing_list:{}, car_id:{}, ego_interact_list:{}, Action_ID:{}'.format(c_id, existing_list, car_id, ego_interact_list, Action_ID))
                    if self.state_info[9][car_id]==0:
                        Action_ID [car_id] = L0_action_id[car_id][0]
                    elif self.state_info[9][car_id]==2:
                        Action_ID [car_id] = L1_action_id[car_id][0]
                    else:
                        Action_ID [car_id] = D1_action_id[car_id]
        # if c_id == 1:
        #     print('ego_interact_list', ego_interact_list)
        #     print("Ego------>: ", Action_ID [c_id])
                    
             
                    
            # if self.state_info[c_id][4][car_id]==1:
            #     Action_id[car_id] = L1_action_id[car_id][0]
            #     # Action_id[car_id] = D1_action_id[car_id]
            #     # Action_id[car_id] = L1_action_id[car_id][0]
            # elif self.state_info[c_id][4][car_id]==2:
            #     # Action_id[car_id] = L0_action_id[car_id][0]
            #     Action_id[car_id] = L1_action_id[car_id][0]
            #     # Action_id[car_id] = D1_action_id[car_id]
            # else:
            #     Action_id[car_id] = L0_action_id[car_id][0]
        #print(Action_id)

        # Level estimation update
        if c_id == 1:
            # for car_id in range(0, self.num_cars):
            count = 0 
            car_id = c_id
            for inter_car in range(0, self.num_cars):
                if inter_car != car_id: 
                    if inter_car in ego_interact_list:
                        if (L0_action_id[inter_car][0]==L1_action_id[inter_car][0]):
                            count = count +1
                        else:                                                          
                            if Action_ID[inter_car] == L0_action_id[inter_car][0]:
                                self.Level_ratio[car_id*(self.num_cars-1)+count, 0] = self.Level_ratio[car_id*(self.num_cars-1) + count, 0] + 0.1
                            if Action_ID[inter_car] == L1_action_id[inter_car][0]:
                                self.Level_ratio[car_id*(self.num_cars-1)+count, 1] = self.Level_ratio[car_id*(self.num_cars-1) + count, 1] + 0.1

                            self.Level_ratio[car_id*(self.num_cars-1)+count,:] = self.Level_ratio[car_id*(self.num_cars-1)+count,:]/ sum(self.Level_ratio[car_id*(self.num_cars-1)+count,:])   # normalizing
                            count = count +1
                    else:
                        count = count +1
                            
            # print('Level_ratio', self.Level_ratio)
        
        # for car_id in range(0, self.num_cars):
        #     count = 0 
        #     for inter_car in range(0, self.num_cars):
        #         if inter_car != car_id:
        #            if (L0_action_id[inter_car][0]==L1_action_id[inter_car][0]):
        #                count = count +1
        #            else:                                                          
        #                 if Action_ID[inter_car] == L0_action_id[inter_car][0]:
        #                    self.Level_ratio[car_id*(self.num_cars-1)+count, 0] = self.Level_ratio[car_id*(self.num_cars-1) + count, 0] + 0.5
        #                 if Action_ID[inter_car] == L1_action_id[inter_car][0]:
        #                    self.Level_ratio[car_id*(self.num_cars-1)+count, 1] = self.Level_ratio[car_id*(self.num_cars-1) + count, 1] + 0.5

        #                 self.Level_ratio[car_id*(self.num_cars-1)+count,:] = self.Level_ratio[car_id*(self.num_cars-1)+count,:]/ sum(self.Level_ratio[car_id*(self.num_cars-1)+count,:])   # normalizing
        #                 count = count+1
        
        # print(self.Level_ratio)

        # State update
        # X_new, R = Environment_Multi_Sim_DiGraph.Environment_Multi_Sim(self.state_info, Action_ID, self.params.t_step_Sim, self.params, L_matrix_all, c_id)
        # self.X_old = X_new

        # Reward plot
        # if c_id == 1:
        #     color = ['b','r','m','g']
        #     self.R_history[self.episode, :, self.step] = np.transpose(R)
        
        # plt.figure(2)
        # plt.plot(step,R[0],color='b',marker='.',markersize=16)
        # plt.plot(step,R[1],color='r',marker='.',markersize=16)


        # Plot the level_history
        ego_car_id = 1  # AV
        opp_car_id = 2  # Car 4
        # plot_level_ratio.plot_level_ratio(self.Level_ratio_history, ego_car_id, opp_car_id, self.params, self.step, self.episode, self.max_step, self.fig_lh)

        # # Timer
        t1 = time.time()
        time_per_step = t1 - t0
        # print('total', time_per_step)


        # Completion check
        if sum(self.complete_flag[self.episode, :]) == self.num_cars:
            pass
        
        if c_id == 1:
            self.step += 1
        self.other_car_id = list(set(others_list) - set(ego_interact_list))
        try:
            self.other_car_id.remove(c_id)
        except:
            pass
            
        
        # for item in ego_interact_list:
        #     if item not in interaction_list:
        #         self.state_info[self.num_cars][:][item] = self.state_info[c_id][:][item]
        
        existing_list = existing_list + ego_interact_list+[c_id]
        # if len(self.other_car_id) == 0:
        #     self.X_old = self.state_info[self.num_cars]
        # print('ego_interact_list{} and Actions{}'.format(ego_interact_list, Action_ID))
        # time.sleep(0.5)
        # simulation update
        # if len(self.other_car_id) == 0:
        #     # plot_sim_DiGraph.plot_sim(self.X_old, self.params, self.step, self.Level_ratio, self.fig_sim)
        #     # plot_level_ratio.plot_level_ratio(self.Level_ratio_history, ego_car_id, opp_car_id, self.params, self.step, self.episode, self.max_step, self.fig_0)
        #     X_new, R = Environment_Multi_Sim_DiGraph.Environment_Multi_Sim(self.state_info, Action_ID, self.params.t_step_Sim, self.params, L_matrix_all, c_id)
        #     self.X_old = X_new
            # self.X_old = self.veh_pos_realtime(self.X_old)
        return Action_ID, self.X_old, self.other_car_id, existing_list, L_matrix_all
    
    def Get_LevelK_action_DiGraph_subprocess(self, state_info, Matrix_buffer, command_carls, c_id):
        Action_ID = [None]*self.num_cars
        L_matrix_all = Matrix_buffer[c_id]['matrix']
        ego_interact_list = Matrix_buffer[c_id]['interact_list']
        self.state_info = state_info
        # L_matrix_all, ego_interact_list = self.laplacian_metrix(self.state_info, self.params, self.Level_ratio, c_id)
        # print(ego_interact_list)
        # print('Car_ID:{}|||ego_interact_list:{}'.format(c_id, ego_interact_list))
        # print('-----------------------------')
        # time.sleep(1)
        # if c_id == 1:
        #     self.Level_ratio_history[self.episode, self.step, :, :] = self.Level_ratio
        # L-0  decision tree
        L0_action_id =  [None]*self.num_cars    # set the action sizes according to car numbers
        L0_Q_value =  [None]*self.num_cars      # set the Q value sizes according to car numbers
        
        for car_id in range(0, self.num_cars): # find the action for level 0 decision 
            if L_matrix_all[car_id][car_id] > 0 or car_id == c_id:
                L0_Q_value[car_id], L0_action_id[car_id] = decisiontree_l01_DiGraph.decisiontree_l0(self.state_info, car_id, self.action_space, self.params, self.Level_ratio, L_matrix_all)
            else:
                L0_Q_value[car_id], L0_action_id[car_id] = [0, 0, 0], [0, 0]
        
        #predict the state for each car with level-0 policy in next 2 steps 
        X_pseudo_L0 = environment_multi_DiGraph.environment_multi(self.state_info, L0_action_id, self.t_step_DT, self.params, L_matrix_all, c_id)
    
        X_pseudo_L0_Id = [None]*self.num_cars
        for car_id in range(0, self.num_cars):
            X_pseudo_L0_Id[car_id] = X_pseudo_L0.copy() # copy the state for each car
            for pre_step in range(0, len(L0_action_id[0])): # 
                X_pseudo_L0_Id[car_id][pre_step, :, car_id] = self.state_info[:, car_id]
        #print(len(X_pseudo_L0_Id))
        # L-1
        L1_action_id = [None]*self.num_cars
        L1_Q_value = [None]*self.num_cars
        for car_id in range(0, self.num_cars):
            start = time.time()
            if L_matrix_all[car_id][car_id] > 0 or car_id == c_id:
                L1_Q_value[car_id], L1_action_id[car_id] = DecisionTree_L11_DiGraph.DecisionTree_L1(X_pseudo_L0_Id[car_id], car_id, self.action_space,  self.params, self.Level_ratio, L_matrix_all) # Call the decision tree function
            else:
                L1_Q_value[car_id], L1_action_id[car_id] = [0, 0, 0], [0, 0]
            end = time.time()
            # print("L1 time: ", end - start)
        X_pseudo_L1 = environment_multi_DiGraph.environment_multi(self.state_info, L1_action_id, self.t_step_DT, self.params, L_matrix_all, c_id)
    
        
        X_pseudo_L1_Id = [None]*self.num_cars
        for car_id in range(0, self.num_cars):
            X_pseudo_L1_Id[car_id] = X_pseudo_L1.copy()
            for pre_step in range(0, len(L1_action_id[0])):
                X_pseudo_L1_Id[car_id][pre_step, :, car_id] = self.state_info[:, car_id]
                
        # D-1
        if c_id == 1:
            D1_action_id =[None]*self.num_cars
            D1_Q_value = [None]*self.num_cars
            D1_Q_value_opt = [None]*(self.num_cars-1)
            D1_action_id_opt = [None]*(self.num_cars-1)
            # for car_id in range(0, self.num_cars):
            car_id = 1
            for add in range(0, self.num_cars-1):
                if (add ==0 and add in ego_interact_list) or (add !=0 and (add+1) in ego_interact_list):
                    # print('----------->', add, ego_interact_list)
                    D1_Q_value[add] = np.dot(self.Level_ratio[car_id*(self.num_cars-1)+add, 0], L0_Q_value[car_id]) + np.dot(self.Level_ratio[car_id*(self.num_cars-1)+add, 1],L1_Q_value[car_id])
                    D1_Q_value_opt[add] = np.max(D1_Q_value[add])
                    D1_action_id_opt[add] = np.argmax(D1_Q_value[add])
                if len(ego_interact_list) == 0:
                    D1_action_id[car_id] = L0_action_id[car_id][0]
                else:
                    D1_action_id[car_id] = D1_action_id_opt[np.argmax(D1_Q_value_opt)]
        
        for car_id in range(0, self.num_cars):
            if car_id in command_carls[c_id]:
                # print('c_id:{}, existing_list:{}, car_id:{}, ego_interact_list:{}, Action_ID:{}'.format(c_id, existing_list, car_id, ego_interact_list, Action_ID))
                if self.state_info[9][car_id]==0:
                    Action_ID [car_id] = L0_action_id[car_id][0]
                elif self.state_info[9][car_id]==2:
                    Action_ID [car_id] = L1_action_id[car_id][0]
                else:
                    Action_ID [car_id] = D1_action_id[car_id]
                # if c_id ==1:
                #     print('YES',  Action_ID [car_id])
 
        # Level estimation update
        if c_id == 1:
            # for car_id in range(0, self.num_cars):
            count = 0 
            car_id = c_id
            for inter_car in range(0, self.num_cars):
                if inter_car != car_id: 
                    if inter_car in ego_interact_list:
                        if (L0_action_id[inter_car][0]==L1_action_id[inter_car][0]):
                            count = count +1
                        else:                                                          
                            if Action_ID[inter_car] == L0_action_id[inter_car][0]:
                                self.Level_ratio[car_id*(self.num_cars-1)+count, 0] = self.Level_ratio[car_id*(self.num_cars-1) + count, 0] + 0.1
                            if Action_ID[inter_car] == L1_action_id[inter_car][0]:
                                self.Level_ratio[car_id*(self.num_cars-1)+count, 1] = self.Level_ratio[car_id*(self.num_cars-1) + count, 1] + 0.1

                            self.Level_ratio[car_id*(self.num_cars-1)+count,:] = self.Level_ratio[car_id*(self.num_cars-1)+count,:]/ sum(self.Level_ratio[car_id*(self.num_cars-1)+count,:])   # normalizing
                            count = count +1
                    else:
                        count = count +1
                            
            # print('Level_ratio', self.Level_ratio)

        # Completion check
        if sum(self.complete_flag[self.episode, :]) == self.num_cars:
            pass
        
        # if c_id == 1:
        #     self.step += 1
        # self.other_car_id = list(set(others_list) - set(ego_interact_list))
        # try:
        #     self.other_car_id.remove(c_id)
        # except:
        #     pass
            
        
        # for item in ego_interact_list:
        #     if item not in interaction_list:
        #         self.state_info[self.num_cars][:][item] = self.state_info[c_id][:][item]
        
        # existing_list = existing_list + ego_interact_list+[c_id]
        # if len(self.other_car_id) == 0:
        #     self.X_old = self.state_info[self.num_cars]
        # print('ego_interact_list{} and Actions{}'.format(ego_interact_list, Action_ID))
        # time.sleep(0.5)
            
            # self.X_old = self.veh_pos_realtime(self.X_old)
        return Action_ID
    
    # def globle_update(self, Action_ID, L_matrix_all=0, c_id=0):
    #     plot_sim_DiGraph.plot_sim(self.X_old, self.params, self.step, self.Level_ratio, self.fig_sim)
    #     # plot_level_ratio.plot_level_ratio(self.Level_ratio_history, ego_car_id, opp_car_id, self.params, self.step, self.episode, self.max_step, self.fig_0)
    #     X_new, R = Environment_Multi_Sim_DiGraph.Environment_Multi_Sim(self.state_info, Action_ID, self.params.t_step_Sim, self.params, L_matrix_all, c_id)
    #     self.X_old = X_new
        
    def get_id_Lmatrix(self):
        state_info = self.veh_pos_realtime(self.X_old)
        # state_info = self.veh_pos_realtime(self.X_old)
        # self.X_old = state_info
        others_list = [i for i in range(self.num_cars)]#[0, 1, 2, 3]
        waiting_list = []
        Matrix_buffer = {}
        command_carls ={}
        for item in others_list:
            Matrix_buffer[item] = 0
            command_carls[item] = []
        #Matrix_buffer = {0:0, 1:0, 2:0, 3:0}
        # command_carls= {0:[], 1:[], 2:[], 3:[]}
        exiting_ls = []
        c_id = 1
        waiting_list.append(c_id)
        L_matrix_all, ego_interact_list = self.laplacian_metrix(state_info, self.params, self.Level_ratio, c_id)
        
        Matrix_buffer[c_id] = {'matrix':L_matrix_all, 'interact_list':ego_interact_list}
        if len(ego_interact_list)==0:
            ego_interact_list = [c_id]
       
        other_car_id = list(set(others_list) - set(ego_interact_list))   
        # print('others_list{} - ego_interact_list{} = other_car_id{}|||'.format(others_list,ego_interact_list,other_car_id))
        
        command_carls[c_id] = ego_interact_list
        for i in ego_interact_list:
            if i not in exiting_ls:
                exiting_ls.append(i)
        
        # exiting_ls.append(ego_interact_list)
        while len(other_car_id) > 0 and len(exiting_ls)<self.num_cars:
            c_id = other_car_id[0]
            L_matrix_all, ego_interact_list = self.laplacian_metrix(state_info, self.params, self.Level_ratio, c_id)
            Matrix_buffer[c_id] = {'matrix':L_matrix_all, 'interact_list':ego_interact_list}
            _ls = []
            if len(ego_interact_list)==0:
                ego_interact_list = [c_id]
            for id in ego_interact_list:
                if id not in exiting_ls:
                    _ls.append(id)
            command_carls[c_id] = _ls
            waiting_list.append(c_id)
            try:
                other_car_id.remove(c_id)
            except:
                pass
            # exiting_ls.append(ego_interact_list)
            for i in ego_interact_list:
                if i not in exiting_ls:
                    exiting_ls.append(i)
            # other_car_id = list(set(other_car_id) - set(ego_interact_list))
            # print('other_car_id{}, exiting_ls{s}'.format(other_car_id, exiting_ls))
        
        return waiting_list, Matrix_buffer, command_carls, state_info
        
        
    def Get_LevelK_action_DiGraph_ego(self, c_id):
        t0 = time.time()
        # Animation plots
        # plot_sim_DiGraph.plot_sim(self.X_old, self.params, self.step, self.Level_ratio, self.fig_sim)
        # Plot level history
        self.X_old = self.veh_pos_realtime(self.X_old)
        # print(self.X_old)
        
        L_matrix_all, ego_interact_list = self.laplacian_metrix(self.X_old, self.params, self.Level_ratio, c_id)
        
        self.Level_ratio_history[self.episode, self.step, :, :] = self.Level_ratio
        
        # L-0
        L0_action_id =  [None]*self.num_cars    # set the action sizes according to car numbers
        L0_Q_value =  [None]*self.num_cars      # set the Q value sizes according to car numbers
        
        for car_id in range(0, self.num_cars): # find the action for level 0 decision 
            start = time.time()
            if L_matrix_all[car_id][car_id] > 0 or car_id == c_id:
                L0_Q_value[car_id], L0_action_id[car_id] = decisiontree_l01_DiGraph.decisiontree_l0(self.X_old, car_id, self.action_space, self.params, self.Level_ratio, L_matrix_all)
            else:
                L0_Q_value[car_id], L0_action_id[car_id] = [0, 0, 0], [0, 0]
            end = time.time()
        
        #predict the state for each car with level-0 policy in next 2 steps 
        X_pseudo_L0 = environment_multi_DiGraph.environment_multi(self.X_old, L0_action_id, self.t_step_DT, self.params, L_matrix_all, c_id)
        
        X_pseudo_L0_Id = [None]*self.num_cars
        for car_id in range(0, self.num_cars):
            X_pseudo_L0_Id[car_id] = X_pseudo_L0.copy() # copy the state for each car
            for pre_step in range(0, len(L0_action_id[0])): # 
                X_pseudo_L0_Id[car_id][pre_step, :, car_id] = self.X_old[:, car_id]
        #print(len(X_pseudo_L0_Id))
        # L-1
        L1_action_id = [None]*self.num_cars
        L1_Q_value = [None]*self.num_cars
        for car_id in range(0, self.num_cars):
            start = time.time()
            if L_matrix_all[car_id][car_id] > 0 or car_id == c_id:
                L1_Q_value[car_id], L1_action_id[car_id] = DecisionTree_L11_DiGraph.DecisionTree_L1(X_pseudo_L0_Id[car_id], car_id, self.action_space,  self.params, self.Level_ratio) # Call the decision tree function
            else:
                L1_Q_value[car_id], L1_action_id[car_id] = [0, 0, 0], [0, 0]
            end = time.time()
            # print("L1 time: ", end - start)
        X_pseudo_L1 = environment_multi_DiGraph.environment_multi(self.X_old, L1_action_id, self.t_step_DT, self.params, L_matrix_all, c_id)
        
        
        X_pseudo_L1_Id = [None]*self.num_cars
        for car_id in range(0, self.num_cars):
            X_pseudo_L1_Id[car_id] = X_pseudo_L1.copy()
            for pre_step in range(0, len(L1_action_id[0])):
                X_pseudo_L1_Id[car_id][pre_step, :, car_id] = self.X_old[:, car_id]
                
        # D-1
        D1_action_id =[None]*self.num_cars
        D1_Q_value = [None]*self.num_cars
        D1_Q_value_opt = [None]*(self.num_cars-1)
        D1_action_id_opt = [None]*(self.num_cars-1)
        for car_id in range(0, self.num_cars):
            for add in range(0, self.num_cars-1):
                D1_Q_value[add] = np.dot(self.Level_ratio[car_id*(self.num_cars-1)+add, 0], L0_Q_value[car_id]) + np.dot(self.Level_ratio[car_id*(self.num_cars-1)+add, 1],L1_Q_value[car_id])
                D1_Q_value_opt[add] = np.max(D1_Q_value[add])
                D1_action_id_opt[add] = np.argmax(D1_Q_value[add])
            D1_action_id[car_id] = D1_action_id_opt[np.argmax(D1_Q_value_opt)]
        # car_id = 1
        # for add in range(0, self.num_cars-1):
        #     if L_matrix_all[add][add] < 0:
        #     D1_Q_value[add] = np.dot(self.Level_ratio[car_id*(self.num_cars-1)+add, 0], L0_Q_value[car_id]) + np.dot(self.Level_ratio[car_id*(self.num_cars-1)+add, 1],L1_Q_value[car_id])
        #     D1_Q_value_opt[add] = np.max(D1_Q_value[add])
        #     D1_action_id_opt[add] = np.argmax(D1_Q_value[add])
        # D1_action_id[car_id] = D1_action_id_opt[np.argmax(D1_Q_value_opt)]

        # Controller selection for each cars
        # AV: Auto controller, HV: Level-K contoller
        Action_id = [None]*self.num_cars
        for car_id in range(0,self.num_cars):
            if self.X_old[4][car_id]==1:
                Action_id[car_id] = L1_action_id[car_id][0]
                # Action_id[car_id] = D1_action_id[car_id]
                # Action_id[car_id] = L1_action_id[car_id][0]
            elif self.X_old[4][car_id]==2:
                # Action_id[car_id] = L0_action_id[car_id][0]
                Action_id[car_id] = L1_action_id[car_id][0]
                # Action_id[car_id] = D1_action_id[car_id]
            else:
                Action_id[car_id] = L0_action_id[car_id][0]
        #print(Action_id)

        # Level estimation update
        # for car_id in range(0, self.num_cars):
        #     count = 0 
        #     for inter_car in range(0, self.num_cars):
        #         if inter_car != car_id:
        #            if (L0_action_id[inter_car][0]==L1_action_id[inter_car][0]):
        #                count = count +1
        #            else:                                                          
        #                 if Action_id[inter_car] == L0_action_id[inter_car][0]:
        #                    self.Level_ratio[car_id*(self.num_cars-1)+count, 0] = self.Level_ratio[car_id*(self.num_cars-1) + count, 0] + 0.5
        #                 if Action_id[inter_car] == L1_action_id[inter_car][0]:
        #                    self.Level_ratio[car_id*(self.num_cars-1)+count, 1] = self.Level_ratio[car_id*(self.num_cars-1) + count, 1] + 0.5

        #                 self.Level_ratio[car_id*(self.num_cars-1)+count,:] = self.Level_ratio[car_id*(self.num_cars-1)+count,:]/ sum(self.Level_ratio[car_id*(self.num_cars-1)+count,:])   # normalizing
        #                 count = count+1
        
        #print(self.Level_ratio)
        
        # State update
        X_new, R = Environment_Multi_Sim_DiGraph.Environment_Multi_Sim(self.X_old, Action_id, self.params.t_step_Sim, self.params, L_matrix_all, c_id)
        self.X_old = X_new


        # Reward plot
        color = ['b','r','m','g']
        self.R_history[self.episode, :, self.step] = np.transpose(R)
        
        # plt.figure(2)
        # plt.plot(step,R[0],color='b',marker='.',markersize=16)
        # plt.plot(step,R[1],color='r',marker='.',markersize=16)


        # Plot the level_history
        ego_car_id = 1  # AV
        opp_car_id = 2  # Car 4
        # plot_level_ratio.plot_level_ratio(self.Level_ratio_history, ego_car_id, opp_car_id, self.params, self.step, self.episode, self.max_step, self.fig_lh)

        # # Timer
        # t1 = time.time()
        # time_per_step = t1 - t0
        # print('total', time_per_step)


        # Completion check
        if sum(self.complete_flag[self.episode, :]) == self.num_cars:
            pass
        
        self.step += 1
        _list = [x for x in range(0, self.num_cars)]
        other_car_id = list(set(_list) - set(ego_interact_list))
        # print('ego_interact_list', ego_interact_list)
        # print('other_car_id', other_car_id)
        return Action_id, self.X_old, other_car_id
        
    def Init_final(self, id, path_id, params, x_car, y_car):
        start_index = params.KDtrees_12[path_id[id][1]].query([x_car, y_car],1)[1]
        Destination = len(params.waypoints[path_id[id][1]])
                                                                    
        return [start_index, Destination]
        
    def get_position(self, get_path, Driver_level):
        self.traffic = self.vehicle_position(self.params, self.traffic, self.AV_cars, get_path, Driver_level)
       
        vehicle_states = np.block ([[self.traffic.x], [self.traffic.y], [self.traffic.orientation], 
                               [self.traffic.v_car], [self.traffic.AV_flag], 
                               [self.traffic.Final_x], [self.traffic.Final_y], [self.traffic.path_id], [self.traffic.Currend_index], [self.traffic.Driver_level]])
        print("Get initial position, done!")
        
        return vehicle_states
        
    def vehicle_position(self, params, traffic, AV_cars, get_path, Driver_level):       
        self.episode = 0
        self.step = 0
        
        # Init positions
        for id in range(0, self.num_cars):
            if id ==0:
                x_car = self.robotstate2[0]
                y_car = self.robotstate2[1]
                v_car = self.robotstate2[2]
                orientation_car = self.robotstate2[4]
                Dr_level = Driver_level[0]
            elif id == 1: # AV
                x_car = self.robotstate1[0]
                y_car = self.robotstate1[1]
                v_car = self.robotstate1[2]
                orientation_car = self.robotstate1[4]
                Dr_level = Driver_level[1]
            elif id == 2:
                x_car = self.robotstate3[0]
                y_car = self.robotstate3[1]
                v_car = self.robotstate3[2]
                orientation_car = self.robotstate3[4]
                Dr_level = Driver_level[2]
            elif id == 3:
                x_car = self.robotstate4[0]
                y_car = self.robotstate4[1]
                v_car = self.robotstate4[2]
                orientation_car = self.robotstate4[4]
                Dr_level = Driver_level[3]
            elif id == 4:
                x_car = self.robotstate5[0]
                y_car = self.robotstate5[1]
                v_car = self.robotstate5[2]
                orientation_car = self.robotstate5[4]
                Dr_level = Driver_level[4]
            elif id == 5:
                x_car = self.robotstate6[0]
                y_car = self.robotstate6[1]
                v_car = self.robotstate6[2]
                orientation_car = self.robotstate6[4]
                Dr_level = Driver_level[5]
            elif id == 6:
                x_car = self.robotstate7[0]
                y_car = self.robotstate7[1]
                v_car = self.robotstate7[2]
                orientation_car = self.robotstate7[4]
                Dr_level = Driver_level[6]
            elif id == 7:
                x_car = self.robotstate8[0]
                y_car = self.robotstate8[1]
                v_car = self.robotstate8[2]
                orientation_car = self.robotstate8[4]
                Dr_level = Driver_level[7]
     
                        
            #v_car = np.random.uniform(v_min, v_max)
            rear_x = x_car - math.cos(orientation_car) * 0.128
            rear_y = y_car - math.sin(orientation_car) * 0.128
            
            get_init_final = self.Init_final(id, get_path, params, rear_x, rear_y)
            init_index = get_init_final[0]
            final_index =  get_init_final[1]
            path_id = get_path[id][1]
        
            for i in range(0, len(AV_cars)):
                if id == AV_cars[i]:
                    AV_flag = 1
                    break
                else:
                    AV_flag = 0
            Current_index = init_index
            
            
            traffic = traff.update(traffic, x_car, y_car, orientation_car, v_car, AV_flag, float(init_index), float(final_index),
                                float(path_id), float(Current_index), float(Dr_level))
            
        return traffic
    
    def veh_pos_realtime(self, X_old):       
        for id in range(0, self.num_cars):
            if id ==0:
                x_car = self.robotstate2[0]
                y_car = self.robotstate2[1]
                v_car = self.robotstate2[2]
                orientation_car = self.robotstate2[4] 
                
            elif id == 1: # AV
                x_car = self.robotstate1[0]
                y_car = self.robotstate1[1]
                v_car = self.robotstate1[2]
                orientation_car = self.robotstate1[4]
                
            elif id == 2:
                x_car = self.robotstate3[0]
                y_car = self.robotstate3[1]
                v_car = self.robotstate3[2]
                orientation_car = self.robotstate3[4]
                
            elif id == 3:
                x_car = self.robotstate4[0]
                y_car = self.robotstate4[1]
                v_car = self.robotstate4[2]
                orientation_car = self.robotstate4[4]
                
            elif id == 4:
                x_car = self.robotstate5[0]
                y_car = self.robotstate5[1]
                v_car = self.robotstate5[2]
                orientation_car = self.robotstate5[4]
                
            elif id == 5:
                x_car = self.robotstate6[0]
                y_car = self.robotstate6[1]
                v_car = self.robotstate6[2]
                orientation_car = self.robotstate6[4]
            elif id == 6:
                x_car = self.robotstate7[0]
                y_car = self.robotstate7[1]
                v_car = self.robotstate7[2]
                orientation_car = self.robotstate7[4]
            elif id == 7:
                x_car = self.robotstate8[0]
                y_car = self.robotstate8[1]
                v_car = self.robotstate8[2]
                orientation_car = self.robotstate8[4]
                
                        
            #v_car = np.random.uniform(v_min, v_max)
            rear_x = x_car - math.cos(orientation_car) * 0.128
            rear_y = y_car - math.sin(orientation_car) * 0.128
            path_id = X_old[7][id]
            path_id = str(int(path_id))
            current_index = self.params.KDtrees_12[path_id].query([rear_x, rear_y],1)[1]
            
            X_old[0][id] = x_car
            X_old[1][id] = y_car
            X_old[2][id] = orientation_car
            X_old[3][id] = v_car# X_old[3][id] # MODIFIY THIS LATER!!!
            X_old[8][id] = current_index
        self.X_old = X_old
            # X_old[4:10][:] = X_old[4:10][:]
            # X_old[4] = X_old[4][id]
            # X_old[4][id] = X_old[5][id]
            # X_old[4][id] = X_old[6][id]
            # X_old[4][id] = X_old[7][id]

            # Current_index = X_old[8][id]
            # Dr_level = X_old[9][id]
            
        return X_old
    
    def param_init(self):
        #Parameter Initialization for MPC-based Level-k               
        self.num_cars = self.params.num_cars
        self.max_episode = self.params.max_episode
        self.t_step_DT = self.params.t_step_DT
        self.complete_flag = self.params.complete_flag
        self.AV_cars = np.array([1])
        self.params.num_AV = len(self.AV_cars)
        self.num_Human = self.num_cars - 1
        self.params.num_Human = self.num_Human
        self.num_lanes = self.params.num_lanes
       
        self.outdir = self.params.outdir
        self.render = self.params.render
        self.max_step = 500#0.2-50
        # Initial guess for the level ratio (0 1)
        # self.Level_ratio = np.array([[0.2, 0.8]])
        self.Level_ratio = np.array([[0.99, 0.01]])
        self.Level_ratio = np.matlib.repmat(self.Level_ratio, self.num_cars * (self.num_cars-1), 1) # 12*2
     
        # Define the sizes of the variables
        self.Level_ratio_history=np.zeros((self.max_episode, self.max_step, np.shape(self.Level_ratio)[0], np.shape(self.Level_ratio)[1]))
        
        self.R_history = np.zeros((self.max_episode, self.num_cars, self.max_step))

        # action space 
        # 0: maintain 1: turn left 2: turn right 3:accelerate 4: decelerate 5: hard brake 6:increased acceleration
        # 7: small left turn 8: small right turn
        
        self.action_space = np.array([[0, 1, 2, 3, 4]]) # 0: maintain 1: accele. 2: deccele.
        self.params.sim_case = 1 # 0-level-0; 1-adaptive; 2-level-1 
        self.new_algorithm = False
        self.multi_processing = True#'case2'
        # self.fig_sim = plt.figure(1, figsize=(6, 6))
        # self.fig_0 = plt.figure(0, figsize=(6, 3))
        # self.fig_2 = plt.figure(2, figsize=(6, 3))
        # self.fig_3 = plt.figure(3, figsize=(6, 3))

        # Create output folder
        if not(os.path.exists(self.outdir)):
                os.mkdir(self.outdir)

        # pick a simulation case
        # 0 - Aggressive
        # 1 - Adaptive
        # 2 - Conservative
        '''self.params.sim_case = 1

        # parameters based on the simulation case
        if self.params.sim_case == 0:
            self.params.outfile = 'aggressive.mp4'
            self.params.plot_fname = 'plot_agg'
        elif self.params.sim_case == 1:
            self.params.outfile = 'adaptive.mp4'
            self.params.plot_fname = 'plot_adp'
        else:
            self.params.outfile = 'conservative.mp4'
            self.params.plot_fname = 'plot_con'''
            
            
    def Initial_GTMPC(self, get_path, Driver_level):
        self.param_init()
        self.traffic = traff.initial() 
        self.X_old = self.get_position(get_path, Driver_level)
        self.get_path =get_path
        #self.state_info = [self.X_old]*(self.num_cars+1)

               

    def calculateTwistCommand(self):
        lad = 0.0  # look ahead distance accumulator
        k = 1
        #ld = k * self.robotstate1[5]
        #print("value of vx is", self.robotstate1[2],"value of vx is", self.robotstate1[5], "value of ld is", ld)
        targetIndex = len(self.currentWaypoints.waypoints) - 1
        print("value of targetIndex is", targetIndex)
        for i in range(len(self.currentWaypoints.waypoints)):
            if ((i + 1) < len(self.currentWaypoints.waypoints)):
                this_x = self.currentWaypoints.waypoints[i].pose.pose.position.x
                this_y = self.currentWaypoints.waypoints[i].pose.pose.position.y
                next_x = self.currentWaypoints.waypoints[i +
                                                         1].pose.pose.position.x
                next_y = self.currentWaypoints.waypoints[i +
                                                         1].pose.pose.position.y

                lad = lad + math.hypot(next_x - this_x, next_y - this_y)
                if (lad > HORIZON):
                    targetIndex = i + 1
                    break
        #print('Next Waypoint Index',targetIndex)
        targetWaypoint = self.currentWaypoints.waypoints[targetIndex]
        targetX = targetWaypoint.pose.pose.position.x
        targetY = targetWaypoint.pose.pose.position.y
        currentX = self.currentPose.pose.position.x
        currentY = self.currentPose.pose.position.y
        # get vehicle yaw angle
        quanternion = (self.currentPose.pose.orientation.x, self.currentPose.pose.orientation.y,
                       self.currentPose.pose.orientation.z, self.currentPose.pose.orientation.w)
        #euler = self.euler_from_quaternion(quanternion)
        euler = tf1.transformations.euler_from_quaternion(quanternion)
        yaw = euler[2]
        # get angle difference
        alpha = math.atan2(targetY - currentY, targetX - currentX) - yaw
        l = math.sqrt(math.pow(currentX - targetX, 2) +
                      math.pow(currentY - targetY, 2))
        theta = math.atan(2 * 0.256 * math.sin(alpha) / l)
        # print(self.robotstate1[5])
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
        self.robotstate1 = [0.0, 0.0, 0.0, 0.0,
                            0.0, 0.0, 0.0]  # x,y,v,w,yaw,vx,vy
        self.robotstate2 = [0.0, 0.0, 0.0, 0.0,
                            0.0, 0.0, 0.0]  # x,y,v,w,yaw,vx,vy
        self.robotstate3 = [0.0, 0.0, 0.0, 0.0,
                            0.0, 0.0, 0.0]  # x,y,v,w,yaw,vx,vyz
        self.robotstate4 = [0.0, 0.0, 0.0, 0.0,
                            0.0, 0.0, 0.0]  # x,y,v,w,yaw,vx,vyz
        self.robotstate5 = [0.0, 0.0, 0.0, 0.0,
                            0.0, 0.0, 0.0]  # x,y,v,w,yaw,vx,vyz
        self.robotstate6 = [0.0, 0.0, 0.0, 0.0,
                            0.0, 0.0, 0.0]  # x,y,v,w,yaw,vx,vy
        self.robotstate7 = [0.0, 0.0, 0.0, 0.0,
                            0.0, 0.0, 0.0]  # x,y,v,w,yaw,vx,vy
        self.robotstate8 = [0.0, 0.0, 0.0, 0.0,
                            0.0, 0.0, 0.0]  # x,y,v,w,yaw,vx,vy

        # distance between qcar1 and qcar2 (old)
        self.d = 0.0
        self.d_1_2 = 0.0					# distance between qcar1 and qcar2
        self.d_1_3 = 0.0					# distance between qcar1 and qcar3
        self.d_1_4 = 0.0					# distance between qcar1 and qcar4

        self.d_last = 0.0                                  # 前一时刻到目标的距离
        self.v_last = 0.0                                  # 前一时刻的速度
        self.accel_last = 0.0
        self.accel_current = 0.0
        self.current_idx_last = 0
        self.current_idx = 0
        self.d_1_2_last = 0.0
        self.d_1_3_last = 0.0
        self.d_1_4_last = 0.0
        self.w_last = 0.0                                  # 前一时刻的角速度
        self.r = 0.0                                  # 奖励
        self.cmd = [0.0, 0.0]                           # agent robot的控制指令
        self.done_list = False                                # episode是否结束的标志

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

    def image_callback(self, data):
        try:
            self.image_matrix_callback = self.bridge.imgmsg_to_cv2(
                data, desired_encoding="passthrough")
        except CvBridgeError as e:
            print(e)

    def laser_states_callback(self, data):
        self.laser = data

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

    # 获取agent robot的回报值
    def getreward(self):

        reward = 0

        # Lose reward for each step... (weight TBD)
        reward = reward - 0.01
        #print('Time passes, reward lose 0.001')

        # 速度发生变化就会有负的奖励 (Avoid unnecessary speed changes)
        #abs(self.w_last - self.cmd[1])
        # reward = reward - 0.002 * \
        #abs(self.v_last - self.cmd[0])
        if self.accel_last != self.accel_current:
            reward = reward - 0.005
            # print('accleration changed, penaulty 0.005',
            #       self.accel_last, self.accel_current)

        if (self.d_1_2 - self.d_1_2_last) < 0 and self.d_1_2 < 0.35 and self.robotstate1[2] > 0:
            reward = reward - 0.1
            print('Too close qcar2 and Apporaching reward lose 0.1')
        if (self.d_1_3 - self.d_1_3_last) < 0 and self.d_1_3 < 0.35 and self.robotstate1[2] > 0:
            reward = reward - 0.1
            print('Too close qcar3 and Apporaching reward lose 0.1')
        if (self.d_1_4 - self.d_1_4_last) < 0 and self.d_1_4 < 0.35 and self.robotstate1[2] > 0:
            reward = reward - 0.1
            print('Too close qcar4 and Apporaching reward lose 0.1')

        if self.current_idx_last < self.current_idx:
            reward = reward + (self.current_idx-self.current_idx_last)*0.1
            #print('reached point:',self.current_idx,'reward increased 0.1')

        #print('last speed: '+str(self.v_last),'current speed: '+ str(self.cmd[0]))
        # print('Speed Changes,reward lose',  0.001 *abs(self.v_last - self.cmd[0])

        if self.other_side == True:
            reward = reward + 2
            print("[Link1] Reached other side! Gain 2 reward.")

        # Collision (-1 reward)
        if self.d_1_2 < 0.2:
            reward = reward - 5
            print("[Link1] Vehicle Collision with qcar2: Lose 5 reward.")

        # elif self.d_1_2 < 0.3:
        #     reward = reward - 0.01
        #     print('Too close to qcar2, reward lose 0.01')

        # Collision (-1 reward)
        if self.d_1_3 < 0.2:
            reward = reward - 5
            print("[Link1] Vehicle Collision with qcar3: Lose 5 reward.")

        # elif self.d_1_3 < 0.3:
        #     reward = reward - 0.01
        #     print('Too close to qcar3, reward lose 0.01')

        # Collision (-1 reward)
        if self.d_1_4 < 0.2:
            reward = reward - 5
            print("[Link1] Vehicle Collision with qcar4: Lose 5 reward.")

        # elif self.d_1_4 < 0.3:
        #     reward = reward - 0.01
        #     print('Too close to qcar4, reward lose 0.01')
        return reward

    def get_env(self):
        self.done_list = False
        env_info = []
        # input2-->agent robot的v,w,d,theta
        selfstate = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        # robotstate--->x,y,v,w,yaw,vx,vy
        selfstate[0] = self.robotstate1[2]  # v
        # selfstate[1] = self.robotstate1[3]  # w

        # Left turn
        if self.path == '10':
            #selfstate[2] = '1'
            selfstate[1] = '1'
            selfstate[2] = self.qcar2_p1
            selfstate[3] = self.qcar4_p1
            selfstate[4] = self.qcar3_p1
            selfstate[5] = self.qcar1_p2
            selfstate[6] = self.qcar1_p2
            selfstate[7] = self.qcar1_p1

        # Straight
        if self.path == '5':
            #selfstate[2] = '2'
            selfstate[1] = '2'
            selfstate[2] = self.qcar2_p2
            selfstate[3] = 0.0
            selfstate[4] = self.qcar3_p3
            selfstate[5] = self.qcar1_p4
            selfstate[6] = 0.0
            selfstate[7] = self.qcar1_p3

        # Right turn
        if self.path == '4':
            #selfstate[2] = '3'
            selfstate[1] = '3'
            selfstate[2] = self.qcar2_p3
            selfstate[3] = 0.0
            selfstate[4] = 0.0
            selfstate[5] = self.qcar1_p5
            selfstate[6] = 0.0
            selfstate[7] = 0.0

        if self.s2 != '1':
            selfstate[2] = 0.0
            selfstate[5] = 0.0

        if self.s4 != '1':
            selfstate[3] = 0.0
            selfstate[6] = 0.0

        if self.s3 != '1':
            selfstate[4] = 0.0
            selfstate[7] = 0.0

        # print(selfstate)
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
            if i < len(laser):
                sensor_info.append(laser[i])
            else:
                sensor_info.append(selfstate[i-len(laser)])

        #print('qcar spawn:', self.s2, self.s3, self.s4)
        #print('qcar1 velocity:',self.robotstate1[2])

        sensor_info.append(self.v2)
        sensor_info.append(self.v4)
        sensor_info.append(self.v3)
        # print(sensor_info[-8:])

        env_info.append(sensor_info)
        #print("The state is:{}".format(sensor_info[-11:]))

        # input1-->相机
        # shape of image_matrix [768,1024,3]
        # self.image_matrix = np.uint8(self.image_matrix_callback)
        # self.image_matrix = cv2.resize(
        #     self.image_matrix, (self.img_size, self.img_size))
        # # shape of image_matrix [80,80,3]
        # self.image_matrix = cv2.cvtColor(self.image_matrix, cv2.COLOR_RGB2GRAY)
        # # shape of image_matrix [80,80]
        # self.image_matrix = np.reshape(
        #     self.image_matrix, (self.img_size, self.img_size))
        # # shape of image_matrix [80,80]
        # # cv2.imshow("Image window", self.image_matrix)
        # # cv2.waitKey(2)
        # # (rows,cols,channels) = self.image_matrix.shape
        # # print("image matrix rows:{}".format(rows))
        # # print("image matrix cols:{}".format(cols))
        # # print("image matrix channels:{}".format(channels))
        # env_info.append(self.image_matrix)
        # print("shape of image matrix={}".format(self.image_matrix.shape))

        # 判断是否终止
        self.done_list = True
        self.other_side = False
        self.collision = 0
        # 是否到达目标点判断
        # If qcar1 reaches end of path 4 (0.94, -0.18)
        # and self.robotstate1[1] < -0.18
        if (self.robotstate1[0] > 0.94 and self.path == '4'):
            self.other_side = True
            self.done_list = True  # 终止 (terminated due to episode completed)
            #print("Reached other side!")

        # If qcar1 reaches end of path 5 (0.18, 0.94)
        # self.robotstate1[0] > 0.18 and
        elif (self.robotstate1[1] > 0.94 and self.path == '5'):
            self.other_side = True
            self.done_list = True  # 终止 (terminated due to episode completed)
            #print("Reached other side!")

        # If qcar1 reaches end of path 6 (-0.94, 0.18)
        # and self.robotstate1[1] > 0.18
        elif (self.robotstate1[0] < -0.94 and self.path == '6'):
            self.other_side = True
            self.done_list = True  # 终止 (terminated due to episode completed)
            #print("Reached other side!")

        # If qcar1 reaches end of path 7 (-0.18, -0.94)
        # self.robotstate1[0] < -0.18 and
        elif (self.robotstate1[1] < -0.94 and self.path == '7'):
            self.other_side = True
            self.done_list = True  # 终止 (terminated due to episode completed)
            #print("Reached other side!")

        # If qcar1 reaches end of path 8 (0.94, -0.18)
        # and self.robotstate1[1] < -0.18
        elif (self.robotstate1[0] > 0.94 and self.path == '8'):
            self.other_side = True
            self.done_list = True  # 终止 (terminated due to episode completed)
            #print("Reached other side!")

        # If qcar1 reaches end of path 10 (-0.94, 0.18)
        # and self.robotstate1[1] > 0.18
        elif (self.robotstate1[0] < -0.94 and self.path == '10'):
            self.other_side = True
            self.done_list = True  # 终止 (terminated due to episode completed)
            #print("Reached other side!")

        else:
            self.done_list = False  # 不终止

        if self.done_list == False:
            if self.d_1_2 >= 0.2 and self.d_1_3 >= 0.2 and self.d_1_4 >= 0.2:
                self.done_list = False
            else:
                self.done_list = True  # 终止 (terminated due to collision)
        if self.d_1_2 < 0.2:
            self.collision = 2
        elif self.d_1_3 < 0.2 :
            self.collision = 3
        elif self.d_1_4 < 0.2:
            self.collision = 4
        else:
            self.collision = 0

        env_info.append(self.collision)
        env_info.append(self.done_list)

        self.r = self.getreward()

        env_info.append(self.r)

        self.v_last = self.cmd[0]
        self.accel_last = self.accel_current
        #self.w_last = self.cmd[1]
        self.d_1_2_last = self.d_1_2
        self.d_1_3_last = self.d_1_3
        self.d_1_4_last = self.d_1_4
        self.current_idx_last = self.current_idx

        return env_info

    def kmph2mps(self, velocity_kmph):
        return (velocity_kmph * 1000.) / (60. * 60.)

    def update_path(self, path):
        self.Waypoints.unregister()
        self.Waypoints = rospy.Subscriber(
            '/final_waypoints' + path, Lane, self.lane_cb, queue_size=1)
        self.path = path

    # Publish velocity and steering cmd for qcarx
    def step(self, cmd=[0.0, 0.0]):
        #self.d_last = math.sqrt((self.robotstate1[0] - self.robotstate2[0])**2 + (self.robotstate1[1] - self.robotstate2[1])**2)
        self.cmd[0] = cmd[0]  # self.kmph2mps(cmd[0])
        self.cmd[1] = self.loop()  # cmd[1]
        cmd_vel = AckermannDriveStamped()
        cmd_vel.drive.speed = self.cmd[0]
        cmd_vel.drive.steering_angle = self.cmd[1]
        #print("speed=",cmd_vel.drive.speed, "steering=",cmd_vel.drive.steering_angle)
        self.pub.publish(cmd_vel)

        # time.sleep(0.05)

        #self.d = math.sqrt((self.robotstate1[0] - self.robotstate2[0])**2 + (self.robotstate1[1] - self.robotstate2[1])**2)
        #self.v_last = cmd[0]
        #self.w_last = cmd[1]
    
    def SteeringCommand(self, X_old, car_id, params):
        lad = 0.0  # look ahead distance accumulator
        k = 3.5
        
        x = self.robotstate1[0]#X_old[0][car_id]
        y = self.robotstate1[1]
        yaw = self.robotstate1[4]
        v = self.robotstate1[2]#X_old[3][car_id]
        rear_x = x - math.cos(yaw) * 0.128
        rear_y = y - math.sin(yaw) * 0.128
        
        ld = k * v
        path_id = X_old[7][car_id]
        path_id = str(int(path_id))
        
        #print("value of vx is", self.robotstate1[2],"value of vx is", self.robotstate1[5], "value of ld is", ld)
        
        targetIndex = params.KDtrees_12[path_id].query([rear_x, rear_y],1)[1]
        current_index = targetIndex
        
        for i in range(current_index, len(params.waypoints[path_id])):
            if ((i + 1) < len(params.waypoints[path_id])):
                this_x = rear_x
                this_y = rear_y
                next_x = params.waypoints[path_id][i+1][0]
                next_y = params.waypoints[path_id][i+1][1]

                lad = lad + math.hypot(next_x - this_x, next_y - this_y)
                HORIZON = ld
                # print(HORIZON)
                if (lad > HORIZON):
                    targetIndex = i + 1
                    break
        #print('Next Waypoint Index',targetIndex)
        targetWaypoint = params.waypoints[path_id][targetIndex]
        targetX = targetWaypoint[0]
        targetY = targetWaypoint[1]
        
        currentX = rear_x
        currentY = rear_y
        
        # get angle difference
        alpha = math.atan2(targetY - currentY, targetX - currentX) - yaw
        l = math.sqrt(math.pow(currentX - targetX, 2) +
                        math.pow(currentY - targetY, 2))
        Kdd = 0.25
        # theta = math.atan(2 * 0.256 * math.sin(alpha) / Kdd*ld)
        theta = math.atan(2 * 0.256 * math.sin(alpha) / (l+0.0001))
        return theta
    
    def controller(self):
        kp = 1
        ki = 0.003#.005#0.1
        kd = 0.015#0.06
        
        delta_v = desired_v - current_v #error term

        # integral = self.integral_val1 + (delta_v * st) #intergral

        # derivative = (delta_v - self.derivative_val1) / st #derivative
        
    def speed(self, pos, action, params):
        car_id = 1
        t1 = time.time()
        # if self.qcar_dict[action] < 0:
        #     speed = 0.0
        # elif self.qcar_dict[action] > 0:
        #     speed = 0.6
        # else:
        #     speed = self.robotstate1[2]
        #     accel = 0
        # speed = pos[3][1]*
        delta_t = t1 - self.previous
        if delta_t > 1.0:
            delta_t = 0.1
        speed = self.robotstate1[2] + self.qcar_dict[action]*delta_t
        kp = 2
        delta_v = kp*(speed - self.robotstate1[2]) #error term
        self.previous = t1
        if speed >0.6:
            speed = 0.6
        elif speed < 0:
            speed = 0
        
        
        self.accel_current = self.qcar_dict[action]
         
        self.cmd[0] = speed*2.1#2.5
        if self.cmd[0] < 0:
            self.cmd[0]=0
        self.cmd[1] = self.SteeringCommand(pos, car_id, params)#self.loop()
            
        if pos[8][car_id]+20 > pos[6][car_id]:
            self.cmd[0] = 1.2
            self.cmd[1] = 0
            

        
        cmd_vel = AckermannDriveStamped()
        
        cmd_vel.drive.speed = self.cmd[0]
        cmd_vel.drive.steering_angle = self.cmd[1]
        #if self.cmd[0]:
        cmd_vel.drive.acceleration = self.qcar_dict[action]
        self.pub.publish(cmd_vel)
        self.data_vel = [speed, self.robotstate1[2], speed-self.robotstate1[2]]
        #print(speed, self.robotstate1[2], speed-self.robotstate1[2])

    def accel(self, accel):
        if accel < 0:
            speed = 0.0
        elif accel > 0:
            speed = 1.5
        else:
            speed = self.robotstate1[2]
            accel = 0
	
	
	#print('Speed of Robot is', self.robotstate1[2])
        self.accel_current = accel
        self.cmd[0] = speed

        self.cmd[1] = self.loop()
        cmd_vel = AckermannDriveStamped()
        
        cmd_vel.drive.speed = self.cmd[0]
        cmd_vel.drive.steering_angle = self.cmd[1]
        cmd_vel.drive.acceleration = accel
        self.pub.publish(cmd_vel)

    def set_velocity(self, v2, v3, v4, s2, s3, s4):

        self.s2 = s2
        self.s3 = s3
        self.s4 = s4

        if s2 == '1':
            self.v2 = self.robotstate2[2]
        else:
            self.v2 = 0.0
        if s3 == '1':
            self.v3 = self.robotstate3[2]
        else:
            self.v3 = 0.0

        if s4 == '1':
            self.v4 = self.robotstate4[2]
        else:
            self.v4 = 0.0

        


'''if __name__ == '__main__':
    pass'''

