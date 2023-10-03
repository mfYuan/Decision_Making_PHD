#!/usr/bin/env python
# -*- coding: utf-8 -*-
# import rospy
import multiprocessing as mp
from functools import partial
import concurrent.futures
import signal
import sys
import random
import numpy as np
import time
import os
import socket
import json



sys.path.append(os.path.join('E:/', 'Mingfeng/safe_RL/MPC_RL/GT_MPC'))#'''E:\Mingfeng\safe_RL\MPC_RL'''
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
import math

from posenode import OptitrackParser


debug=1
num_cars =4
action_space = np.array([[0, 1, 2, 3, 4]])

def do_something(time_for_sleep):
    print('Sleeping {} second...'.format(time_for_sleep))
    time.sleep(time_for_sleep)
    print('Done Sleeping...')


def LevelK_action_DiGraph_subprocess(state_info, Matrix_buffer, command_carls, Level_ratio, params, c_id):
    if c_id == 0:
        print('car 0')
    elif c_id ==1:
        print('car 1')
    elif c_id ==2:
        print('car 2')
    elif c_id ==3:
        print('car 3')
    Action_ID = [-1.0]*num_cars
    L_matrix_all = Matrix_buffer[c_id]['matrix']
    ego_interact_list = Matrix_buffer[c_id]['interact_list']
    state_info = state_info
    
    # L-0  decision tree
    L0_action_id =  [None]*num_cars    # set the action sizes according to car numbers
    L0_Q_value =  [None]*num_cars      # set the Q value sizes according to car numbers
    
    for car_id in range(0, num_cars): # find the action for level 0 decision 
        if L_matrix_all[car_id][car_id] > 0 or car_id == c_id:
            L0_Q_value[car_id], L0_action_id[car_id] = decisiontree_l01_DiGraph.decisiontree_l0(state_info, car_id, action_space, params, Level_ratio, L_matrix_all)
        else:
            L0_Q_value[car_id], L0_action_id[car_id] = [0, 0, 0], [0, 0]
    
    #predict the state for each car with level-0 policy in next 2 steps 
    X_pseudo_L0 = environment_multi_DiGraph.environment_multi(state_info, L0_action_id, params.t_step_DT, params, L_matrix_all, c_id)

    X_pseudo_L0_Id = [None]*num_cars
    for car_id in range(0, num_cars):
        X_pseudo_L0_Id[car_id] = X_pseudo_L0.copy() # copy the state for each car
        for pre_step in range(0, len(L0_action_id[0])): # 
            X_pseudo_L0_Id[car_id][pre_step, :, car_id] = state_info[:, car_id]
    #print(len(X_pseudo_L0_Id))
    # L-1
    L1_action_id = [None]*num_cars
    L1_Q_value = [None]*num_cars
    for car_id in range(0, num_cars):
        start = time.time()
        if L_matrix_all[car_id][car_id] > 0 or car_id == c_id:
            L1_Q_value[car_id], L1_action_id[car_id] = DecisionTree_L11_DiGraph.DecisionTree_L1(X_pseudo_L0_Id[car_id], car_id, action_space,  params, Level_ratio, L_matrix_all) # Call the decision tree function
        else:
            L1_Q_value[car_id], L1_action_id[car_id] = [0, 0, 0], [0, 0]
        end = time.time()
        # print("L1 time: ", end - start)
    X_pseudo_L1 = environment_multi_DiGraph.environment_multi(state_info, L1_action_id, params.t_step_DT, params, L_matrix_all, c_id)

    
    X_pseudo_L1_Id = [None]*num_cars
    for car_id in range(0, num_cars):
        X_pseudo_L1_Id[car_id] = X_pseudo_L1.copy()
        for pre_step in range(0, len(L1_action_id[0])):
            X_pseudo_L1_Id[car_id][pre_step, :, car_id] = state_info[:, car_id]
            
    # D-1
    if c_id == 1:
        D1_action_id =[-1]*num_cars
        D1_Q_value = [-1e6]*num_cars
        D1_Q_value_opt = [-1e6]*(num_cars-1)
        D1_action_id_opt = [-1]*(num_cars-1)
        # for car_id in range(0, self.num_cars):
        car_id = 1
        for add in range(0, num_cars-1):
            if (add ==0 and add in ego_interact_list) or (add !=0 and (add+1) in ego_interact_list):
                # print('----------->', add, ego_interact_list)
                D1_Q_value[add] = np.dot(Level_ratio[car_id*(num_cars-1)+add, 0], L0_Q_value[car_id]) + np.dot(Level_ratio[car_id*(num_cars-1)+add, 1],L1_Q_value[car_id])
                D1_Q_value_opt[add] = np.max(D1_Q_value[add])
                D1_action_id_opt[add] = np.argmax(D1_Q_value[add])
            if len(ego_interact_list) == 0:
                D1_action_id[car_id] = L0_action_id[car_id][0]
            else:
                D1_action_id[car_id] = D1_action_id_opt[np.argmax(D1_Q_value_opt)]
    
    for car_id in range(0, num_cars):
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
        for inter_car in range(0, num_cars):
            if inter_car != car_id: 
                if inter_car in ego_interact_list:
                    if (L0_action_id[inter_car][0]==L1_action_id[inter_car][0]):
                        count = count +1
                    else:                                                          
                        if Action_ID[inter_car] == L0_action_id[inter_car][0]:
                            Level_ratio[car_id*(num_cars-1)+count, 0] = Level_ratio[car_id*(num_cars-1) + count, 0] + 0.1
                        if Action_ID[inter_car] == L1_action_id[inter_car][0]:
                            Level_ratio[car_id*(num_cars-1)+count, 1] = Level_ratio[car_id*(num_cars-1) + count, 1] + 0.1

                        Level_ratio[car_id*(num_cars-1)+count,:] = Level_ratio[car_id*(num_cars-1)+count,:]/ sum(Level_ratio[car_id*(num_cars-1)+count,:])   # normalizing
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

def quit(sig, frame):
    sys.exit(0)
    
class DQN:
    def __init__(self, opt=None):
        # rospy.init_node('control_node_main', anonymous=True)
        # Algorithm Information
        self.qcar_dict = {0: 0.0, 1: 0.25, 2: -0.25, 3: 0.5, 4: -0.5}
        self.previous = {0: 0, 1: 0, 2: 0, 3: 0}
        self.integral_val = {0: 0, 1: 0, 2: 0, 3: 0}
        self.derivative_val = {0: 0, 1: 0, 2: 0, 3: 0}
        self.params = get_params.get_params()
        self.debug =False

        if opt is not None:
            self.is_hardware=True
            self.opt = opt

            self.port = 1234 #temporary
            self.clients = {
                # 1: "192.168.2.15",
                # 2: "192.168.2.13",
                # 3: "192.168.2.16",
                # 4: "192.168.2.14"
            }

            self.init_clients()

    def init_clients(self):
        for c in self.clients:
            try:
                client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                client.connect((self.clients[c], self.port))
                self.clients[c] = client
                print("Connected to QCar: {}".format(c))
                # client.send(str.encode("test"))
                
            except Exception as e:
                print("Error connecting to qcar: {}".format(c))
                print(e)

        
    def select_path(self):
        if self.debug:
            path_ls_car1 = {'strait':'5'}
            path_ls_car2 = {'strait':'8'}
            path_ls_car3 = {'strait':'6'}
            path_ls_car4 = {'strait':'7'}
        else:
            path_ls_car1 = {'right':'4', 'strait':'5', 'left':'10'}
            path_ls_car2 = {'right':'3', 'strait':'8', 'left':'9'}
            path_ls_car3 = {'right':'1', 'strait':'6', 'left':'11'}
            path_ls_car4 = {'right':'2', 'strait':'7', 'left':'12'}
            path_ls_car5 = {'right':'3', 'strait':'8', 'left':'9'}
        # car1_path = random.choice(path_ls_car1)
        # print(car1_path)
        car1_path = random.choice(list(path_ls_car1.values()))
        car2_path = random.choice(list(path_ls_car2.values()))
        car3_path = random.choice(list(path_ls_car3.values()))
        car4_path = random.choice(list(path_ls_car4.values())) 
        car5_path = random.choice(list(path_ls_car5.values()))
        print('car2:{}, car1:{}, car3:{}, car4:{}, car5:{}'.format(car2_path, car1_path, car3_path, car4_path, car5_path))
        return [car2_path, car1_path, car3_path, car4_path, car5_path]
    
    def Init_final(self, id, path_id, params, x_car, y_car):
        start_index = params.KDtrees_12[
            path_id[id]
        ].query(
            [x_car, y_car],
            1
        )[1]
        Destination = len(params.waypoints[path_id[id]])
                                                                    
        return [start_index, Destination]
        
    def get_position(self, get_path, Driver_level):
        self.traffic = self.vehicle_position(self.params, self.traffic, self.AV_cars, get_path, Driver_level)
       
        vehicle_states = np.block ([[self.traffic.x], [self.traffic.y], [self.traffic.orientation], 
                               [self.traffic.v_car], [self.traffic.AV_flag], 
                               [self.traffic.Final_x], [self.traffic.Final_y],[self.traffic.path_id], [self.traffic.Currend_index], [self.traffic.Driver_level]])
        print("Get initial position, done!")
        
        return vehicle_states
        
    def vehicle_position(self, params, traffic, AV_cars, get_path, Driver_level):       
        self.episode = 0
        self.step = 0
        path_list =[None] * self.num_cars
        # for i in range(0, self.num_cars):
        #     path_list[i] = get_path[i][1]
        
        if self.is_hardware:
            # self.start1 = [0.0, 0.0, 0.0]  # -1.16
            # self.start2 = [0.0, 0.0, 0.0]
            # self.start3 = [0.0, 0.0, 0.0]
            # self.start4 = [0.0, 0.0, 0.0]
            self.start1 = [0.18, -1.16, 0.5*np.pi]  # -1.16
            self.start2 = [-1.16, -0.18, 0.0]
            self.start3 = [1.16, 0.18, np.pi]
            self.start4 = [-0.18, 1.16, -np.pi*0.5]
            self.start5 = [-18, -0.18, 0.0]
        else:
            self.start1 = [0.18, -1.16, 0.5*np.pi]  # -1.16
            self.start2 = [-1.16, -0.18, 0.0]
            self.start3 = [1.16, 0.18, np.pi]
            self.start4 = [-0.18, 1.16, -np.pi*0.5]
            self.start5 = [-18, -0.18, 0.0]
        
        # Init positions
        for id in range(0, self.num_cars):
            if id ==0:
                x_car = self.start2[0]
                y_car = self.start2[1]
                v_car = 0
                orientation_car = self.start2[2]
                Dr_level = Driver_level[0]
            elif id == 1: # AV
                x_car = self.start1[0]
                y_car = self.start1[1]
                v_car = 0
                orientation_car = self.start1[2]
                Dr_level = Driver_level[1]
            elif id == 2:
                x_car = self.start3[0]
                y_car = self.start3[1]
                v_car = 0
                orientation_car = self.start3[2]
                Dr_level = Driver_level[2]
            elif id == 3:
                x_car = self.start4[0]
                y_car = self.start4[1]
                v_car = 0
                orientation_car = self.start4[2]
                Dr_level = Driver_level[3]
            elif id == 4:
                x_car = self.start5[0]
                y_car = self.start5[1]
                v_car = 0
                orientation_car = self.start5[2]
                Dr_level = Driver_level[4]
            # elif id == 5:
            #     x_car = self.robotstate6[0]
            #     y_car = self.robotstate6[1]
            #     v_car = self.robotstate6[2]
            #     orientation_car = self.robotstate6[4]
            #     Dr_level = Driver_level[5]
                        
            #v_car = np.random.uniform(v_min, v_max)
            rear_x = x_car - math.cos(orientation_car) * 0.128
            rear_y = y_car - math.sin(orientation_car) * 0.128
            
            get_init_final = self.Init_final(id, get_path, params, rear_x, rear_y)
            init_index = get_init_final[0]
            final_index =  get_init_final[1]
            path_id = get_path[id]
        
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
        if self.is_hardware:
            poses = self.opt.poses
            #print("POSES")
            #print(poses[1])

            for id in range(0, self.num_cars):
                # print(poses)
                pose = poses[id+1][0]

                x_car = pose[0]
                y_car = pose[1]
                v_car = pose[2]
                orientation_car = pose[3]


                #v_car = np.random.uniform(v_min, v_max)
                rear_x = x_car - math.cos(orientation_car) * 0.128
                rear_y = y_car - math.sin(orientation_car) * 0.128
                path_id = X_old[7][id]
                path_id = str(int(path_id))
                current_index = self.params.KDtrees_12[path_id].query([rear_x, rear_y],1)[1]
                
                X_old[0][id] = x_car
                X_old[1][id] = y_car
                X_old[2][id] = orientation_car
                X_old[3][id] = v_car
                X_old[8][id] = current_index
                # print('speed:{}'.format(v_car))
        else:    
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
                
                        
                #v_car = np.random.uniform(v_min, v_max)
                rear_x = x_car - math.cos(orientation_car) * 0.128
                rear_y = y_car - math.sin(orientation_car) * 0.128
                
                X_old[0][id] = x_car
                X_old[1][id] = y_car
                X_old[2][id] = orientation_car
                X_old[3][id] = v_car# X_old[3][id] # MODIFIY THIS LATER!!!
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
        self.num_Human = self.num_cars - self.params.num_AV
        self.params.num_Human = self.num_Human
        self.num_lanes = self.params.num_lanes
       
        self.outdir = self.params.outdir
        self.render = self.params.render
        self.max_step = 80#0.2-50
        # Initial guess for the level ratio (0 1)
        self.Level_ratio = np.array([[0.2, 0.8]])
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

        self.fig_sim = plt.figure(1, figsize=(6, 6))
        # self.fig_0 = plt.figure(0, figsize=(6, 3))
        # self.fig_2 = plt.figure(2, figsize=(6, 3))
        # self.fig_3 = plt.figure(3, figsize=(6, 3))
        self.data = {0:[], 1:[], 2:[], 3:[]}
        self.re_time = {0: 0.0, 1:0.0, 2:0.0, 3:0.0}

        # Create output folder
        '''if not(os.path.exists(self.outdir)):
                os.mkdir(self.outdir)

        # pick a simulation case
        # 0 - Aggressive
        # 1 - Adaptive
        # 2 - Conservative
        self.params.sim_case = 1

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
        #self.state_info = [self.X_old]*(self.num_cars+1)

    def get_id_Lmatrix(self):
        state_info = self.veh_pos_realtime(self.X_old)
        others_list = [0, 1, 2, 3]
        waiting_list = []
        Matrix_buffer = {0:[], 1:[], 2:[], 3:[]}
        c_id = 1
        waiting_list.append[c_id]
        L_matrix_all, ego_interact_list = self.laplacian_metrix(state_info, self.params, self.Level_ratio, c_id)
        other_car_id = list(set(others_list) - set(ego_interact_list))
        try:
            other_car_id.remove(c_id)
        except:
            pass
        Matrix_buffer[c_id] = L_matrix_all
        while len(other_car_id) > 0:
            L_matrix_all, ego_interact_list = self.laplacian_metrix(state_info, self.params, self.Level_ratio, other_car_id[0])
            Matrix_buffer[other_car_id[0]] = L_matrix_all
            waiting_list.append[other_car_id[0]]
            try:
                other_car_id.remove(other_car_id[0])
            except:
                pass
        return waiting_list, Matrix_buffer

    def globle_update(self, Action_ID, L_matrix_all=0, c_id=0):
        # plot_sim_DiGraph.plot_sim(self.X_old, self.params, self.step, self.Level_ratio, self.fig_sim)
        # plot_level_ratio.plot_level_ratio(self.Level_ratio_history, ego_car_id, opp_car_id, self.params, self.step, self.episode, self.max_step, self.fig_0)
        X_new, R = Environment_Multi_Sim_DiGraph.Environment_Multi_Sim(self.X_old, Action_ID, 0.1, self.params, L_matrix_all, c_id)#self.params.t_step_Sim
        self.X_old = X_new
        
    def get_id_Lmatrix(self):
        if debug == 0:
            state_info = self.veh_pos_realtime(self.X_old)
        else:
            # print(self.X_old)
            state_info = self.X_old
        others_list = [0, 1, 2, 3]
        waiting_list = []
        Matrix_buffer = {0:0, 1:0, 2:0, 3:0}
        command_carls= {0:[], 1:[], 2:[], 3:[]}
        exiting_ls = []
        c_id = 1
        waiting_list.append(c_id)
        L_matrix_all, ego_interact_list = self.laplacian_metrix(state_info, self.params, self.Level_ratio, c_id)
        Matrix_buffer[c_id] = {'matrix':L_matrix_all, 'interact_list':ego_interact_list}
        if len(ego_interact_list)==0:
            ego_interact_list = [c_id]
        other_car_id = list(set(others_list) - set(ego_interact_list))   
        command_carls[c_id] = ego_interact_list
        exiting_ls.append(ego_interact_list)
        while len(other_car_id) > 0:
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
            exiting_ls.append(ego_interact_list)
            
        return waiting_list, Matrix_buffer, command_carls, state_info
 

    def Get_LevelK_action_DiGraph(self, c_id, others_list, existing_list, Action_ID):
        t0 = time.time()
        # Animation plots
        # if c_id ==1:
        #     plot_sim_DiGraph.plot_sim(self.X_old, self.params, self.step, self.Level_ratio, self.fig_sim)
        # Plot level history
        self.state_info = self.X_old#self.veh_pos_realtime(self.X_old)
        # if c_id ==1:
        #     print(self.state_info)
        # self.state_info = self.X_old
        # print(self.X_old)
        # self.state_info = self.X_old
        L_matrix_all, ego_interact_list = self.laplacian_metrix(self.state_info, self.params, self.Level_ratio, c_id)
        # print(L_matrix_all)
        # time.sleep(0.3)
        
        if c_id == 1:
            self.Level_ratio_history[self.episode, self.step, :, :] = self.Level_ratio
        
        # L-0  decision tree
        L0_action_id =  [None]*self.num_cars    # set the action sizes according to car numbers
        L0_Q_value =  [None]*self.num_cars      # set the Q value sizes according to car numbers
        # print(self.state_info)
        for car_id in range(0, self.num_cars): # find the action for level 0 decision 
            start = time.time()
            if L_matrix_all[car_id][car_id] > 0 or car_id == c_id:
                
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
                # if car_id ==1:
                #     print(L1_Q_value[1])
                #     print(max(L1_Q_value[1]), np.argmax(L1_Q_value[1]))
                #     print('-------------------------')
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
            D1_action_id =[-1]*self.num_cars
            D1_Q_value = [-1e6]*self.num_cars
            D1_Q_value_opt = [-1e6]*(self.num_cars-1)
            D1_action_id_opt = [-1]*(self.num_cars-1)
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
        
        for car_id in range(0, self.num_cars):
            if car_id in ego_interact_list or car_id == c_id:
                if car_id not in existing_list:
                    if self.state_info[9][car_id]==0:
                        Action_ID [car_id] = L0_action_id[car_id][0]
                    elif self.state_info[9][car_id]==2:
                        Action_ID [car_id] = L1_action_id[car_id][0]
                    else:
                        Action_ID [car_id] = D1_action_id[car_id]
       
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
        # if c_id == 1:
        #     # for car_id in range(0, self.num_cars):
        #     count = 0 
        #     car_id = 1
        #     for inter_car in range(0, self.num_cars):
        #         if inter_car != car_id and inter_car in ego_interact_list:
        #             if (L0_action_id[inter_car][0]==L1_action_id[inter_car][0]):
        #                 count = count +1
        #             else:                                                          
        #                 if Action_ID[inter_car] == L0_action_id[inter_car][0]:
        #                     self.Level_ratio[car_id*(self.num_cars-1)+count, 0] = self.Level_ratio[car_id*(self.num_cars-1) + count, 0] + 0.05
        #                 if Action_ID[inter_car] == L1_action_id[inter_car][0]:
        #                     self.Level_ratio[car_id*(self.num_cars-1)+count, 1] = self.Level_ratio[car_id*(self.num_cars-1) + count, 1] + 0.05

        #                 self.Level_ratio[car_id*(self.num_cars-1)+count,:] = self.Level_ratio[car_id*(self.num_cars-1)+count,:]/ sum(self.Level_ratio[car_id*(self.num_cars-1)+count,:])   # normalizing
        #                 count = count+1
        #     # print('Level_ratio', self.Level_ratio[4:8])


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

        existing_list = existing_list + ego_interact_list+[c_id]

        if len(self.other_car_id) == 0:
            plot_sim_DiGraph.plot_sim(self.X_old, self.params, self.step, self.Level_ratio, self.fig_sim)
            # print(Action_ID)
            X_new, R = Environment_Multi_Sim_DiGraph.Environment_Multi_Sim(self.state_info, Action_ID, self.params.t_step_Sim, self.params, L_matrix_all, c_id)
            self.X_old = X_new
            
  
        return Action_ID, self.X_old, self.other_car_id, existing_list
    
    def Get_LevelK_action_DiGraph_subprocess(self, state_info, Matrix_buffer, command_carls, c_id):
        Action_ID = [-1.0]*self.num_cars
        L_matrix_all = Matrix_buffer[c_id]['matrix']
        ego_interact_list = Matrix_buffer[c_id]['interact_list']
        self.state_info = state_info
        # L_matrix_all, ego_interact_list = self.laplacian_metrix(self.state_info, self.params, self.Level_ratio, c_id)
        # print(ego_interact_list)
        # print('Car_ID:{}|||ego_interact_list:{}'.format(c_id, ego_interact_list))
        # print('-----------------------------')
        # time.sleep(1)
        if c_id == 1:
            self.Level_ratio_history[self.episode, self.step, :, :] = self.Level_ratio
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
            # D1_action_id =[None]*self.num_cars
            # D1_Q_value = [None]*self.num_cars
            # D1_Q_value_opt = [None]*(self.num_cars-1)
            # D1_action_id_opt = [None]*(self.num_cars-1)
            D1_action_id =[-1]*self.num_cars
            D1_Q_value = [-1e6]*self.num_cars
            D1_Q_value_opt = [-1e6]*(self.num_cars-1)
            D1_action_id_opt = [-1]*(self.num_cars-1)
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
    

    def return_action(self):
        new_algorithm = True
        multi_proc = True
        if new_algorithm == True:
            if multi_proc:
                
                waiting_list, Matrix_buffer, command_carls, state_info = self.get_id_Lmatrix()
                # print('waiting_list', waiting_list)
                GTMPC_actions_ID = [0]*self.num_cars
                self.Level_ratio_history[self.episode, self.step, :, :] = self.Level_ratio
                
                fn = partial(LevelK_action_DiGraph_subprocess, state_info, Matrix_buffer, command_carls, self.Level_ratio, self.params)
                start = time.time()
                Action_ls = []
                trust_ls = []
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    results = [executor.submit(fn, var) for var in waiting_list] 
                    for f in concurrent.futures.as_completed(results):
                        Action_ls.append(f.result()[0])
                        trust_ls.append(f.result()[1])
                        # print(type(f.result()))
                # print(Action_ls)
                finish = time.time()
                # print('Finished in {} second(s)'.format(round(finish-start,2 )))
                for i in trust_ls:
                    if len(i) !=0:
                        self.Level_ratio = i
                print(self.Level_ratio)
                for item in Action_ls:
                    idx_ls = []
                    for i in item:
                        if i != -1:
                            idx = item.index(i)
                            GTMPC_actions_ID[idx] = i
                            if idx not in idx_ls:
                                idx_ls.append(idx)
                            if len(idx_ls) == 4:
                                break
                print(GTMPC_actions_ID)
                            
                # for i in waiting_list:
                #     for item in command_carls[i]:
                #         GTMPC_actions_ID[item] = Action_ls[item]
                # for i in waiting_list:
                #     # Actions = self.Get_LevelK_action_DiGraph_subprocess(state_info, Matrix_buffer, command_carls, i)
                #     for item in command_carls[i]:
                #         GTMPC_actions_ID[item] = Actions[item]
                self.globle_update(GTMPC_actions_ID)
                X_old = self.X_old
                self.step += 1

                    # print(GTMPC_actions_ID)
            else:
                ego_id = 1
                # GTMPC_actions_ID, X_old, others_list = self.env.Get_LevelK_action_DiGraph_ego(ego_id
                others_list = [0, 1, 2, 3]
                Interaction_list = []
                GTMPC_actions_ID = [None]*self.num_cars
                t1 = time.time()

                '''param_list = [1,2,3,4]
                fn = partial(test, a=1, b=2)
                with mp.Pool(processes=4) as pool:
                    result = pool.starmap(fn, param_list)

                def test(param_list, a, b):
                    #do something with param_list

                fn = partial(self.Get_LevelK_action_DiGraph, others_list=others_list, Interaction_list=Interaction_list)
                with mp.Pool(processes=4) as pool:
                    pool.starmap(fn, others_list[0])'''
                GTMPC_actions_ID, X_old, others_list, Interaction_list = self.Get_LevelK_action_DiGraph(ego_id, others_list, Interaction_list, GTMPC_actions_ID)
                print('ego_id:{}'.format(ego_id))
                # print('{} Hz'.format(1/(time.time() - t1)))
                while len(others_list) > 0:
                    GTMPC_actions_ID, X_old, others_list, Interaction_list = self.Get_LevelK_action_DiGraph(others_list[0], others_list, Interaction_list, GTMPC_actions_ID)
                    # print('other_id:{}'.format(others_list[0]))
                    # print('Other the rest of others_list: {}----Action:{}'.format(others_list, GTMPC_actions_ID))
                # print('---------------------------------')
        else:
            t1 = time.time()
            GTMPC_actions_ID, X_old = self.Get_LevelK_action()
            # print('{} Hz'.format(1/(time.time()- t1)))
            print('{} Hz'.format(1/(time.time() - t1)))
        return GTMPC_actions_ID, X_old
            
    def laplacian_metrix(self, X_old, params, Level_ratio, car_id):
        t1 = time.time()
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
                        
    def Get_LevelK_action(self):
        t0 = time.time()
        # Animation plots
        # print(self.X_old)
        plot_sim.plot_sim(self.X_old, self.params, self.step, self.Level_ratio, self.fig_sim)
        # Plot level history
        # self.X_old = self.veh_pos_realtime(self.X_old)
        # print(self.X_old)
        # car_id = 1
        # L_matrix_all, ego_interact_list = self.laplacian_metrix(self.X_old, self.params, self.Level_ratio, car_id)
      
        self.Level_ratio_history[self.episode, self.step, :, :] = self.Level_ratio
        
        # L-0
        L0_action_id =  [None]*self.num_cars    # set the action sizes according to car numbers
        L0_Q_value =  [None]*self.num_cars      # set the Q value sizes according to car numbers
        
        for car_id in range(0, self.num_cars): # find the action for level 0 decision 
            start = time.time()
            L0_Q_value[car_id], L0_action_id[car_id] = decisiontree_l01.decisiontree_l0(self.X_old, car_id, self.action_space, self.params, self.Level_ratio)
            # print(L0_Q_value[car_id], L0_action_id[car_id])
            #L0_Q_value[car_id], L0_action_id[car_id] = [0, 0, 0], [0, 0]
            end = time.time()
            # if car_id == 1:
            #     print("L0 time: ", end - start)
        # print('L0_action_id', L0_action_id)
        # print('L0_Q_value', L0_Q_value)
        #predict the state for each car with level-0 policy in next 2 steps 
        X_pseudo_L0 = environment_multi.environment_multi(self.X_old, L0_action_id, self.t_step_DT, self.params)
     
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
            L1_Q_value[car_id], L1_action_id[car_id] = DecisionTree_L11.DecisionTree_L1(X_pseudo_L0_Id[car_id], car_id, self.action_space,  self.params, self.Level_ratio) # Call the decision tree function
            end = time.time()
            # print("L1 time: ", end - start)
        X_pseudo_L1 = environment_multi.environment_multi(self.X_old, L1_action_id, self.t_step_DT, self.params)
       
        
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
                D1_Q_value[add] = np.dot(self.Level_ratio[car_id*(self.num_cars-1)+add, 1], L0_Q_value[car_id]) + np.dot(self.Level_ratio[car_id*(self.num_cars-1)+add, 0],L1_Q_value[car_id])
                # print(self.Level_ratio[car_id*(self.num_cars-1)+add, 0])
                # print(L0_Q_value[car_id])
                D1_Q_value_opt[add] = np.max(D1_Q_value[add])
                D1_action_id_opt[add] = np.argmax(D1_Q_value[add])
           D1_action_id[car_id] = D1_action_id_opt[np.argmax(D1_Q_value_opt)]
                

        # Controller selection for each cars
        # AV: Auto controller, HV: Level-K contoller
        Action_id = [None]*self.num_cars
        for car_id in range(0,self.num_cars):
            if self.X_old[4][car_id]==1:
                # Action_id[car_id] = L1_action_id[car_id][0]
                Action_id[car_id] = D1_action_id[car_id]
                # Action_id[car_id] = L1_action_id[car_id][0]
            elif self.X_old[4][car_id]==2:
                # Action_id[car_id] = L0_action_id[car_id][0]
                Action_id[car_id] = L1_action_id[car_id][0]
                # Action_id[car_id] = D1_action_id[car_id]
            else:
                Action_id[car_id] = L0_action_id[car_id][0]
        #print(Action_id)

        # Level estimation update
        for car_id in range(0, self.num_cars):
            count = 0 
            for inter_car in range(0, self.num_cars):
                if inter_car != car_id:
                   if (L0_action_id[inter_car][0]==L1_action_id[inter_car][0]):
                       count = count +1
                   else:                                                          
                        if Action_id[inter_car] == L0_action_id[inter_car][0]:
                           self.Level_ratio[car_id*(self.num_cars-1)+count, 0] = self.Level_ratio[car_id*(self.num_cars-1) + count, 0] + 0.1
                        if Action_id[inter_car] == L1_action_id[inter_car][0]:
                           self.Level_ratio[car_id*(self.num_cars-1)+count, 1] = self.Level_ratio[car_id*(self.num_cars-1) + count, 1] + 0.1

                        self.Level_ratio[car_id*(self.num_cars-1)+count,:] = self.Level_ratio[car_id*(self.num_cars-1)+count,:]/ sum(self.Level_ratio[car_id*(self.num_cars-1)+count,:])   # normalizing
                        count = count+1
        
        #print(self.Level_ratio)

        # State update
        X_new, R = Environment_Multi_Sim.Environment_Multi_Sim(self.X_old, Action_id, self.params.t_step_Sim, self.params)
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
        # plot_level_ratio.plot_level_ratio(self.Level_ratio_history, ego_car_id, 1, self.params, self.step, self.episode, self.max_step, self.fig_0)
        # plot_level_ratio.plot_level_ratio(self.Level_ratio_history, ego_car_id, 2, self.params, self.step, self.episode, self.max_step, self.fig_2)
        # plot_level_ratio.plot_level_ratio(self.Level_ratio_history, ego_car_id, 3, self.params, self.step, self.episode, self.max_step, self.fig_3)
        # # Timer
        t1 = time.time()
        time_per_step = t1 - t0
        print('total', time_per_step)


        # Completion check
        if sum(self.complete_flag[self.episode, :]) == self.num_cars:
            pass
        
        self.step += 1
    
        return Action_id, self.X_old
    
    def SteeringCommand(self, X_old, car_id, params):
        lad = 0.0  # look ahead distance accumulator
        k = 3.5
        
        # use real time info
        if self.is_hardware:
            poses = self.opt.poses
            pose = poses[car_id+1][0]
            x = pose[0]
            y = pose[1]
            v = pose[2]
            yaw = pose[3]
            # print('x:{}, y:{}, v:{}, yaw:{}'.format(x, y, v, yaw))
        else:
            x = X_old[0][car_id]
            y = X_old[1][car_id]
            yaw = X_old[2][car_id]
            v = X_old[3][car_id]


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
    
    def speed(self, pos, action, params, car_id):
        self.cmd = [0, 0]
        t1 = time.time()
        if self.is_hardware:
            poses = self.opt.poses
            pose = poses[car_id+1][0]
            current_v = pose[2]
            # print('current_v',current_v)
        else:
            current_v = pos[2][car_id]
        # if self.qcar_dict[action] < 0:
        #     speed = 0.0
        # elif self.qcar_dict[action] > 0:
        #     speed = 0.6
        # else:
        #     speed = self.robotstate1[2]
        #     accel = 0
        # speed = pos[3][1]*
        delta_t = t1 - self.previous[car_id]
        if delta_t > 1.0:
            delta_t = 0.1
        #speed = self.robotstate1[2] + self.qcar_dict[action]*delta_t

        speed = current_v + self.qcar_dict[action]*delta_t
        self.previous[car_id] = t1
        if speed >0.6:
            speed = 0.6
        elif speed < 0:
            speed = 0
        if pos[8][car_id]+25 > pos[6][car_id]:
            speed = 0
        #PID###################################################
        desried_v = speed
        kp =0.3#H0.5 #G0.03#--vel--0.5#0.05-vel-0.3-0.4#0.15
        ki = 0.08#H0.0 #G0.08#0.1#0.001#0.01#.005#0.1
        kd = 0.01#H0#G0.01#.05#0.04#0.06
        delta_v = desried_v - current_v #error term
        integral = self.integral_val[car_id] + (delta_v * delta_t) #intergral
        derivative = (delta_v - self.derivative_val[car_id]) / delta_t #derivative
        rst = kp * delta_v + ki * integral + kd * derivative #PID 2.6*desired_v
        pid_output = rst
        ###########################
        self.accel_current = self.qcar_dict[action]
        # if desried_v > 0.01 and current_v<0.01:
        #     self.cmd[0] = 0.06
        # else:
        self.cmd[0] = 0.15*desried_v + pid_output#speed*0.5
        if car_id ==1:
            print('desired_v:{}, cmd:{}'.format(desried_v, current_v))
        self.cmd[1] = self.SteeringCommand(pos, car_id, params)#self.loop()
        self.re_time[car_id] +=  delta_t
        self.data[car_id].append([self.re_time[car_id], desried_v, current_v])
        return self.cmd
        # print(speed, self.robotstate1[2])

    def main(self):
        print("Main begins.")
        self.step_for_newenv = 0

        get_path = self.select_path()

        Driver_level = [2, 1, 2, 2] #0-l0, 1-adp,3 conservative, 
        # Driver_level = [0, 0, 0, 0]
        self.Initial_GTMPC(get_path, Driver_level)

        while True:
 
            GTMPC_actions_ID, X_old = self.return_action()
            # self.return_action()
            # time.sleep(0.5)
            self.step_for_newenv +=1
            # try:
            # cmd1 = self.speed(X_old, GTMPC_actions_ID[0], self.params, 0)
            # cmd2 = self.speed(X_old, GTMPC_actions_ID[1], self.params, 1) #Ego car
            # cmd3 = self.speed(X_old, GTMPC_actions_ID[2], self.params, 2)
            # cmd4 = self.speed(X_old, GTMPC_actions_ID[3], self.params, 3)
        
            # cmd1_json = {
            #     "speed": cmd1[0],
            #     "angle": cmd1[1]
            # }
            # cmd2_json = {
            #     "speed": cmd2[0],
            #     "angle": cmd2[1]
            # }
            # cmd3_json = {
            #     "speed": cmd3[0],
            #     "angle": cmd3[1]
            # }
            # cmd4_json = {
            #     "speed": cmd4[0],
            #     "angle": cmd4[1]
            # }

            # cmd_json = {
            #     "speed": 0.0,
            #     "angle": 0.0
            # }
            # CMD = [cmd1_json, cmd2_json, cmd3_json, cmd4_json, cmd_json]
            # print('cmd1', cmd1)
            # print('cmd2', cmd2)
            # print('cmd3', cmd3)
            # print('cmd4', cmd4)
            # for id in range(0, self.num_cars):
            #     if X_old[8][id]+20 < X_old[6][id]:
            #         self.clients[id+1].send(str.encode(json.dumps(CMD[id])))
            #     else:
            #         self.clients[id+1].send(str.encode(json.dumps(cmd_json)))
            # except:
            #     pass
            # print('------------------------------')
            # client.send(str.encode(cmd1))
            cmd_json = {
                    "speed": 0.0,
                    "angle": 0.0
                }
 

            # self.clients[2].send(str.encode(json.dumps(cmd2_json)))
            # self.clients[3].send(str.encode(json.dumps(cmd3_json)))
            # self.clients[4].send(str.encode(json.dumps(cmd4_json)))
            #self.clients[2].send(cmd2)
            #self.clients[3].send(cmd3)
            #self.clients[4].send(cmd4)

            # print(GTMPC_actions_ID)
            # link1.speed(X_old, GTMPC_actions_ID[1], params)
            # link2.speed(X_old, GTMPC_actions_ID[0], params)
            # link3.speed(X_old, GTMPC_actions_ID[2], params)
            # link4.speed(X_old, GTMPC_actions_ID[3], params)
            # link5.speed(X_old, GTMPC_actions_ID[4], params)
              
            # Reset environment
            if self.step_for_newenv == self.max_step:
                data = np.array(self.data[1])
             
                # for id in range(0, self.num_cars):
                #     plt.figure(figsize=(8, 6))
                #     data = np.array(self.data[id])
                #     plt.plot(data[:, 0], data[:,1], linestyle='solid', c='k')
                #     plt.plot(data[:,0], data[:, 2], linestyle='solid', c='r')
                #     plt.xlabel('time (sec)')
                #     plt.ylabel('Reference speed m/s')
                # plt.show()
                
                # if debug == 0:
                #     self.clients[1].send(str.encode(json.dumps(cmd_json)))
                #     self.clients[2].send(str.encode(json.dumps(cmd_json)))
                #     self.clients[3].send(str.encode(json.dumps(cmd_json)))
                #     self.clients[4].send(str.encode(json.dumps(cmd_json)))

                check_state = eval(input('Next round of test? (1=Ready / 2=Wait): '))
                if check_state:
                    # stop all vehicles
                    print('new episode took', self.step_for_newenv)
                    self.step_for_newenv = 0
                    get_path = self.select_path()
                    Driver_level = [2, 1, 2, 2]
                    # Driver_level = [0, 0, 0, 0]
            
                    self.Initial_GTMPC(get_path, Driver_level)


    def stream(command, client):
        pass

if __name__ == '__main__':
    
    signal.signal(signal.SIGINT, quit)
    opt = OptitrackParser()
    agent = DQN(opt)
    # agent = DQN()
    agent.main()

    # with concurrent.futures.ProcessPoolExecutor() as executor:
    #     vars = [2, 2]
    #     result = executor.map(do_something, vars)
    # vars = [2, 2, 2, 2, 2, 2]
    # with mp.Pool(len(vars)) as pool:
    #     start = time.time()
    #     pool.map(do_something, vars)
    #     finish = time.time()
    #     print('Finished in {} second(s)'.format(round(finish-start,2 )))

    # while step_for_newenv<3:

    #     agent.return_action()
    #     step_for_newenv +=1

        # # Reset environment
        # if self.step_for_newenv == self.max_step:
        #     step_for_newenv 