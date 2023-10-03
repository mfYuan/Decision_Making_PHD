#!/usr/bin/env python
# -*- coding: utf-8 -*-


import signal
import sys
import random
import numpy as np
import time
import os
sys.path.append(os.path.join('/home/sdcnlab025/ROS_test/four_qcar/src/tf_pkg/safe_RL/MPC_RL/GT_MPC'))#'''E:\Mingfeng\safe_RL\MPC_RL'''
import get_params

sys.path.append(os.path.join('/home/sdcnlab025/ROS_test/four_qcar/src/tf_pkg/safe_RL/MPC_RL/GT_MPC/Sim_paper_plot'))#'''E:\Mingfeng\safe_RL\MPC_RL'''



# import plot_sim_DiGraph
# import plot_level_ratio_DiGraph

# import plot_graph_offline
# import plot_level_ratio
# import switching_tp
import paper_plt_offline
# import plot_sim_DiGraph_new_offline
import plot_sim_DiGraph_new_offline_copy
import plot_graph_offline_copy
# import plot_sim


# import traff
import numpy.matlib
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
import math


debug=0
num_cars =4
# action_space = np.array([[0, 1, 2, 3, 4]])
action_space = np.array([[0, 1, 2]])



def quit(sig, frame):
    sys.exit(0)
N = 1.0

class plt_figs:
    def __init__(self):
        # rospy.init_node('control_node_main', anonymous=True)
        # Algorithm Information
        
        self.qcar_dict = {0: 0.0, 1: 0.25/N, 2: -0.25/N, 3: 0.5/N, 4: -0.5/N}
        self.previous = {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0}
        self.integral_val = {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0}
        self.derivative_val = {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0}
        self.params = get_params.get_params()
        self.debug =False
        self.episode = 0
        
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
        self.max_step = 400#0.2-50
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
#E:/Mingfeng/safe_RL/MPC_RL/Images/Record/paper/New round/333201020005/txtfile
    def main(self):
        # path_tra = 'E:/Mingfeng/safe_RL/MPC_RL/Images/Record/paper/New round/Straight/222221200001/txtfile/'
        # path = 'E:/Mingfeng/safe_RL/MPC_RL/Images/Record/paper/New round/Straight/222221200001/txtfile/'
        #traditional data
        path_tra = '/home/sdcnlab025/ROS_test/four_qcar/src/tf_pkg/safe_RL/MPC_RL/Images/Recorded1/Scalable_result/111011201020020000001a_7cars/txtfile/'
        #our data
        path = '/home/sdcnlab025/ROS_test/four_qcar/src/tf_pkg/safe_RL/MPC_RL/Images/Recorded1/Scalable_result/111011201020020000001_7cars_video/txtfile/'
        # path_tra = 'E:/Mingfeng/safe_RL/MPC_RL/Images/Record/paper/New round/Simulation/11/txtfile/'
        # #our data
        # path = 'E:/Mingfeng/safe_RL/MPC_RL/Images/Record/paper/New round/Simulation/11/txtfile/'
        path_ls = path+'figs/'
        isExist = os.path.exists(path_ls)
        if not isExist:
            os.makedirs(path_ls)
        debug = 1
        
        print("Main begins.")
        self.step_for_newenv = 0
        car_id = [2, 1, 3, 4]
        #plot graph annimate
        file_data4 = path + 'matrix.txt'
        read_data4 = np.array(np.loadtxt(file_data4))
        load_original_matrix = read_data4.reshape(read_data4.shape[0], read_data4.shape[1] // self.params.num_cars, self.params.num_cars)
        print('matrix', load_original_matrix.shape)
        Num_max = load_original_matrix.shape[0]

        #plot running time 
        file_data1 = path + 'run0.txt'
        read_data1 = np.array(np.loadtxt(file_data1))
        file_data22 = path_tra + 'run0.txt'
        read_data22 = np.array(np.loadtxt(file_data22))
        paper_plt_offline.plot_runtime(read_data1, read_data22, path_ls)

        #plot trust levels
        file_data2 = path + 'ratio.txt'
        read_data2 = np.array(np.loadtxt(file_data2))
        read_data2 = np.array(read_data2)
        load_original_arr = read_data2.reshape(read_data2.shape[0], read_data2.shape[1] // 2, 2)
        print('ratio',load_original_arr.shape)
        # paper_plt_offline.plot_level_ratio(read_data1, load_original_arr, path_ls, self.params, Num_max)

        #plot pos annimate
        file_data3 = path + 'pos.txt'
        read_data3 = np.array(np.loadtxt(file_data3))
        read_data3 = np.array(read_data3)
        load_original_pos = read_data3.reshape(read_data3.shape[0], read_data3.shape[1] // self.params.num_cars, self.params.num_cars)
        print('pos', load_original_pos.shape)

        
        
        pre_time = 0
        counter = 0
        for k in range(len(load_original_matrix)):
            step =  read_data1[k][0]- pre_time
            pre_time = read_data1[k][0]
            # print('---level_ratio---',level_ratio)
            lr = load_original_arr[counter, :, :]
            pos = load_original_pos[counter, :, :]
            
            # Dynamic graph
            fig_graph = 0
            fig_sim = 0
            # plot_graph_offline_copy.plot_graph(pos, load_original_matrix[counter], fig_graph, counter, path_ls)
            # plot_sim_DiGraph_new_offline_copy.plot_sim(pos, self.params, step, lr, fig_sim, counter, path_ls)
      
            counter +=1

        #Progress
        file_data5 = path + 'progress.txt'
        read_data5 = np.array(np.loadtxt(file_data5))

        # paper_plt_offline.plot_progress(read_data1, read_data5, path_ls, self.params)
        
                
if __name__ == '__main__':
    signal.signal(signal.SIGINT, quit)
    agent = plt_figs()
    agent.main()

