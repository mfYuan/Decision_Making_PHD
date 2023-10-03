#!/usr/bin/env python
# -*- coding: utf-8 -*-
import rospy
import signal
import sys
import random
import numpy as np
import time
sys.path.append('/home/sdcnlab025/ROS_test/four_qcar/src/tf_pkg/safe_RL/MPC_RL/GT_MPC')
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

def quit(sig, frame):
    sys.exit(0)
    
class DQN:
    def __init__(self):
        rospy.init_node('control_node_main', anonymous=True)
        # Algorithm Information
        self.qcar_dict = {0: 0.0, 1: 0.25, 2: -0.25, 3: 0.5, 4: -0.5}
        self.previous = 0
        self.params = get_params.get_params()
        self.debug =False
        
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
        car1_path = random.choice(path_ls_car1.items())
        car2_path = random.choice(path_ls_car2.items())
        car3_path = random.choice(path_ls_car3.items())
        car4_path = random.choice(path_ls_car4.items()) 
        car5_path = random.choice(path_ls_car5.items())
        print('car2:{}, car1:{}, car3:{}, car4:{}, car5:{}'.format(car2_path, car1_path, car3_path, car4_path, car5_path))
        return [car2_path, car1_path, car3_path, car4_path, car5_path]
    
    def Init_final(self, id, path_id, params, x_car, y_car):
        print(path_id)
        print(id)
        start_index = params.KDtrees_12[path_id[id][1]].query([x_car, y_car],1)[1]
        Destination = len(params.waypoints[path_id[id][1]])
                                                                    
        return [start_index, Destination]
        
    def get_position(self, get_path, Driver_level):
        self.traffic = self.vehicle_position(self.params, self.traffic, self.AV_cars, get_path, Driver_level)
       
        vehicle_states = np.block ([[self.traffic.x], [self.traffic.y], [self.traffic.orientation], 
                               [self.traffic.v_car], [self.traffic.AV_flag], 
                               [self.traffic.Final_x], [self.traffic.Final_y],[self.traffic.Final_orientation], [self.traffic.Currend_index], [self.traffic.Driver_level]])
        print("Get initial position, done!")
        
        return vehicle_states
        
    def vehicle_position(self, params, traffic, AV_cars, get_path, Driver_level):       
        self.episode = 0
        self.step = 0
        path_list =[None] * self.num_cars
        # for i in range(0, self.num_cars):
        #     path_list[i] = get_path[i][1]
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
        self.max_step = 50#0.2-50
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

    def Get_LevelK_action_DiGraph(self, c_id, others_list, existing_list, Action_ID):
        t0 = time.time()
        # Animation plots
        # if c_id ==1:
        #     plot_sim_DiGraph.plot_sim(self.X_old, self.params, self.step, self.Level_ratio, self.fig_sim)
        # Plot level history
        # self.state_info = self.veh_pos_realtime(self.X_old)
        self.state_info = self.X_old
        # print(self.X_old)
        # self.state_info = self.X_old
        L_matrix_all, ego_interact_list = self.laplacian_metrix(self.state_info, self.params, self.Level_ratio, c_id)
        # if c_id == 1:
        #     print(L_matrix_all)
        #     time.sleep(0.5)
        
        if c_id == 1:
            self.Level_ratio_history[self.episode, self.step, :, :] = self.Level_ratio
        
        # L-0  decision tree
        L0_action_id =  [None]*self.num_cars    # set the action sizes according to car numbers
        L0_Q_value =  [None]*self.num_cars      # set the Q value sizes according to car numbers
        
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
                    # print(L1_Q_value[1])
                    # print(max(L1_Q_value[1]), np.argmax(L1_Q_value[1]))
                    # print('-------------------------')
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

        existing_list = existing_list + ego_interact_list

        if len(self.other_car_id) == 0:
            # print(Action_ID)
            X_new, R = Environment_Multi_Sim_DiGraph.Environment_Multi_Sim(self.state_info, Action_ID, self.params.t_step_Sim, self.params, L_matrix_all, c_id)
            self.X_old = X_new
            plot_sim_DiGraph.plot_sim(self.X_old, self.params, self.step, self.Level_ratio, self.fig_sim)
  
        return Action_ID, self.X_old, self.other_car_id, existing_list
    
    def return_action(self):
        new_algorithm = True
        if new_algorithm == True:
            ego_id = 1
            # GTMPC_actions_ID, X_old, others_list = self.env.Get_LevelK_action_DiGraph_ego(ego_id
            others_list = [0, 1, 2, 3]
            Interaction_list = []
            GTMPC_actions_ID = [None]*self.num_cars
            t1 = time.time()
            GTMPC_actions_ID, X_old, others_list, Interaction_list = self.Get_LevelK_action_DiGraph(ego_id, others_list, Interaction_list, GTMPC_actions_ID)
            
            print('{} Hz'.format(1/(time.time() - t1)))
            while len(others_list) > 0:
                GTMPC_actions_ID, X_old, others_list, Interaction_list = self.Get_LevelK_action_DiGraph(others_list[0], others_list, Interaction_list, GTMPC_actions_ID)
                # print('Other the rest of others_list: {}----Action:{}'.format(others_list, GTMPC_actions_ID))
            # print('---------------------------------')
        else:
            t1 = time.time()
            GTMPC_actions_ID, X_old = self.Get_LevelK_action()
            print('{} Hz'.format(1/(time.time()- t1)))
            
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
                           self.Level_ratio[car_id*(self.num_cars-1)+count, 0] = self.Level_ratio[car_id*(self.num_cars-1) + count, 0] + 0.5
                        if Action_id[inter_car] == L1_action_id[inter_car][0]:
                           self.Level_ratio[car_id*(self.num_cars-1)+count, 1] = self.Level_ratio[car_id*(self.num_cars-1) + count, 1] + 0.5

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
        # print('total', time_per_step)


        # Completion check
        if sum(self.complete_flag[self.episode, :]) == self.num_cars:
            pass
        
        self.step += 1
    
        return Action_id, self.X_old
   
    def main(self):
        print("Main begins.")
        self.step_for_newenv = 0

        get_path = self.select_path()

        Driver_level = [2, 1, 2, 2] #0-l0, 1-adp,3 conservative, 
        # Driver_level = [0, 0, 0, 0]
        self.Initial_GTMPC(get_path, Driver_level)

        while True:
 
            # action_qcar1, GTMPC_actions_ID, X_old = self.return_action()
            self.return_action()
            # time.sleep(0.5)
            self.step_for_newenv +=1
              
            # print(GTMPC_actions_ID)
            # link1.speed(X_old, GTMPC_actions_ID[1], params)
            # link2.speed(X_old, GTMPC_actions_ID[0], params)
            # link3.speed(X_old, GTMPC_actions_ID[2], params)
            # link4.speed(X_old, GTMPC_actions_ID[3], params)
            # link5.speed(X_old, GTMPC_actions_ID[4], params)
              
            # Reset environment
            if self.step_for_newenv == self.max_step:
                # stop all vehicles
                print('new episode took', self.step_for_newenv)
                self.step_for_newenv = 0
                get_path = self.select_path()
                Driver_level = [2, 1, 2, 2]
                # Driver_level = [0, 0, 0, 0]
        
                self.Initial_GTMPC(get_path, Driver_level)

if __name__ == '__main__':
    signal.signal(signal.SIGINT, quit)
    agent = DQN()
    agent.main()