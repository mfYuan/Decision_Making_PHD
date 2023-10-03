#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Author: Mingfeng Yuan;
Date: Nov/2022
"""

#import random
import random
import numpy as np
#import matplotlib.pyplot as plt
import datetime
import time
#import cv2
import math
import rospy
import signal
import sys
import numpy.matlib
import matplotlib.pyplot as plt
import os
from qcar_link1 import DQN1
from qcar_link2 import DQN2
from qcar_link3 import DQN3
from qcar_link4 import DQN4
from qcar_link5 import DQN5
from qcar_link6 import DQN6
from qcar_link7 import DQN7
from qcar_link8 import DQN8
from gazebo_env_four_qcar_links import envmodel
import plot_sim_DiGraph_new
import plot_level_ratio
import plot_graph
import paper_plt
#import os
# os.environ["CUDA_VISIBLE_DEVICES"] = '0'  # 默认显卡0

link1 = DQN1()
link2 = DQN2()
link3 = DQN3()
link4 = DQN4()
link5 = DQN5()
link6 = DQN6()
link7 = DQN7()
link8 = DQN8()
env = envmodel()


qcar1_dict = {0: -3.0, 1: -1.5, 2: 0.0, 3: 1.5, 4: 3.0}
qcarx_dict = {0: [0.3, 0.0], 1: [0.6, 0.0], 2: [0.9, 0.0], 3: [1.2, 0.0], 4: [1.5, 0.0]}
delay = 0.35

def quit(sig, frame):
    sys.exit(0)

class DQN:
    def __init__(self):
        rospy.init_node('control_node_main', anonymous=True)
        # Algorithm Information
        # self.Number='train2'
        self.debug = False

        # Get parameters
        self.progress = ''
        # ------------------------------

        self.step = 1
        self.episode = 0
        
        
        #spawn cars manually or automatically(True/False)
        self.manual_spawn = False
        self.manual_select_path = False
        self.manual_select_policy = False
        self.sava_data = False
        # Initialize agent robot
        self.agentrobot1 = 'qcar1' #Ego
        self.agentrobot2 = 'qcar2' #left # 1
        self.agentrobot3 = 'qcar3' #right# 1
        self.agentrobot4 = 'qcar4' #top# 1
        self.agentrobot5 = 'qcar5' #left # 2
        self.agentrobot6 = 'qcar6' #right# 2
        self.agentrobot7 = 'qcar7' #top# 2
        self.agentrobot8 = 'qcar8' #After Ego
        # Define the distance from start point to goal point
        self.d = 4.0
        # self.get_path = self.choose_path([2, 1, 3, 4])
        # self.Driver_level = self.set_behaviour([2, 1, 3, 4])
        # Define the step for updating the environment
        self.MAXSTEPS = 500
        # ------------------------------

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
            path_ls_car6 = {'right':'1', 'strait':'6', 'left':'11'}
            path_ls_car7 = {'right':'2', 'strait':'7', 'left':'12'}
            path_ls_car8 = {'right':'4', 'strait':'5', 'left':'10'}
            
        car1_path = random.choice(path_ls_car1.items())
        car2_path = random.choice(path_ls_car2.items())
        car3_path = random.choice(path_ls_car3.items())
        car4_path = random.choice(path_ls_car4.items()) 
        car5_path = random.choice(path_ls_car5.items())
        car6_path = random.choice(path_ls_car6.items())
        car7_path = random.choice(path_ls_car7.items()) 
        car8_path = random.choice(path_ls_car8.items())
        path_collection = [car1_path, car2_path, car3_path, car4_path, car5_path, car6_path, car7_path, car8_path]
        path = []
        for car in car_id:
            _path = path_collection[car-1]
            if car ==1:
                path_ls = {'1':'4', '2':'5', '3':'10'}
            elif car ==2:
                path_ls ={'1':'3', '2':'8', '3':'9'}
            elif car ==3:
                path_ls ={'1':'1', '2':'6', '3':'11'}
            elif car==4:
                path_ls ={'1':'2', '2':'7', '3':'12'}
            elif car==5:
                path_ls ={'1':'3', '2':'8', '3':'9'}
            elif car==6:
                path_ls ={'1':'1', '2':'6', '3':'11'}
            elif car==7:
                path_ls ={'1':'2', '2':'7', '3':'12'}
            elif car==8:
                path_ls ={'1':'4', '2':'5', '3':'10'}
            path.append(path_ls.items()[int(_path)])
        # print('car2:{}, car1:{}, car3:{}, car4:{}, car5:{}'.format(car2_path, car1_path, car3_path, car4_path, car5_path))
        return path
    
    def choose_path(self, car_id):
        
        path = []
        for car in car_id:
            if self.manual_select_path == True:
                _path = input('set path for car:{}: Right:1; Strait:2; Left:3: '.format(car))
            else:
                _path = str(random.randint(1,3))
                
            if car ==1:
                path_ls = {'1':'4', '2':'5', '3':'10'}
            elif car ==2:
                path_ls ={'1':'3', '2':'8', '3':'9'}
            elif car ==3:
                path_ls ={'1':'1', '2':'6', '3':'11'}
            elif car==4:
                path_ls ={'1':'2', '2':'7', '3':'12'}
            elif car==5:
                path_ls ={'1':'3', '2':'8', '3':'9'}
            elif car==6:
                path_ls ={'1':'1', '2':'6', '3':'11'}
            elif car==7:
                path_ls ={'1':'2', '2':'7', '3':'12'}
            elif car==8:
                path_ls ={'1':'4', '2':'5', '3':'10'}
            #print('input path: {}, path: {}'.format(_path, path_ls[_path]))
            path.append((_path, path_ls[_path]))
        # print('car2:{}, car1:{}, car3:{}, car4:{}, car5:{}'.format(path[0], path[1], path[2], path[3], path[4]))
        return path
    
    def set_behaviour(self, car_id):
        Driver_level = []
        for car in car_id:
            if self.manual_select_policy == True:
                _driver = input('set driving behaviour for car:{} (0=Aggresive / 1=Adptive / 2=Conservative): '.format(car))
            else:
                if car == 1:
                    _driver = 1
                else:
                    _driver = random.choice([0, 2])
            Driver_level.append(_driver)
        return Driver_level
    
    def set_car_num(self, car_num):
        #Todo for RL training
        if self.manual_spawn == True:
            c_id = input('how many cars?: 4 - 8?: ')
        else:
            c_id = car_num
        if int(c_id)==3:
            car_id = [2, 1] 
        elif int(c_id)==3:
           car_id = [2, 1, 3] 
        elif int(c_id)==4:
            car_id = [2, 1, 3, 4]
        elif int(c_id)==5:
            car_id = [2, 1, 3, 4, 5]
        elif int(c_id)==6:
            car_id = [2, 1, 3, 4, 5, 6]
        elif int(c_id)==7:
            car_id = [2, 1, 3, 4, 5, 6, 7]
        else:
            car_id = [2, 1, 3, 4, 5, 6, 7, 8]
        return car_id
    

    def main(self):
        print("Main begins.")
        car_id = self.set_car_num(4)
        self.step_for_newenv = 0
        spawn_ls = [str(0)] * 8
        for i in car_id:
            spawn_ls[i-1] = str(1)

        env.reset_env(spawn_ls)
        self.get_path = self.choose_path(car_id)
        self.Driver_level = self.set_behaviour(car_id)
        
        link1.env.Initial_GTMPC(self.get_path, self.Driver_level)
        new_algorithm, multi_processing = False, False
        
        link1.main_func()
        link2.main_func()
        link3.main_func()   
        link4.main_func()
        
        
        '''
        # setting1 = input('new_algorithm? (1=True / 2=False): ')
        # if setting1 ==1:
        #     new_algorithm = True
        # else:
        #     new_algorithm = False
        # print('new_algorithm{}'.format(new_algorithm))
            
        # if new_algorithm == False:
        #     multi_processing = False
        # else:
        #     setting2 = input('multi_processing? (1=True / 2=False): ')
        #     if setting2 ==1:
        #         multi_processing = True
        #         print('multi_processing{}'.format(multi_processing))
        #     else:
        #         multi_processing = False
        #         print('multi_processing{}'.format(multi_processing))'''
        if self.sava_data == True:
            self.Record_data = {}
            _data = []
            _vel = []
            _matrix = []
            Level_ratio_his=np.zeros((1, link1.env.max_step, np.shape(link1.env.Level_ratio)[0], np.shape(link1.env.Level_ratio)[1]))
            Pos_his = np.zeros((1, link1.env.max_step, 10, len(car_id)))
            
            creat_folder = input('new folder name: ')
            path = '/home/sdcnlab025/ROS_test/four_qcar/src/tf_pkg/safe_RL/MPC_RL/Images/Recorded1/'+str(creat_folder)
            
            # Check whether the specified path exists or not
            isExist = os.path.exists(path)
            path_ls = []
            if not isExist:
                name_ls = ['/motion', '/graph', '/others', '/level', '/txtfile']
                for i in name_ls:
                    # Create a new directory because it does not exist
                    sub_folder = path + i
                    path_ls.append(sub_folder)
                    os.makedirs(sub_folder)
            
        speed_data=[]
        while True:
            # env.render_env(X_old)
            strt = time.time()
            progress, Epsilon = link1.get_progress(self.step, self.Epsilon)
            self.progress = progress
            if link1.isTraining == True:
                self.step += 1
                self.Epsilon = Epsilon
            action_qcar1 = link1.return_action()
            cmd_qcar1 = qcar1_dict[action_qcar1]
            
            
            GTMPC_actions_ID, X_old, l_ratio, complet_ls, Matrix_buffer = link1.select_action_new(new_algorithm, multi_processing, len(car_id))
            
            #print('GTMPC_actions_ID: {}'.format(GTMPC_actions_ID))
            params = link1.env.params
            
            link1.speed(X_old, GTMPC_actions_ID[1], params)
            link2.speed(X_old, GTMPC_actions_ID[0], params)
            link3.speed(X_old, GTMPC_actions_ID[2], params)
            
            if len(car_id) >=4:
                link4.speed(X_old, GTMPC_actions_ID[3], params)
            if len(car_id) >=5:
                link5.speed(X_old, GTMPC_actions_ID[4], params)
            if len(car_id) >=6:
                link6.speed(X_old, GTMPC_actions_ID[5], params)
            if len(car_id) >=7:
                link7.speed(X_old, GTMPC_actions_ID[6], params)
            if len(car_id) >=8:
                link8.speed(X_old, GTMPC_actions_ID[7], params)
            
            ##speed_data.append([time_stamp]+link1.env.data_vel)
                
            X_old = link1.env.veh_pos_realtime(X_old)
            ##_data.append([time_stamp, link1.running_time])
            if new_algorithm ==True and self.sava_data == True:
                if multi_processing == True:
                    _matrix.append(Matrix_buffer[1]['matrix'])
                else:
                    _matrix.append(Matrix_buffer[1])
            # _vel.append(link1.env.data_vel)
            _tem = []
            for it in range(len(X_old[0, :])):
                if X_old[8, it]+100 >=X_old[6, it]:
                    current_idx = X_old[6, it]-100
                else:
                    current_idx = X_old[8, it]
                ##_tem.append((current_idx-X_old[5, it])/(X_old[6, it]-100-X_old[5, it]))
                # _tem.append(X_old[8, it]/(X_old[6, it]-X_old[5, it]))
            ##_vel.append(_tem)
            ##Pos_his[link1.env.episode, self.step_for_newenv, :, :] = X_old
            ##Level_ratio_his[link1.env.episode, self.step_for_newenv, :, :] = l_ratio
            terminal_1 = link1.update_information()
            self.step_for_newenv += 1
            end = time.time()
            
            print('time: {}'.format(end-strt))
            # Reset environment
            if self.step_for_newenv == link1.env.max_step or len(complet_ls) == len(car_id):#or terminal_1 == True
                env.stop_all_cars()
                
                time.sleep(0.1)
                if self.sava_data == True:
                    self.Record_data[str(self.episode)] = _data
                    # print(str(self.episode), self.Record_data[str(self.episode)])
                    # print('--------------------------------')
                    _episode = 0
                    ls = []
                    #print(self.Record_data.keys())
                    while _episode != -1:
                        _episode = input('choose episodes: ')
                        ls.append(_episode)
                    ls.remove(-1)
                    
                    plt.ion()
                    
                    ##############################################
                    data = np.array(speed_data)
                    save_path = path_ls[4]+"/speedfile.txt"
                    np.savetxt(save_path, data)
                    ##############################################
                    for item in self.Record_data.keys():
                        _data = self.Record_data[item]
                        data1 = np.array(_data)
                        save_path = path_ls[4]+"/run"+item+".txt"
                        np.savetxt(save_path, data1)
                    ##############################################
                    ##################
                    data1 = np.array(_vel)
                    save_path = path_ls[4]+"/progress.txt"
                    np.savetxt(save_path, data1)
                    ##############################
                    data1 = np.array(Level_ratio_his[self.episode, :, :, :])
                    data1_reshape = data1.reshape(data1.shape[0], -1)
                    save_path = path_ls[4]+"/ratio.txt"
                    np.savetxt(save_path, data1_reshape)
                    ##############################
                    data3 = np.array(Pos_his[self.episode, :, :, :])
                    data3_reshape = data3.reshape(data3.shape[0], -1)
                    save_path = path_ls[4]+"/pos.txt"
                    np.savetxt(save_path, data3_reshape)
                    ##############################
                    ##################
                    data4 = np.array(_matrix)
                    data4_reshape = data4.reshape(data4.shape[0], -1)
                    save_path = path_ls[4]+"/matrix.txt"
                    np.savetxt(save_path, data4_reshape)
                    ##############################
                    if len(ls)>0:
                        # plt.figure('Running Time', figsize=(8, 6))
                        paper_plt.plot_vel(speed_data)
                        paper_plt.plot_runtime(self.Record_data, ls, path_ls[2], params)

                        pre_time = 0
                        fig_sim = plt.figure('Intersection Scenario', figsize=(6, 6))
                        fig_graph = plt.figure('Dynamic graph', figsize=(6, 6))
                        
                        counter = 0
                        paper_plt.plot_progress(self.Record_data, _vel, path_ls, ls, params)
                        # plt.figure('vel', figsize=(8, 6))

                        paper_plt.plot_level_ratio(self.Record_data, Level_ratio_his, path_ls, ls, params)
                                    
                        
                        # self.fig_0 = plt.figure('Probability to car1', figsize=(6, 3))
                        # self.fig_2 = plt.figure('Probability to car2', figsize=(6, 3))
                        # self.fig_3 = plt.figure('Probability to car3', figsize=(6, 3))
                        for k in range(len(_data)):
                            step =  _data[k][0]- pre_time
                            pre_time = _data[k][0]
                            # print('---level_ratio---',level_ratio)
                            lr = Level_ratio_his[link1.env.episode, counter, :, :]
                            pos = Pos_his[link1.env.episode, counter, :, :]
                            # 
                            # Dynamic graph
                            if new_algorithm ==True:
                                plot_graph.plot_graph(pos, _matrix[counter], fig_graph, counter, path_ls)
                            plot_sim_DiGraph_new.plot_sim(pos, link1.env.params, step, lr, fig_sim, counter, path_ls)
                            #Estimation
                            # for ii in range(1, link1.env.num_cars):
                            #     plot_level_ratio.plot_level_ratio(Level_ratio_his, 1, ii, params, k, link1.env.episode, link1.env.max_step, ii, step)
                            counter +=1
                    # plot_level_ratio.plot_level_ratio(self.Level_ratio_history, ego_car_id, opp_car_id, self.params, self.step, self.episode, self.max_step, self.fig_0)
                    
                self.episode +=1 
                self.step_for_newenv = 0

                self.spawn_qcar2 = str(0)
                self.spawn_qcar3 = str(0)
                self.spawn_qcar4 = str(0)
                self.spawn_qcar5 = str(1)

      
                # Choose path for qcar1 - 4 is right turn, 5 is straight, 10 is left turn
                # qcar1_path = str(np.random.choice([4, 5, 10]))
                    
                spawn_ls = [str(0)] * 8   
                car_id = self.set_car_num(4) 
                for i in car_id:
                    spawn_ls[i-1] = str(1)
                
                env.reset_env(spawn_ls)
                
                self.get_path = self.choose_path(car_id)
                self.Driver_level = self.set_behaviour(car_id)
                link1.env.Initial_GTMPC(self.get_path, self.Driver_level)
                
                '''
                check_state = input('Next round of test? (1=Ready / 2=Wait): ')
                if check_state:      
                    ans = input('Reset Driving behavioursh? (1=True / 2=False): ')
                    if ans ==1:
                        self.Driver_level = self.set_behaviour(car_id)
                    # self.Driver_level = [2, 1, 2, 2, 2]
                    # Driver_level = [0, 0, 0, 0]
                    
                    link1.env.Initial_GTMPC(self.get_path, self.Driver_level)
                    setting1 = input('new_algorithm? (1=True / 2=False): ')
                    if setting1 ==1:
                        new_algorithm = True
                        print('new_algorithm{}'.format(new_algorithm))
                    else:
                        new_algorithm = False
                        print('new_algorithm{}'.format(new_algorithm))
                    if new_algorithm == False:
                        multi_processing = False
                    else:
                        setting2 = input('multi_processing? (1=True / 2=False): ')
                        if setting2 ==1:
                            multi_processing = True
                            print('multi_processing{}'.format(multi_processing))
                        else:
                            multi_processing = False
                            print('multi_processing{}'.format(multi_processing))
                    # link1.env.Initial_GTMPC(get_path)
                    _data = []
                    _vel = []
                    _matrix = []
                    Level_ratio_his=np.zeros((1, link1.env.max_step, np.shape(link1.env.Level_ratio)[0], np.shape(link1.env.Level_ratio)[1]))
                    Pos_his = np.zeros((1, link1.env.max_step, 10, len(car_id)))
                    
                    creat_folder = input('new folder name: ')
                    path = '/home/sdcnlab025/ROS_test/four_qcar/src/tf_pkg/safe_RL/MPC_RL/Images/Recorded1/'+str(creat_folder)
                    
                    # Check whether the specified path exists or not
                    isExist = os.path.exists(path)
                    path_ls = []
                    if not isExist:
                        name_ls = ['/motion', '/graph', '/others', '/level', '/txtfile']
                        for i in name_ls:
                            # Create a new directory because it does not exist
                            sub_folder = path + i
                            path_ls.append(sub_folder)
                            os.makedirs(sub_folder)
                    speed_data=[]'''
                    

if __name__ == '__main__':
    signal.signal(signal.SIGINT, quit)
    agent = DQN()
    agent.main()