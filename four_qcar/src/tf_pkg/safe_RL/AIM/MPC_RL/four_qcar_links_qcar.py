#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Author: Mingfeng Yuan;
Date: Nov/2022
"""

# Import modules

#import random
import random
import numpy as np
import matplotlib.pyplot as plt
import datetime
import time
#import cv2
import math
import rospy
import signal
import sys


from qcar_link1 import DQN1
# from qcar_link2 import DQN2
# from qcar_link3 import DQN3
# from qcar_link4 import DQN4
# from qcar_link5 import DQN5
# from gazebo_env_four_qcar_links import envmodel
#import os
# os.environ["CUDA_VISIBLE_DEVICES"] = '0'  # 默认显卡0

link1 = DQN1()
# link2 = DQN2()
# link3 = DQN3()
# link4 = DQN4()
# link5 = DQN5()
# env = envmodel()

# qcar1_dict = {0: -1.0, 1: -0.5, 2: 0.0, 3: 0.5, 4: 1.0}
#qcar1_dict = {0: [0.0, 0.0], 1: [1.5, 0.0]}
qcar1_dict = {0: -3.0, 1: -1.5, 2: 0.0, 3: 1.5, 4: 3.0}
# qcarx_dict = {0: [0.3, 0.0], 1: [0.75, 0.0], 2: [1.5, 0.0]}
#qcarx_dict = {0: [0.0, 0.0], 1: [0.75, 0.0], 2: [1.5, 0.0]}
qcarx_dict = {0: [0.3, 0.0], 1: [0.6, 0.0], 2: [0.9, 0.0], 3: [1.2, 0.0], 4: [1.5, 0.0]}
#qcarx_dict= {1:[0.75, 0.0], 2: [1.5, 0.0]}
delay = 0.35

def quit(sig, frame):
    sys.exit(0)

class DQN:
    def __init__(self):
        rospy.init_node('control_node_main', anonymous=True)
        # Algorithm Information
        self.algorithm = 'D3QN_PER'
        self.Task = 'training'
        self.Number = 'train1'
        # self.Number='train2'
        self.debug = False

        # Get parameters
        self.progress = ''

        # Initial parameters
        # ------------------------------
        self.Num_start_training = 6000  # 5000
        self.Num_training = 150000  # 60000
        # ------------------------------
        self.Num_test = 100000  # Stable Training parameter

        self.learning_rate = 0.001  # 0.001
        self.Gamma = 0.95  # 0.99

        # ------------------------------
        self.train_num = 2
        self.Start_epsilon = 0.5
        self.Epsilon = self.Start_epsilon
        self.Final_epsilon = 0.01
        # ------------------------------

        self.step = 1
        self.score = 0
        self.episode = 0

        # date - hour - minute - second of training time
        self.date_time = str(datetime.date.today())

        # Initialize agent robot
        self.agentrobot1 = 'qcar1'
        self.agentrobot2 = 'qcar2'
        self.agentrobot3 = 'qcar3'
        self.agentrobot4 = 'qcar4'
        self.agentrobot5 = 'qcar5'

        # Define the distance from start point to goal point
        self.d = 4.0

        # Define the step for updating the environment
        self.MAXSTEPS = 500
        # ------------------------------
        if self.Task == 'training':
            self.MAXEPISODES = 1000  # 100000
        else:
            self.MAXEPISODES = self.Num_test

        # ------------------------------

        link1.update_parameters(self.Num_start_training, self.Num_training, self.Num_test,
                                self.learning_rate, self.Gamma, self.Epsilon, self.Final_epsilon, self.MAXEPISODES, self.Start_epsilon, self.train_num)
        link2.update_parameters(self.Num_start_training, self.Num_training, self.Num_test,
                                self.learning_rate, self.Gamma, self.Epsilon, self.Final_epsilon, self.MAXEPISODES)
        link3.update_parameters(self.Num_start_training, self.Num_training, self.Num_test,
                                self.learning_rate, self.Gamma, self.Epsilon, self.Final_epsilon, self.MAXEPISODES)
        link4.update_parameters(self.Num_start_training, self.Num_training, self.Num_test,
                                self.learning_rate, self.Gamma, self.Epsilon, self.Final_epsilon, self.MAXEPISODES)

        link1.initialize_network()
        link2.initialize_network()
        link3.initialize_network()
        link4.initialize_network()
        
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
    
    # def PID1(self, cmd_v, max_speed1):
    #     current_time = 1.0*self.elapsed_time()#np.mean(self.time_factors)*self.elapsed_time()
    #     #print("real time factor", np.mean(self.time_factors))
    #     st = current_time - self.previous_t
    #     if st < 0.04:
    #         st = 0.04
    #     if cmd_v != self.previous_action and cmd_v == 0.0:
    #         #print(cmd_v, self.previous_action, cmd_v == self.previous_action)
    #         self.maintain =self.robotstate1[2]
    #     if cmd_v ==0.0:
    #         try:
    #             desired_v = self.maintain
    #             state = 'fix'
    #         except:
    #             print('error #1 here!')
    #     else:
    #         desired_v = self.desired_velocity(cmd_v, st, 1, max_speed1)
    #         state = 'var'
    #     #print('desired_v:{}{}'.format(desired_v, state))
    #     #print('self.maintain',self.maintain)
    #     self.previous_action = cmd_v
    #     current_v = self.robotstate1[2]
    #     #print("time step is {}".format(st))
    #     kp = 1
    #     ki = 0.003#.005#0.1
    #     kd = 0.015#0.06
        
    #     delta_v = desired_v - current_v #error term

    #     integral = self.integral_val1 + (delta_v * st) #intergral

    #     derivative = (delta_v - self.derivative_val1) / st #derivative


    #     rst = kp * delta_v + ki * integral + kd * derivative #PID 2.6*desired_v


    #     pid_output = rst

    #     self.previous_t = current_time

    #     return desired_v, pid_output

    def main(self):
        print("Main begins.")
        self.step_for_newenv = 0

        # spawn_cars = [np.random.randint(0,2) for i in range(4)]
        
        # 0 is no, 1 is yes
        self.spawn_qcar2 = str(1)
        self.spawn_qcar3 = str(1)
        self.spawn_qcar4 = str(1)
        self.spawn_qcar5 = str(1)

        # self.level_qcar2 = str(0)
        # self.level_qcar3 = str(0)
        # self.level_qcar4 = str(0)

        env.reset_env(self.spawn_qcar2, self.spawn_qcar3, self.spawn_qcar4, self.spawn_qcar5)
        
        get_path = self.select_path()

        link1.update_path(get_path[0][1])
        link2.update_path(get_path[1][1],get_path[0][1])
        link3.update_path(get_path[2][1],get_path[0][1])
        link4.update_path(get_path[3][1],get_path[0][1])
        link5.update_path(get_path[4][1],get_path[0][1])
        

        action_qcar4 = np.random.randint(5)
        action_qcar2 = np.random.randint(5)
        action_qcar3 = np.random.randint(5)

        cmd_qcar2 = qcarx_dict[action_qcar2]
        cmd_qcar3 = qcarx_dict[action_qcar3]
        cmd_qcar4 = qcarx_dict[action_qcar4]

        go_first = np.random.choice([23, 4])

        link1.set_velocity(0, 0, 0, self.spawn_qcar2,
                           self.spawn_qcar3, self.spawn_qcar4)
        link2.set_velocity(0, 0, 0, '1', self.spawn_qcar3, self.spawn_qcar4)
        link3.set_velocity(0, 0, 0, '1', self.spawn_qcar2, self.spawn_qcar4)
        link4.set_velocity(0, 0, 0, '1', self.spawn_qcar2, self.spawn_qcar3)

        link1.main_func()
        link2.main_func()
        link3.main_func()
        link4.main_func()

        test_count = 0
        Driver_level = [2, 1, 2, 2, 2] #0-l0, 1-adp,3 conservative, 
        # Driver_level = [2, 1, 2, 2]
        # Driver_level = [0, 0, 0, 0]
        
        # Get_graph = link1.env.
        link1.env.Initial_GTMPC(get_path, Driver_level)

        # if link1.isTraining == False:
        #     inference = True
        # else:
        #     inference = False

        # Training & Testing
        while True:
            start = datetime.datetime.now()
            progress, Epsilon = link1.get_progress(self.step, self.Epsilon)
            link2.get_progress(self.step, self.Epsilon)
            link3.get_progress(self.step, self.Epsilon)
            link4.get_progress(self.step, self.Epsilon)

            self.progress = progress

            if link1.isTraining == True:
                self.step += 1
                self.Epsilon = Epsilon

            if self.progress == 'Observing':
                self.MAXSTEPS = link1.env.max_step
            elif self.progress == 'Not Training':
                self.MAXSTEPS = link1.env.max_step
            else:
                self.MAXSTEPS = link1.env.max_step

            action_qcar1, GTMPC_actions_ID, X_old = link1.return_action()
            # env.render_env(X_old)
            params = link1.env.params
            # print(GTMPC_actions_ID)
            link1.speed(X_old, GTMPC_actions_ID[1], params)
            link2.speed(X_old, GTMPC_actions_ID[0], params)
            link3.speed(X_old, GTMPC_actions_ID[2], params)
            link4.speed(X_old, GTMPC_actions_ID[3], params)
            link5.speed(X_old, GTMPC_actions_ID[4], params)
              
            cmd_qcar1 = qcar1_dict[action_qcar1]

            qcar2, qcar3, qcar4_1, qcar4_2, qcar4_check1 = env.get_collision_status(
                self.spawn_qcar2, self.spawn_qcar3, self.spawn_qcar4)

            '''link1.accelerate(cmd_qcar1)
            if go_first == 23:
                link2.move(cmd_qcar2)
                link3.move(cmd_qcar3)
                if qcar3 and not qcar4_check1:
                    link4.move(cmd_qcar4)
                elif qcar3 and qcar2:
                    link4.move(cmd_qcar4)

            else:
                link4.move(cmd_qcar4)
                if qcar4_1:
                    link3.move(cmd_qcar3)
                if qcar4_2:
                    link2.move(cmd_qcar2)

            link1.set_velocity(0, 0, 0, self.spawn_qcar2,
                               self.spawn_qcar3, self.spawn_qcar4)
            link2.set_velocity(
                0, 0, 0, '1', self.spawn_qcar3, self.spawn_qcar4)
            link3.set_velocity(
                0, 0, 0, '1', self.spawn_qcar2, self.spawn_qcar4)
            link4.set_velocity(
                0, 0, 0, '1', self.spawn_qcar2, self.spawn_qcar3)

            # if self.progress == "Observing" and link1.isTraining == True:
            #     time.sleep(0.35)  # 0.54
            # if link1.isTraining == False:
            #     time.sleep(0.35)  # 0.54

            # Update information (returns self.done_list)'''
            terminal_1 = link1.update_information()
            # terminal_2 = link2.update_information()
            # terminal_3 = link3.update_information()
            # terminal_4 = link4.update_information()

            self.step_for_newenv += 1

            stop = datetime.datetime.now()-start
            if stop < datetime.timedelta(seconds=delay):
                diff = (datetime.timedelta(seconds=delay)-stop).total_seconds()
                time.sleep(diff)

            if self.progress == 'Training' or self.progress == 'Stable Training':
                if self.step % 50 == 0 or self.step < 6000:
                    print('Training took', datetime.datetime.now()-start)
            else:
                if self.progress == 'Observing' and self.step % 50 == 0:
                    print('Observing Took', datetime.datetime.now()-start)
                elif self.progress == 'Not Training' and test_count % 50 == 0:
                    print('Not Training took', datetime.datetime.now()-start)
                    test_count += 1

            # Reset environment
            if self.step_for_newenv == self.MAXSTEPS or terminal_1 == True:
                # stop all vehicles
                env.stop_all_cars()
                time.sleep(0.1)
                if self.step_for_newenv == self.MAXSTEPS:
                    print('Too Slow.....')

                # if self.episode % 50 == 0 and self.episode > 3:
                #     link1.save_model()
                    # link2.save_model()
                    # link3.save_model()

                link1.print_information()
                link2.print_information()
                link3.print_information()
                link4.print_information()

                print('new episode took', self.step_for_newenv)

                self.step_for_newenv = 0
                test_count = 0

                if self.progress != 'Observing':
                    self.episode += 1

                # Choose whether to spawn vehicle - 0 is no, 1 is yes
                self.spawn_qcar2 = str(0)
                self.spawn_qcar3 = str(0)
                self.spawn_qcar4 = str(0)

                Num_cars = 3#np.random.choice([3, 2, 1, 0], p=[0.4, 0.3, 0.2, 0.1])

                if Num_cars == 3:
                    self.spawn_qcar2 = str(1)
                    self.spawn_qcar3 = str(1)
                    self.spawn_qcar4 = str(1)
                elif Num_cars == 2:
                    two_qcar = np.random.choice([23, 24, 34])
                    if two_qcar == 23:
                        self.spawn_qcar2 = str(1)
                        self.spawn_qcar3 = str(1)
                    elif two_qcar == 24:
                        self.spawn_qcar2 = str(1)
                        self.spawn_qcar4 = str(1)
                    elif two_qcar == 34:
                        self.spawn_qcar3 = str(1)
                        self.spawn_qcar4 = str(1)
                elif Num_cars == 1:
                    spawn = np.random.choice([2, 3, 4])
                    if spawn == 2:
                        self.spawn_qcar2 = str(1)
                    elif spawn == 3:
                        self.spawn_qcar3 = str(1)
                    elif spawn == 4:
                        self.spawn_qcar4 = str(1)

                # Choose path for qcar1 - 4 is right turn, 5 is straight, 10 is left turn
                qcar1_path = str(np.random.choice([4, 5, 10]))
                get_path = self.select_path()
                link1.update_path(get_path[0][1])
                link2.update_path(get_path[1][1],get_path[0][1])
                link3.update_path(get_path[2][1],get_path[0][1])
                link4.update_path(get_path[3][1],get_path[0][1])
                link5.update_path(get_path[4][1],get_path[0][1])

                action_qcar4 = np.random.randint(5)
                action_qcar2 = np.random.randint(5)
                action_qcar3 = np.random.randint(5)

                cmd_qcar2 = qcarx_dict[action_qcar2]
                cmd_qcar3 = qcarx_dict[action_qcar3]
                cmd_qcar4 = qcarx_dict[action_qcar4]

                go_first = np.random.choice([23, 4])

                link1.set_velocity(0, 0, 0, self.spawn_qcar2,
                                   self.spawn_qcar3, self.spawn_qcar4)
                link2.set_velocity(
                    0, 0, 0, '1', self.spawn_qcar3, self.spawn_qcar4)
                link3.set_velocity(
                    0, 0, 0, '1', self.spawn_qcar2, self.spawn_qcar4)
                link4.set_velocity(
                    0, 0, 0, '1', self.spawn_qcar2, self.spawn_qcar3)

                env.reset_env(self.spawn_qcar2,
                              self.spawn_qcar3, self.spawn_qcar4)

                link1.new_environment()
                link2.new_environment()
                link3.new_environment()
                link4.new_environment()
                Driver_level = [2, 1, 2, 2, 2]
                # Driver_level = [0, 0, 0, 0]
        
                link1.env.Initial_GTMPC(get_path, Driver_level)
                # link1.env.Initial_GTMPC(get_path)

                if self.episode == self.MAXEPISODES or self.progress == "Finished":
                    print("Finished training!")
                    link1.save_model()
                    # plt.savefig(link1.save_location + '/test_average.png')
                    # plt.show()
                    link1.save_fig()
                    # link2.save_fig()
                    # link3.save_fig()
                    # link4.save_fig()
                    break


if __name__ == '__main__':
    signal.signal(signal.SIGINT, quit)
    agent = DQN()
    agent.main()

