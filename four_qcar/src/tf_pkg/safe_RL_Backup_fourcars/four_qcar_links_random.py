#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Author: Mingfeng Yuan; Michael;
Date: 12/2020
"""

# Import modules

import random
import numpy as np
import matplotlib.pyplot as plt
import datetime
import time
import cv2
import math
import rospy
import signal
import sys

from four_qcar_link1_test import DQN1
#from Dueling_PER import Dueling

from four_qcar_link2 import DQN2
from four_qcar_link3 import DQN3
from four_qcar_link4 import DQN4
from gazebo_env_four_qcar_links import envmodel
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'  # 默认显卡0

link1 = DQN1()
#link1_Dueling = Dueling()
link2 = DQN2()
link3 = DQN3()
link4 = DQN4()
env = envmodel()

# qcar1_dict = {0: -1.0, 1: -0.5, 2: 0.0, 3: 0.5, 4: 1.0}
# qcar1_dict = {0: [0.0, 0.0], 1: [1.5, 0.0]}
# qcarx_dict = {0: [0.0, 0.0], 1: [1.5, 0.0]}
qcar1_dict = {0: -3.0, 1: -1.5, 2: 0.0, 3: 1.5, 4: 3.0}
qcarx_dict = {0: [0.3, 0.0], 1: [0.6, 0.0], 2: [
    0.9, 0.0], 3: [1.2, 0.0], 4: [1.5, 0.0]}
# qcarx_dict = {0: [0.3, 0.0], 1: [0.75, 0.0], 2: [1.5, 0.0]}
#qcarx_dict = {0: [0.03, 0.0], 1: [0.75, 0.0], 2: [1.5, 0.0]}
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

        # Get parameters
        self.progress = ''

        # Initial parameters
        # ------------------------------
        self.Num_start_training = 6000  # 5000
        self.Num_training = 200000  # 60000
        # ------------------------------
        self.Num_test = 500000  # Stable Training parameter

        self.learning_rate = 0.001  # 0.001
        self.Gamma = 0.95  # 0.99

        # ------------------------------
        self.train_num = 1
        self.Start_epsilon = 1
        self.Epsilon = self.Start_epsilon
        self.Final_epsilon = 0.05
        # ------------------------------

        self.step = 1
        self.score = 0
        self.episode = 0

        # date - hour - minute - second of training time
        self.date_time = str(datetime.date.today())

        self.save_location = 'saved_networks/' + 'two_qcar_links_' + \
            self.date_time + '_' + self.Number + '/qcar1'

        # Initialize agent robot
        self.agentrobot1 = 'qcar1'
        self.agentrobot2 = 'qcar2'
        self.agentrobot3 = 'qcar3'
        self.agentrobot4 = 'qcar4'

        self.action_list = []
        self.episode_action = []
        self.steps_list = []
        self.env_list = []
        self.env_settings = []
        self.episode_position = []
        self.violation_list = []
	self.position_list = []
	self.status_list = []
	self.dangerzone_1 = False

        # Define the distance from start point to goal point
        self.d = 4.0

        # Define the step for updating the environment
        self.MAXSTEPS = 150
        # ------------------------------
        if self.Task == 'training':
            self.MAXEPISODES = 2000  # 100000
        else:
            self.MAXEPISODES = self.Num_test

        # ------------------------------

        link1.update_parameters(self.Num_start_training, self.Num_training, self.Num_test,
                                self.learning_rate, self.Gamma, self.Epsilon, self.Final_epsilon, self.MAXEPISODES, self.Start_epsilon, self.train_num)
        link2.update_parameters(self.Num_start_training, self.Num_training, self.Num_test,
                                self.learning_rate, self.Gamma, self.Epsilon, self.Final_epsilon, self.MAXEPISODES, self.Start_epsilon, self.train_num)
        link3.update_parameters(self.Num_start_training, self.Num_training, self.Num_test,
                                self.learning_rate, self.Gamma, self.Epsilon, self.Final_epsilon, self.MAXEPISODES, self.Start_epsilon, self.train_num)
        link4.update_parameters(self.Num_start_training, self.Num_training, self.Num_test,
                                self.learning_rate, self.Gamma, self.Epsilon, self.Final_epsilon, self.MAXEPISODES, self.Start_epsilon, self.train_num)

        link1.initialize_network()
        link2.initialize_network()
        link3.initialize_network()
        link4.initialize_network()

    def main(self):
        print("Main begins.")
        self.step_for_newenv = 0

        # 0 is no, 1 is yes
        self.spawn_qcar2 = str(1)
        self.spawn_qcar3 = str(1)
        self.spawn_qcar4 = str(1)

        self.level_qcar2 = str(0)
        self.level_qcar3 = str(0)
        self.level_qcar4 = str(0)

        env.reset_env(self.spawn_qcar2, self.spawn_qcar3, self.spawn_qcar4)
        qcar1_path = str(5)
        link1.update_path(str(5))
        link2.update_path(str(8), str(5))
        link3.update_path(str(6), str(5))
        link4.update_path(str(7), str(5))

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

        # if link1.isTraining == False:
        #     inference = True
        # else:
        #     inference = False
        # Training & Testing
        start_episode = time.time()
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
                self.MAXSTEPS = 150
            elif self.progress == 'Not Training':
		if delay == 0.35:
                    self.MAXSTEPS = 150
		else:
		    self.MAXSTEPS = 1000
            else:
                self.MAXSTEPS = 150

            action_qcar1 = link1.return_action()
            cmd_qcar1 = qcar1_dict[action_qcar1]

            self.episode_position.extend(env.return_position())
            self.episode_action.append(cmd_qcar1)

            qcar2, qcar3, qcar4_1, qcar4_2, qcar4_check1 = env.get_collision_status(
                self.spawn_qcar2, self.spawn_qcar3, self.spawn_qcar4)

            link1.accelerate(cmd_qcar1)
            if go_first == 23:
                link2.move(cmd_qcar2)
                link3.move(cmd_qcar3)
                if qcar3 and qcar2:
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
            #     time.sleep(delay)  # 0.45)

            # if link1.isTraining == False:
            #     time.sleep(delay)  # 0.45)

            terminal_1, collided_1, dangerzone_1 = link1.update_information()
            # terminal_2 = link2.update_information()
            # terminal_3 = link3.update_information()
            # terminal_4 = link4.update_information()
	    if dangerzone_1 == True:
	   	self.dangerzone_1 = True

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

                self.steps_list.append(self.step_for_newenv)
                np.savetxt(self.save_location + '/qcar1_steps_list.txt',
                           self.steps_list, delimiter=',')

                self.action_list.append(self.episode_action)
                self.episode_action = []
                print(self.action_list)
                np.savetxt(self.save_location + '/qcar1_action_list.csv',
                           self.action_list, fmt ="%s", delimiter=",")

                self.env_settings = [int(qcar1_path), int(go_first), int(self.spawn_qcar2), int(self.spawn_qcar3),
                                     int(self.spawn_qcar4), cmd_qcar2[0], cmd_qcar3[0], cmd_qcar4[0]]
                print(self.env_settings)
                self.env_list.append(self.env_settings)

                np.savetxt(self.save_location + '/qcar1_env_list.txt',
                           self.env_list, delimiter=',')

                self.position_list.append(self.episode_position)
                self.episode_position = []
                
                np.savetxt(self.save_location + '/qcar1_pos_list.csv',
                           self.position_list, fmt ="%s", delimiter=',')

                if collided_1 != 0 and terminal_1 == True:
                    status = 0
                elif self.step_for_newenv == self.MAXSTEPS:
                    status = 1
                else:
                    status = 2
		if self.dangerzone_1 == True :
		   dangerzone_1 = 1
		else:
		   dangerzone_1 = 0 
		

                status_set = [status, collided_1, dangerzone_1]
                self.status_list.append(status_set)
                np.savetxt(self.save_location + '/qcar1_status_list.txt',
                           self.status_list, delimiter=',')

                print(self.status_list)

                # stop all vehicles
                env.stop_all_cars()
                time.sleep(0.1)
                if self.step_for_newenv == self.MAXSTEPS:
                    print('Too Slow.....')

                # if self.episode % 50 == 0 and self.episode > 3:
                    # link1.save_model()
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

                Num_cars = np.random.choice(
                    [3, 2, 1, 0], p=[0.4, 0.3, 0.2, 0.1])

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
                link1.update_path(qcar1_path)
                link2.update_path(str(8), qcar1_path)
                link3.update_path(str(6), qcar1_path)
                link4.update_path(str(7), qcar1_path)

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
		
		self.dangerzone_1 = False



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

