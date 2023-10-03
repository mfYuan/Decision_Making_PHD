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

from two_qcar_link1_test import DQN1
from two_qcar_link2 import DQN2
from two_qcar_link3 import DQN3
from two_qcar_link4 import DQN4
from gazebo_env_four_qcar_links import envmodel
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = '0'  # 默认显卡0

link1 = DQN1()
link2 = DQN2()
link3 = DQN3()
link4 = DQN4()
env = envmodel()

# qcar1_dict = {0: -1.0, 1: -0.5, 2: 0.0, 3: 0.5, 4: 1.0}
# qcar1_dict = {0: [0.0, 0.0], 1: [1.5, 0.0]}
# qcarx_dict = {0: [0.0, 0.0], 1: [1.5, 0.0]}
qcar1_dict = {0: -3.0, 1: -1.5, 2: 0.0, 3: 1.5, 4: 3.0}
qcarx_dict = {0: -3.0, 1: -1.5, 2: 0.0, 3: 1.5, 4: 3.0}
delay = 0.35
# qcarx_dict = {0: [0.3, 0.0], 1: [0.75, 0.0], 2: [1.5, 0.0]}
#qcarx_dict = {0: [0.03, 0.0], 1: [0.75, 0.0], 2: [1.5, 0.0]}


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
        self.Num_test = 300000  # Stable Training parameter

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

        # Define the distance from start point to goal point
        self.d = 4.0

        # Define the step for updating the environment
        self.MAXSTEPS = 150
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

    def main(self):
        print("Main begins.")
        self.step_for_newenv = 0

        # 0 is no, 1 is yes
        self.spawn_qcar2 = str(1)
        self.spawn_qcar3 = str(1)
        self.spawn_qcar4 = str(1)

        # Available levels: 1 and 2
        self.level_qcar2 = str(1)
        self.level_qcar3 = str(1)
        self.level_qcar4 = str(1)

        link2.update_model(self.level_qcar2)
        link3.update_model(self.level_qcar3)
        link4.update_model(self.level_qcar4)

        env.reset_env(self.spawn_qcar2, self.spawn_qcar3, self.spawn_qcar4)
        link1.update_path(str(5))
        link2.update_path(str(8), str(5))
        link3.update_path(str(6), str(5))
        link4.update_path(str(7), str(5))

        link1.set_velocity(0, 0, 0, self.spawn_qcar2,
                           self.spawn_qcar3, self.spawn_qcar4)
        link2.set_velocity(0, 0, 0, '1', self.spawn_qcar3, self.spawn_qcar4)
        link3.set_velocity(0, 0, 0, '1', self.spawn_qcar2, self.spawn_qcar4)
        link4.set_velocity(0, 0, 0, '1', self.spawn_qcar2, self.spawn_qcar3)

        link1.main_func()
        link2.main_func()
        link3.main_func()
        link4.main_func()

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
                self.MAXSTEPS = 150
            elif self.progress == 'Not Training':
                self.MAXSTEPS = 150
            else:
                self.MAXSTEPS = 150

            action_qcar1 = link1.return_action()
            action_qcar2 = link2.return_action()
            action_qcar3 = link3.return_action()
            action_qcar4 = link4.return_action()

            cmd_qcar1 = qcar1_dict[action_qcar1]
            cmd_qcar2 = qcarx_dict[action_qcar2]
            cmd_qcar3 = qcarx_dict[action_qcar3]
            cmd_qcar4 = qcarx_dict[action_qcar4]

            link1.accelerate(cmd_qcar1)
            link2.accelerate(cmd_qcar2)
            link3.accelerate(cmd_qcar3)
            link4.accelerate(cmd_qcar4)
            
            link1.set_velocity(0, 0, 0, self.spawn_qcar2,
                               self.spawn_qcar3, self.spawn_qcar4)
            link2.set_velocity(
                0, 0, 0, '1', self.spawn_qcar3, self.spawn_qcar4)
            link3.set_velocity(
                0, 0, 0, '1', self.spawn_qcar2, self.spawn_qcar4)
            link4.set_velocity(
                0, 0, 0, '1', self.spawn_qcar2, self.spawn_qcar3)
            
            if self.progress == "Observing" and link1.isTraining == True:
                time.sleep(delay)  # 0.45)

            if link1.isTraining == False:
                time.sleep(delay)  # 0.45)

            # Update information (returns self.done_list)
            terminal_1 = link1.update_information()
            link2.update_information()
            link3.update_information()
            link4.update_information()

            self.step_for_newenv += 1
            
            stop = datetime.datetime.now()-start
            if stop < datetime.timedelta(seconds=delay):
                diff = (datetime.timedelta(seconds=delay)-stop).total_seconds()
                time.sleep(diff)

            if self.progress == 'Training' or self.progress == 'Stable Training':
                if self.step % 50 == 0 or self.step < 6000:
                    print('Training took', datetime.datetime.now()-start)
            else:
                if self.step % 50 == 0:
                    if self.progress == 'Observing':
                        print('Observing Took', datetime.datetime.now()-start)
                    else:
                        print('Testing Took', datetime.datetime.now()-start)

            # Reset environment
            if self.step_for_newenv == self.MAXSTEPS or terminal_1 == True:
                # stop all vehicles
                env.stop_all_cars()
                time.sleep(0.1)
                if self.step_for_newenv == self.MAXSTEPS:
                    print('Too Slow.....')

                link1.print_information()
                link2.print_information()
                link3.print_information()
                link4.print_information()

                print('new episode took', self.step_for_newenv)

                self.step_for_newenv = 0

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

                # Choose level - 1 or 2
                #self.level_qcar2 = str(np.random.choice([1,2]))
                #self.level_qcar3 = str(np.random.choice([1,2]))
                #self.level_qcar4 = str(np.random.choice([1,2]))
                # link2.update_model(self.level_qcar2)
                # link3.update_model(self.level_qcar3)
                # link4.update_model(self.level_qcar4)

                # Choose path for qcar1 - 4 is right turn, 5 is straight, 10 is left turn
                qcar1_path = str(np.random.choice([4, 5, 10]))
                link1.update_path(qcar1_path)
                link2.update_path(str(8), qcar1_path)
                link3.update_path(str(6), qcar1_path)
                link4.update_path(str(7), qcar1_path)

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

                if self.episode == self.MAXEPISODES or self.progress == "Finished":
                    print("Finished training!")
                    link1.save_model()
                    link1.save_fig()
                    break


if __name__ == '__main__':
    signal.signal(signal.SIGINT, quit)
    agent = DQN()
    agent.main()