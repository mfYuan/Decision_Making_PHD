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

from qcar_ddpg import DDPG
from two_qcar_link2_v2 import DQN2
from two_qcar_link3_v2 import DQN3
from gazebo_env_two_qcar_links import envmodel
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = '0'  # 默认显卡0

link1 = DDPG()
link2 = DQN2()
link3 = DQN3()
env = envmodel()

#qcar1_dict = {0: -1.0, 1: -0.5, 2: 0.0, 3: 0.5, 4: 1.0}
qcar1_dict = {0: [0.0, 0.0], 1: [1.5, 0.0]}
qcarx_dict = {0: [0.3, 0.0], 1: [1.5, 0.0]}


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
        self.Num_start_training = 0 # 5000
        self.Num_training = 40000 # 40000
        # ------------------------------
        self.Num_test = 40000

        self.learning_rate = 0.0001
        self.Gamma = 0.90

        # ------------------------------
        self.Epsilon = 0.5
        self.Final_epsilon = 0.1
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

        # Define the distance from start point to goal point
        self.d = 4.0

        # Define the step for updating the environment
        self.MAXSTEPS = 300
        # ------------------------------
        if self.Task == 'training':
            self.MAXEPISODES = 4000
        else:
            self.MAXEPISODES = 2000

        # ------------------------------

        link1.update_parameters(self.Num_start_training, self.Num_training,
                                self.Num_test, self.learning_rate, self.Gamma, self.MAXEPISODES)
        link2.update_parameters(self.Num_start_training, self.Num_training, self.Num_test,
                                self.learning_rate, self.Gamma, self.Epsilon, self.Final_epsilon, self.MAXEPISODES)
        link3.update_parameters(self.Num_start_training, self.Num_training, self.Num_test,
                                self.learning_rate, self.Gamma, self.Epsilon, self.Final_epsilon, self.MAXEPISODES)

        # link1.initialize_network()
        link2.initialize_network()
        link3.initialize_network()

    def main(self):
        print("Main begins.")
        self.step_for_newenv = 0

        # 0 is no, 1 is yes
        self.spawn_qcar2 = str(1)
        self.spawn_qcar3 = str(1)

        # Available levels: 1 and 2
        self.level_qcar2 = str(1)
        self.level_qcar3 = str(1)
        link2.update_model(self.level_qcar2)
        link3.update_model(self.level_qcar3)

        env.reset_env(self.spawn_qcar2, self.spawn_qcar3)
        link1.update_path(str(5))
        link2.update_path(str(8))
        link3.update_path(str(6))

        '''env.reset_env(path=self.path)
        link1.update_path(self.path)
        link2.update_path(self.path)
        link3.update_path(self.path)'''

        link1.main_func()
        link2.main_func()
        link3.main_func()

        # Training & Testing
        while True:
            progress = link1.get_progress(self.step)
            # This can also return link2's progress if needed
            link2.get_progress(self.step, self.Epsilon)
            # This can also return link3's progress if needed
            link3.get_progress(self.step, self.Epsilon)

            self.progress = progress

            action_qcar1 = link1.return_action()

            action_qcar2 = link2.return_action()
            action_qcar3 = link3.return_action()

            cmd_qcar1 = [action_qcar1, 0.0]
            print(action_qcar1)
            cmd_qcar2 = qcarx_dict[action_qcar2]
            cmd_qcar3 = qcarx_dict[action_qcar3]

            #link1.accelerate(cmd_qcar1) ######
            link1.move(cmd_qcar1)
            link2.move(cmd_qcar2)
            link3.move(cmd_qcar3)
            time.sleep(0.05)

            # Update information (returns self.done_list)
            terminal_1 = link1.update_information()
            terminal_2 = link2.update_information()
            terminal_3 = link3.update_information()

            self.step += 1
            self.step_for_newenv += 1

            # Reset environment
            if self.step_for_newenv >= self.MAXSTEPS or terminal_1 == True:
                # stop all vehicles
                if self.step_for_newenv == self.MAXSTEPS:
                    print('.....too slow......')
                env.stop_all_cars()
                time.sleep(0.1)

                if self.episode % 50 == 0:
                    link1.save_models()
                    # link2.save_model()
                    # link3.save_model()

                link1.print_information()
                link2.print_information()
                link3.print_information()

                self.step_for_newenv = 0

                if self.progress != 'Observing':
                    self.episode += 1

                # Choose whether to spawn vehicle - 0 is no, 1 is yes
                self.spawn_qcar2 = str(np.random.choice([0, 1]))
                self.spawn_qcar3 = str(np.random.choice([0, 1]))

                # Choose level - 1 or 2
                self.level_qcar2 = str(np.random.choice([1, 2]))
                self.level_qcar3 = str(np.random.choice([1, 2]))
                link2.update_model(self.level_qcar2)
                link3.update_model(self.level_qcar3)

                # Choose path for qcar1 - 4 is right turn, 5 is straight, 10 is left turn
                link1.update_path(str(np.random.choice([4, 5, 10])))
                link2.update_path(str(8))
                link3.update_path(str(6))

                env.reset_env(self.spawn_qcar2, self.spawn_qcar3)

                link1.new_environment()
                link2.new_environment()
                link3.new_environment()

                if self.episode >= self.MAXEPISODES or self.progress == 'Finished':
                    print("Finished training!")
                    link1.save_models()
                    # link2.save_model()
                    # link3.save_model()

                    link1.save_fig()
                    link2.save_fig()
                    link3.save_fig()
                    break


if __name__ == '__main__':
    agent = DQN()
    agent.main()

