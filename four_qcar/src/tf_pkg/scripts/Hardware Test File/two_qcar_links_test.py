#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Author: Wangcai
Date: 06/2019
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

from two_qcar_link1_test import DQN1
from two_qcar_link2_test import DQN2
from gazebo_env_two_qcar_links_test import envmodel
#from gazebo_env_two_qcar_link1_test import envmodel1
#from gazebo_env_two_qcar_link2_test import envmodel2

#import os
#os.environ["CUDA_VISIBLE_DEVICES"] = '0'  # 默认显卡0

link1 = DQN1()
link2 = DQN2()
env = envmodel()
#env1 = envmodel1()
#env2 = envmodel2()

qcarx_dict = {0: [0.0, 0.0], 1: [2.0, 0.0]}

class DQN:
    def __init__(self):
	rospy.init_node('control_node_main', anonymous=True)
        # Algorithm Information
        self.algorithm = 'D3QN_PER'

        self.Number='1'

        # Get parameters
        self.progress = ''
         
        # Initial parameters
        # ------------------------------
        self.Num_start_training = 0#5000
        self.Num_training       = 100000
        # ------------------------------
        self.Num_test           = 0

        self.learning_rate      = 0.0005
        self.Gamma              = 0.99

        # ------------------------------
        self.Epsilon            = 0.5
        # ------------------------------

        self.step    = 1
        self.score   = 0
        self.episode = 0

        # date - hour - minute - second of training time
        self.date_time = str(datetime.date.today())

        # Initialize agent robot
        self.agentrobot1 = 'qcar1'
        self.agentrobot2 = 'qcar2'

        # Define the distance from start point to goal point
        self.d = 4.0

        # Define the step for updating the environment
        self.MAXSTEPS = 80
        # ------------------------------
        self.MAXEPISODES = 100
        # ------------------------------
       
        link1.update_parameters(self.Num_start_training, self.Num_training, self.Num_test, self.learning_rate, self.Gamma, self.Epsilon)
        link2.update_parameters(self.Num_start_training, self.Num_training, self.Num_test, self.learning_rate, self.Gamma, self.Epsilon)
        
        link1.initialize_network()
        link2.initialize_network()
        
    def main(self):
        print("Main begins.")
        self.step_for_newenv = 0
        self.path = str(5)
        env.reset_env(path=self.path)
        link1.update_path(self.path)
        link2.update_path(self.path)
        
        link1.main_func()
        link2.main_func()
        
        # Training & Testing
        while True:
            action_qcar1 = link1.return_action()
            action_qcar2 = link2.return_action()
                        
            cmd_qcar1 = qcarx_dict[action_qcar1]
            cmd_qcar2 = qcarx_dict[action_qcar2]
            
            link1.move(cmd_qcar1)
            link2.move(cmd_qcar2)
            time.sleep(0.05)
            
            # Update information (returns self.done_list)
            terminal_1 = link1.update_information()
            terminal_2 = link2.update_information()
            
            self.step += 1
            self.step_for_newenv += 1
            
            # Reset environment
            if self.step_for_newenv == self.MAXSTEPS or terminal_1 == True:#
                # stop all vehicles
                link1.move([0, 0])
                link2.move([0, 0])
		time.sleep(0.5)

		link1.print_information()
		link2.print_information()
                
                self.step_for_newenv = 0
                self.episode += 1
                        
		# Choose path in [5, 6, 7, 8]
                self.path = str(np.random.choice([5,6,7,8]))
                link1.update_path(self.path)
                link2.update_path(self.path)


                
                env.reset_env(path=self.path)
                
                link1.new_environment()
                link2.new_environment()
                
                if self.episode == self.MAXEPISODES:
                    print("Finished testing!")
		    link1.end_func()
		    link2.end_func()
                    break
                
if __name__ == '__main__':
    agent = DQN()
    agent.main()
