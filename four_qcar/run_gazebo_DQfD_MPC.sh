#!/bin/bash

sudo killall rosmaster
sudo killall gzserver
sudo killall gzclient
source devel/setup.bash
roslaunch qcar qcar_ackermann_gazebo_DQfD_MPC.launch 
