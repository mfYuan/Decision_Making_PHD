#!/bin/bash
sudo killall rosmaster
sudo killall gzserver
sudo killall gzclient
source devel/setup.bash
roslaunch qcar qcar_ackermann_gazebo.launch world:=obstacle_sensor
