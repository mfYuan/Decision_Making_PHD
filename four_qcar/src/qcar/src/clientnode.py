#!/usr/bin/env python3

from __future__ import division, print_function, absolute_import

import roslib
import rospy
import numpy as np
import cv2
from qcar.q_misc import BasicStream 
import time
import math 

from geometry_msgs.msg import PoseStamped

class ClientNode(object):
	def __init__(self):
		super().__init__()
		self.myClient = BasicStream('tcpip://localhost:18002', agent='c', send_buffer_size=7 * 8)
		self.prev_con = False
		self.pose_sub = rospy.Subscriber('/slam_out_pose', PoseStamped, self.send_pose, queue_size=10)
		
			
	
	
	# ---------------------------------------------------------------------------------------------------------
	def send_pose(self, pose_data):
		if not self.myClient.connected:
				self.myClient.checkConnection()
		
		if self.myClient.connected and not self.prev_con:
			print('Connection to Server was successful.')
			self.prev_con = self.myClient.connected
			

		if self.myClient.connected:
			x = pose_data.pose.position.x
			y = pose_data.pose.position.y
			z = pose_data.pose.position.z

			qx = pose_data.pose.orientation.x
			qy = pose_data.pose.orientation.y
			qz = pose_data.pose.orientation.z
			qw = pose_data.pose.orientation.w

			pose = np.array([x, y, z, qx, qy, qz, qw])
			print(x, y, z, qx, qy, qz, qw)
			self.myClient.send(pose)

if __name__ == '__main__':
	rospy.init_node('client_node')
	r = ClientNode()

	rospy.spin()

