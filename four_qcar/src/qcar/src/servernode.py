#!/usr/bin/env python3

from __future__ import division, print_function, absolute_import

import roslib
import rospy
import numpy as np
import cv2
from qcar.q_misc import BasicStream 
import time
import math 

from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import PoseStamped

class ServerNode(object):
	def __init__(self):
		super().__init__()
		self.lidar_pub = rospy.Publisher('/scan', LaserScan, queue_size=10000)

		self.num_measurements = 720
		self.scans = np.zeros((self.num_measurements,1), dtype= np.float32)
		self.pose_num = 7

		self.myServer = BasicStream('tcpip://localhost:18002', agent='s', send_buffer_size = self.pose_num * 8, recv_buffer_size=self.num_measurements * 4)
		self.prev_con = False

		while True:
			if not self.myServer.connected:
				self.myServer.checkConnection()

			if self.myServer.connected and not self.prev_con:
				print('Connection to Client was successful.')
			self.prev_con = self.myServer.connected

			if self.myServer.connected:
				starTime = time.time()
				lidar_data, bytes_received = self.myServer.receive(self.scans)
				if bytes_received == 0:
					print('Client stopped sending data over.')
					break
				scan_time = time.time() - starTime
				# lidar_data = lidar_data.astype(np.float32)
				self.process_lidar_data(self.lidar_pub, lidar_data, self.num_measurements, scan_time)

				self.pose_sub = rospy.Subscriber('/slam_out_pose', PoseStamped, self.send_pose, queue_size=1)

			
#--------------------------------------------------------------------------------------------------------------
	def process_lidar_data(self, pub_lidar, distances, num_measurement, scan_times):

		scan = LaserScan()
		scan.header.stamp = rospy.Time.now()
		scan.header.frame_id = 'lidar'
		scan.angle_min = 0.0
		scan.angle_max = 6.2744
		scan.angle_increment = 6.2744 / num_measurement
		scan.time_increment = scan_times / num_measurement
		scan.scan_time = scan_times
		scan.range_min = 0.15
		scan.range_max = 12.0
		scan.ranges = distances.tolist()
		# for i in range(num_measurement):
		# 	scan.ranges.append(distances[i])
		pub_lidar.publish(scan)

	def send_pose(self, pose_data):
		x = pose_data.pose.position.x
		y = pose_data.pose.position.y
		z = pose_data.pose.position.z

		qx = pose_data.pose.orientation.x
		qy = pose_data.pose.orientation.y
		qz = pose_data.pose.orientation.z
		qw = pose_data.pose.orientation.w

		pose = np.array([x, y, z, qx, qy, qz, qw])
		self.myServer.send(pose)

if __name__ == '__main__':
	rospy.init_node('server_node')
	r = ServerNode()

	rospy.spin()






			

