#!/usr/bin/env python3

from __future__ import division, print_function, absolute_import

from qcar.q_ui import gamepadViaTarget

import roslib
import rospy
import numpy as np
import pygame
import time

from ackermann_msgs.msg import AckermannDriveStamped
from std_msgs.msg import Float64


class GazeboControlNode(object):
	def __init__(self):
		super().__init__()
		self.cmd_gzb_pub_ = rospy.Publisher('/ackermann_cmd', AckermannDriveStamped, queue_size=100)
		self.gpad = gamepadViaTarget(8)
		while True:
			new = self.gpad.read()
			# left_lateral, left_longitudinal, right_lateral, right_longitudinal, LT, RT, A, B, X, Y, LB, RB, BACK, START, Logitech, hat = self.gamepad_io_qcar() # .................... Logitech......................
			
			pose = self.control_from_gamepad(new, self.gpad.LB, self.gpad.RT, self.gpad.LLA, self.gpad.A)
			self.process_command(pose)
			time.sleep(0.01)

	def process_command(self, pose):
		pub_cmd = AckermannDriveStamped()
		pub_cmd.header.stamp = rospy.Time.now()
		pub_cmd.header.frame_id = "command_input"
		pub_cmd.drive.steering_angle = float(pose[1])
		pub_cmd.drive.speed = float(pose[0]) * 30.0
		self.cmd_gzb_pub_.publish(pub_cmd)


	def control_from_gamepad(self, new, LB, RT, left_lateral, A):
		if LB and new:
			if A == 1 :
				throttle_axis = -(RT + 1) /2 * 0.1 #going backward
				steering_axis = -left_lateral * 0.5 
			else:
				throttle_axis = (RT + 1) /2 * 0.3 #going forward
				steering_axis = -left_lateral * 0.5
		else:
			throttle_axis = 0 
			steering_axis = 0

		command = np.array([throttle_axis, steering_axis])
		return command

if __name__ == '__main__':
	rospy.init_node('gazebo_control')
	r = GazeboControlNode()

	rospy.spin()
