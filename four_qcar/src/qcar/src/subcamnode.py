#!/usr/bin/env python3

from __future__ import division, print_function, absolute_import

import roslib
import rospy
import numpy as np
import cv2
from qcar.q_essential import Camera2D

from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge, CvBridgeError

class SubCamNode(object):
	def __init__(self):
		super().__init__()
		self.bridge = CvBridge()
		self.cam_sub_r = rospy.Subscriber('/qcar/csi_right', Image, self.process_color_data, queue_size=1)
		# self.cam_sub_b = rospy.Subscriber('/qcar/csi_back', Image, self.process_color_data, queue_size=1)
		# self.cam_sub_l = rospy.Subscriber('/qcar/csi_left', Image, self.process_color_data, queue_size=1)
		# self.cam_sub_f = rospy.Subscriber('/qcar/csi_front', Image, self.process_color_data, queue_size=1)

#-----------------------------------------------------------------------------------------------------------------

	def process_color_data(self, cam_data):
		try:
			cv_img = self.bridge.imgmsg_to_cv2(cam_data, "bgr8")
			# do your stuff here...
			
		except CvBridgeError as e:
			print(e)
		cv2.imshow(str(cam_data), cv_img)
		cv2.waitKey(33)



if __name__ == '__main__':
	rospy.init_node('subcam_node')
	r = SubCamNode()

	rospy.spin()