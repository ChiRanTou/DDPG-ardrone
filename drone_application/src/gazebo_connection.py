#!/usr/bin/env python

import rospy
import numpy as np
import random
from std_srvs.srv import Empty
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SetModelState

class GazeboConnection():
    
    def __init__(self):

        self.state_msg = ModelState()
        self.state_msg.model_name = 'quadrotor'
        self.state_msg.pose.position.x = 0
        self.state_msg.pose.position.y = 0
        self.state_msg.pose.position.z = 0
        self.state_msg.pose.orientation.x = 0
        self.state_msg.pose.orientation.y = 0
        self.state_msg.pose.orientation.z = 0
        self.state_msg.pose.orientation.w = 0
        self.unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        self.pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        self.reset_model = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
        self.reset_world = rospy.ServiceProxy('/gazebo/reset_world', Empty)
    def pauseSim(self):
        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            self.pause()
        except rospy.ServiceException, e:
            print ("/gazebo/pause_physics service call failed")
        
    def unpauseSim(self):
        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            self.unpause()
        except rospy.ServiceException, e:
            print ("/gazebo/unpause_physics service call failed")
        
    def resetSim(self):
        rospy.wait_for_service('/gazebo/set_model_state')
        rospy.wait_for_service('/gazebo/reset_world')
        try:
            self.reset_world()
            self.reset_model(self.state_msg)
        except rospy.ServiceException, e:
            print("/gazebo/reset_world service call failed")

    def set_location(self):
        random_pos = list(np.random.uniform(-1.5, 1.5, size=(2,)))
        pos_x, pos_y= random_pos[0], random_pos[1]

        self.state_msg.pose.position.x = pos_x
        self.state_msg.pose.position.y = pos_y
        rospy.wait_for_service('/gazebo/set_model_state')
        try:
            self.reset_model(self.state_msg)
        except rospy.ServiceException, e:
            print ("/gazebo/reset_simulation service call failed")
