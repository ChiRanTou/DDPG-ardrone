#! /usr/bin/env python

# test code
import rospy
import rospkg
import time
import random
from gazebo_connection import GazeboConnection
from drone_controller import DroneController
from optitrack import OptiTrack

if __name__ == '__main__':
    rospy.init_node('test_to_fly', anonymous=True)
    controller = DroneController()
    #controller.reset()
    # gazebo = GazeboConnection()
    # gazebo.set_location()
    controller.send_takeoff()
    time.sleep(3)
    # rospy.loginfo("Land Drone")
    controller.send_land()
    # controller.send_emergency()