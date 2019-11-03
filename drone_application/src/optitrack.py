#! /usr/bin/env python

import rospy

# Importing messages
from visualization_msgs.msg import Marker  # for receiving marker detection
from geometry_msgs.msg import PoseStamped


class OptiTrack:
    def __init__(self):
        self.pos_x = 0
        self.pos_y = 0
        self.pos_z = 0

        rospy.Subscriber('/vrpn_client_node/TestTed/pose', PoseStamped, self.get_pose)

        self.locaiton = PoseStamped()
        self.publisher = rospy.Publisher('/ardrone/pose', PoseStamped, queue_size=10)

    def get_pose(self, data):
        # Drone Position
        self.pos_x = data.pose.position.x
        self.pos_y = data.pose.position.y
        self.pos_z = data.pose.position.z

        rospy.loginfo((str(self.pos_x) + " " + str(self.pos_y) + " " + str(self.pos_z)))

        # print '[{0:.3f}] X: {1:.3f}'.format(data.header.stamp.to_sec(), data.pose.position.x)

    def get_destination(self, data):
        destination_x = data.pose.position.x
        destination_y = data.pose.position.y
        destination_z = data.pose.position.z

    def publish_value(self):
        self.location.pose.position.x = self.pose_x
        self.location.pose.position.y = self.pose_y
        self.location.pose.position.z = self.pose_z
        self.publisher.publish(self.location)
