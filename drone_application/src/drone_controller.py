#! /usr/bin/env python

# Parrot AR Drone Python Controller
import rospy
import numpy as np
import time
import tf
import random
import gym


from geometry_msgs.msg import Twist, Pose, PoseStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import Empty
from gazebo_connection import GazeboConnection
from cvg_sim_msgs.msg import Altimeter
from ardrone_autonomy.msg import Navdata
from gym import spaces
from gym.utils import seeding
from gym.envs.registration import register
from ddpg_learning import OUNoise

COMMAND_PERIOD = 100  # ms

# register the training environment in the gym as an available one
reg = register(
    id='QuadcopterLiveShow-v0',
    entry_point='drone_controller:DroneController',
    timestep_limit=50,
)

class DroneController(gym.Env):

    # Initialize the commands
    def __init__(self):
        # Subscribe to the /ardrone/navdata topic, of message type navdata, and call self.ReceiveNavdata when a message is received
        self.subNavdata = rospy.Subscriber('/ardrone/navdata', Navdata, self.receive_navdata)
        self.subOdom = rospy.Subscriber('/ground_truth/state', Odometry, self.odom_callback)
        rospy.Subscriber('/vrpn_client_node/TestTed/pose', PoseStamped, self.optictrack_callback)

        # Allow the controller to publish to the /ardrone/takeoff, land and reset topics
        self.pubLand = rospy.Publisher('/ardrone/land', Empty, queue_size=0)
        self.pubTakeoff = rospy.Publisher('/ardrone/takeoff', Empty, queue_size=0)
        self.pubReset = rospy.Publisher('/ardrone/reset', Empty, queue_size=1)

        # Allow the controller to publish to the /cmd_vel topic and thus control the drone
        self.pubCommand = rospy.Publisher('/cmd_vel', Twist, queue_size=1)

        # Put location into odometry
        self.location = Odometry()
        self.status = Navdata()
        self.real_loc = PoseStamped()
        # Gets parameters from param server
        self.speed_value = rospy.get_param("/speed_value")
        self.run_step = rospy.get_param("/run_step")
        self.desired_pose = Pose()
        self.desired_pose.position.z = rospy.get_param("/desired_pose/z")
        self.desired_pose.position.x = rospy.get_param("/desired_pose/x")
        self.desired_pose.position.y = rospy.get_param("/desired_pose/y")
        self.max_incl = rospy.get_param("/max_incl")
        self.max_altitude = rospy.get_param("/max_altitude")
        self.exporation_noise = OUNoise()
        self.on_place = 0

        # initialize action space
        # Forward,Left,Right,Up,Down
        #self.action_space = spaces.Discrete(8)
        self.action_space = spaces.Box(np.array((-0.5, -0.5, -0.5, -0.5)), np.array((0.5, 0.5, 0.5, 0.5)))
        # Gazebo Connection
        self.gazebo = GazeboConnection()

        self._seed()
        # Land the drone if we are shutting down
        rospy.on_shutdown(self.send_land)

    # ---------------------- Basic Functions ---------------------- #
    # Receive the navigation data
    def receive_navdata(self, navdata):
        # Although there is a lot of data in this packet, we're only interested in the state at the moment
        self.status = navdata
        # print '[{0:.3f}] X: {1:.3f}'.format(navdata.header.stamp.to_sec(), navdata.vx)

    # call back odo data to subscriber
    def odom_callback(self, data):
        self.location = data.pose.pose

    # call back the position and orientation to subscriber
    def optictrack_callback(self, data):
        self.real_loc = data
        self.x_real = self.real_loc.pose.position.x  # along the short length of space
        self.y_real = self.real_loc.pose.position.y  # along the long hall
        self.z_real = self.real_loc.pose.position.z  # up and down
        self.imu_x_real = self.real_loc.pose.orientation.x
        self.imu_y_real = self.real_loc.pose.orientation.y
        self.imu_z_real = self.real_loc.pose.orientation.z
        self.imu_w_real = self.real_loc.pose.orientation.w
        # self.x_real, self.y_real = self.y_real, self.x_real
        euler = tf.transformations.euler_from_quaternion(
            [self.imu_x_real, self.imu_y_real, self.imu_z_real, self.imu_w_real])
        roll = euler[0]  # pitch
        pitch = euler[1]  # roll
        yaw = euler[2]  # yaw

        rospy.loginfo(("pitch: " + str(roll)))
    # take observation for pose and orientation
    # orientation is used for calculate euler angle
    def take_observation(self):

        data_pose = self.location.position
        data_imu = self.location.orientation
        state = self.status
        return data_pose, data_imu, state

    # Takeoff
    def send_takeoff(self):
        # Send a takeoff message to the ardrone driver
        self.reset_action()
        time.sleep(0.5)
        self.pubTakeoff.publish(Empty())

    # Send Land Message
    def send_land(self):
        # Send a landing message to the ardrone driver
        # Note we send this in all states, landing can do no harm
        self.reset_action()
        time.sleep(0.5)
        self.pubLand.publish(Empty())

    # Send emergency message to drone
    def send_emergency(self):
        # Send an emergency (or reset) message to the ardrone driver
        self.pubReset.publish(Empty())

    # Take Action
    def take_action(self, roll, pitch, z_velocity, yaw_velocity=0, something1=0, something2=0):

        # Called by the main program to set the current command
        command = Twist()
        command.linear.x = pitch  # go y direction / green axis
        command.linear.y = roll  # go x direction / red axis
        command.linear.z = z_velocity  # Spinning anti-clockwise
        command.angular.z = yaw_velocity  # go z direction / blue axis
        command.angular.x = something1
        command.angular.y = something2

        self.pubCommand.publish(command)

    # ---------------------- Initialize Simulation ---------------------- #
    # reset command action and takeoff the drone
    def takeoff_sequence(self, seconds_taking_off=2):
        # Before taking off be sure that cmd_vel value there is is null to avoid drifts
        self.reset_action()

        # wait for 1 seconds
        time.sleep(1)
        rospy.loginfo("Taking-Off Start")
        self.send_takeoff()
        time.sleep(seconds_taking_off)
        rospy.loginfo("Taking-Off sequence completed")

    # Check if any topic is published
    def check_topic_publishers_connection(self):

        rate = rospy.Rate(10)  # 10hz
        while self.pubTakeoff.get_num_connections() == 0:
            rospy.loginfo("No susbribers to Takeoff yet so we wait and try again")
            rate.sleep()
        rospy.loginfo("Takeoff Publisher Connected")

        while self.pubCommand.get_num_connections() == 0:
            rospy.loginfo("No susbribers to Cmd_vel yet so we wait and try again")
            rate.sleep()
        rospy.loginfo("Cmd_vel Publisher Connected")

    # Reset the action
    def reset_action(self):
        self.take_action(0, 0, 0)

    # A function to initialize the random generator
    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    # Reset the simulation
    def _reset(self):
        # reset action
        self.reset_action()
        self.send_land()
        time.sleep(0.6)
        self.exporation_noise.reset()
        time.sleep(1)
        # reset the pose on the simulation
        self.gazebo.resetSim()
        self.gazebo.set_location()

        self.gazebo.unpauseSim()
        # check any topic published
        self.check_topic_publishers_connection()

        # takeoff the drone
        self.takeoff_sequence()

        # get the distance from origin to desire position before takeoff
        self.init_desired_pose()

        # take observation
        data_pose, data_imu, vel = self.take_observation()

        # observe the x position for now
        pos_x, pos_y, pos_z = data_pose.x, data_pose.y, data_pose.z

        pos = [pos_x, pos_y, pos_z]

        state = np.concatenate((pos, [1. if self.on_place else 0.]))
        #self.gazebo.pauseSim()
        return state

    # ---------------------- Action Processing ---------------------- #
    # take an action from action space [0,5]
    def _step(self, action):

        if action == 0:  # FORWARD
            self.take_action(0, self.speed_value, 0)
        elif action == 1:  # BACKWARD
            self.take_action(0, -self.speed_value, 0)
        elif action == 2:  # LEFT
            self.take_action(self.speed_value, 0, 0)
        elif action == 3:  # RIGHT
            self.take_action(-self.speed_value, 0, 0)
        elif action == 4:  # Up
            self.take_action(0, 0, self.speed_value)
        elif action == 5:  # Down
            self.take_action(0, 0, -self.speed_value)
        elif action == 6:
            self.take_action(0, 0, 0, 0)
        #self.take_action(action[0], action[1], action[2], action[3])
        time.sleep(self.run_step)

        # get the now state
        self.data_pose, self.data_imu, self.vel = self.take_observation()

        # finally we get an evaluation based on what happened in the sim
        reward, done = self.process_data(self.data_pose, self.data_imu)

        current_dist, loc_dist = self.calculate_dist_between_two_Points(self.data_pose, self.desired_pose.position)

        if current_dist <= 0.35:
            reward += 10
            self.on_place += 1
            if self.on_place > 70:
                done = True
                reward += 50
        elif current_dist > 5:
            reward -= 25
            self.on_place = 0
        else:
            reward += 5 - (current_dist / 0.2)
            self.on_place = 0
        if (loc_dist[0] < 0.4 or loc_dist[1] < 0.4 or loc_dist[2] < 0.4):
            reward += 1
        elif (loc_dist[0] < 0.1 and loc_dist[1] < 0.1 and loc_dist[2] < 0.1):
            reward += 5

        pos_x, pos_y, pos_z = self.data_pose.x, self.data_pose.y, self.data_pose.z

        pos = [pos_x, pos_y, pos_z]

        state = np.concatenate((pos, [1. if self.on_place else 0.]))
        return state, reward, done, {}

    # calculate the distance between two location
    def calculate_dist_between_two_Points(self, p_init, p_end):
        a = np.array((p_init.x, p_init.y, p_init.z))
        b = np.array((p_end.x, p_end.y, p_end.z))

        dist_loc = abs(b - a)
        dist = np.linalg.norm(a - b)

        return dist, dist_loc

    # initialize the desired pose
    def init_desired_pose(self):

        current_init_pose, imu, vel = self.take_observation()

        self.best_dist = self.calculate_dist_between_two_Points(current_init_pose, self.desired_pose.position)

    # imporve the reward if the drone keep in desire distance
    def improved_distance_reward(self, current_pose):
        current_dist = self.calculate_dist_between_two_Points(current_pose, self.desired_pose.position)
        # rospy.loginfo("Calculated Distance = "+str(current_dist))

        if current_dist < self.best_dist:
            reward = 10
            self.best_dist = current_dist
        elif current_dist == self.best_dist:
            reward = 0
        else:
            reward = -10
            # print "Made Distance bigger= "+str(self.best_dist)

        return reward

    def process_data(self, data_position, data_imu):

        done = False

        euler = tf.transformations.euler_from_quaternion(
            [data_imu.x, data_imu.y, data_imu.z, data_imu.w])
        roll = euler[0]
        pitch = euler[1]
        yaw = euler[2]

        pitch_bad = not (-self.max_incl < pitch < self.max_incl)
        roll_bad = not (-self.max_incl < roll < self.max_incl)
        altitude_too_high = data_position.z > self.max_altitude
        altitude_too_low = data_position.z < 0.3

        if altitude_too_high or pitch_bad or roll_bad or altitude_too_low:
            rospy.loginfo("(Drone flight status is wrong) >>> (" + str(altitude_too_high) + "," + str(pitch_bad) +
                          "," + str(roll_bad) + "," + str(altitude_too_low) + ")")
            done = True
            self.on_place = 0
            reward = -100
        else:
            reward = self.improved_distance_reward(data_position)
            #reward = 0
        return reward, done




