#!/usr/bin/env python

# Parrot AR Drone Python Controller
import rospy
import numpy as np
import time
import tf
import random
import gym

from geometry_msgs.msg import Twist, Pose, PoseStamped
from visualization_msgs.msg import Marker
from nav_msgs.msg import Odometry
from std_msgs.msg import Empty
from gazebo_connection import GazeboConnection
from cvg_sim_msgs.msg import Altimeter
from ardrone_autonomy.msg import Navdata
from gym import spaces
from gym.utils import seeding
from gym.envs.registration import register

COMMAND_PERIOD = 100  # ms

# register the training environment in the gym as an available one
reg = register(
    id='QuadcopterLiveShow-v0',
    entry_point='drone_controller:DroneController',
    timestep_limit=200,
)


def get_on_goal():
    return self.count_on_place


class DroneController(gym.Env):

    # Initialize the commands
    def __init__(self):

        # initialise state
        # ground truth position
        self.x = 0
        self.y = 0
        self.z = 0
        # ground truth velocity
        self.x_dot = 0
        self.y_dot = 0
        self.z_dot = 0
        # ground truth quaternion
        self.imu_x = 0  # q0
        self.imu_y = 0  # q1
        self.imu_z = 0  # q2
        self.imu_w = 0  # q3
        # nav drone angle
        self.rot_x = 0  # roll
        self.rot_y = 0  # pitch
        self.rot_z = 0  # yaw
        # Optitrack Information
        self.x_real = 0
        self.y_real = 0
        self.z_real = 0

        self.dist = 0
        # Subscribe to the /ardrone/navdata topic, of message type navdata, and call self.ReceiveNavdata when a message is received
        self.subNavdata = rospy.Subscriber('/ardrone/navdata', Navdata, self.receive_navdata)
        self.subOdom = rospy.Subscriber('/ground_truth/state', Odometry, self.odom_callback)
        # rospy.Subscriber('/vrpn_client_node/Rigid_grant/pose', PoseStamped, self.optictrack_callback)

        # Allow the controller to publish to the /ardrone/takeoff, land and reset topics
        self.pubLand = rospy.Publisher('/ardrone/land', Empty, queue_size=0)
        self.pubTakeoff = rospy.Publisher('/ardrone/takeoff', Empty, queue_size=0)
        self.pubReset = rospy.Publisher('/ardrone/reset', Empty, queue_size=1)

        # Allow the controller to publish to the /cmd_vel topic and thus control the drone
        self.pubCommand = rospy.Publisher('/cmd_vel', Twist, queue_size=1)

        # Put location into odometry
        self.location = Odometry()
        self.status = Navdata()
        self.loc_real = PoseStamped()

        # Gets parameters from param server
        self.speed_value = rospy.get_param("/speed_value")
        self.run_step = rospy.get_param("/run_step")
        self.target_x = rospy.get_param("/desired_pose/x")
        self.target_y = rospy.get_param("/desired_pose/y")
        self.target_z = rospy.get_param("/desired_pose/z")
        self.desired_pose = np.array([self.target_x, self.target_y, self.target_z])
        # self.desired_pose = [0, 0, 1.5]
        self.max_incl = rospy.get_param("/max_incl")
        self.max_altitude = rospy.get_param("/max_altitude")
        self.min_altitude = rospy.get_param("/min_altitude")
        self.x_max = rospy.get_param("/max_pose/max_x")
        self.y_max = rospy.get_param("/max_pose/max_y")

        self.on_place = 0
        self.count_on_place = 0

        # initialize action space
        # Forward,Left,Right,Up,Down
        # self.action_space = spaces.Discrete(8)
        self.action_space = spaces.Box(-0.3, 0.3, (3,))
        self.up_bound = np.array([np.inf, np.inf, np.inf, np.inf, 1])
        self.low_bound = np.array([-np.inf, -np.inf, -np.inf, -np.inf, 0])
        self.observation_space = spaces.Box(self.low_bound, self.up_bound)  # position[x,y,z], linear velocity[x,y,z]
        #self.observation_space = spaces.Box(-np.inf, np.inf, (9,))
        # Gazebo Connection
        self.gazebo = GazeboConnection()

        self._seed()
        # Land the drone if we are shutting down
        rospy.on_shutdown(self.send_land)

    # ---------------------- Basic Functions ---------------------- #
    # Receive the navigation data
    def receive_navdata(self, navdata):

        self.rot_x = navdata.rotX
        self.rot_y = navdata.rotY
        self.rot_z = navdata.rotZ

    # call back odo data to subscriber
    def odom_callback(self, data):

        self.x = data.pose.pose.position.x
        self.y = data.pose.pose.position.y
        self.z = data.pose.pose.position.z
        self.imu_x = data.pose.pose.orientation.x
        self.imu_y = data.pose.pose.orientation.y
        self.imu_z = data.pose.pose.orientation.z
        self.imu_w = data.pose.pose.orientation.w
        self.x_dot = data.twist.twist.linear.x
        self.y_dot = data.twist.twist.linear.y
        self.z_dot = data.twist.twist.linear.z

    # call back the position and orientation to subscriber
    def optictrack_callback(self, data):

        self.x_real = data.pose.position.x
        self.y_real = data.pose.position.y
        self.z_real = data.pose.position.z
        # self.imu_x_real = opt_data.pose.position.x
        # self.imu_y_real = opt_data.pose.position.y
        # self.imu_z_real = opt_data.pose.position.z
        # self.imu_w_real = opt_data.pose.position.w

    # take observation for pose and orientation
    # orientation is used for calculate euler angle
    def take_observation(self):
        # state = [self.x, self.y, self.z, self.x_dot, self.y_dot, self.z_dot, self.rot_x, self.rot_y, self.rot_z]
        state = [self.x, self.y, self.z]
        return state

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
    def take_action(self, roll, pitch, z_velocity, yaw_velocity=0):

        # Called by the main program to set the current command
        command = Twist()
        command.linear.x = pitch  # go y direction / green axis
        command.linear.y = roll  # go x direction / red axis
        command.linear.z = z_velocity  # go z direction / blue axis
        command.angular.x = 0
        command.angular.y = 0
        command.angular.z = yaw_velocity  # Spinning anti-clockwise

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
        self.reset_action()
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
        self.gazebo.unpauseSim()
        # reset action
        self.reset_action()

        self.on_place = 0
        self.count_on_place = 0

        self.send_land()
        time.sleep(0.6)

        time.sleep(1)
        # reset the pose on the simulation
        self.gazebo.resetSim()
        self.gazebo.set_location()

        # check any topic published
        self.check_topic_publishers_connection()

        # takeoff the drone
        self.takeoff_sequence()

        # take observation
        state = self.take_observation()

        init_pose = [state[0], state[1], state[2]]
        self.dist = self.calculate_dist(init_pose)
        self.prev_pose = init_pose

        self.gazebo.pauseSim()

        state = np.concatenate((state, [self.dist], [1. if self.on_place else 0.]))
        return state

    # ---------------------- Action Processing ---------------------- #
    # take an action from action space [0,5]
    def _step(self, action):

        self.gazebo.unpauseSim()
        self.take_action(action[0], action[1], action[2])

        time.sleep(self.run_step)

        state = self.take_observation()
        self.gazebo.pauseSim()

        data_pose = np.array([state[0], state[1], state[2]])
        data_imu = [self.imu_x, self.imu_y, self.imu_z, self.imu_w]
        reward, done, self.dist = self.process_data(data_pose, data_imu)

        self.prev_pose = data_pose
        state = np.concatenate((state, [self.dist], [1. if self.on_place else 0.]))
        return state, reward, done, {}

    # calculate the distance between two location
    def calculate_dist(self, data_pose):
        # err = np.subtract(data_pose, self.desired_pose)
        # w = np.array([1, 1, 4])
        # err = np.multiply(w, err)
        # dist = np.linalg.norm(err)
        x_dist = data_pose[0] - self.desired_pose[0]
        y_dist = data_pose[1] - self.desired_pose[1]
        z_dist = data_pose[2] - self.desired_pose[2]
        self.dist = np.sqrt(x_dist**2 + y_dist**2 + z_dist**2)
        return self.dist

    def get_reward(self, data_pose):
        reward = 0
        reach_goal = False

        self.dist = self.calculate_dist(data_pose)
        #rospy.loginfo(str(self.dist))
        reward -= 0.03 * self.dist
        # current_pose = [data_pose[0], data_pose[1], data_pose[2]]
        if self.dist < 0.45:
            reward += 1
            self.on_place += 1
            self.count_on_place += 1
            if self.on_place > 100:
                reach_goal = True
        else:
            self.on_place = 0

        return reward, reach_goal, self.dist

    def process_data(self, data_pose, data_imu):
        done = False

        euler = tf.transformations.euler_from_quaternion(
            [data_imu[0], data_imu[1], data_imu[2], data_imu[3]])
        roll = euler[0]
        pitch = euler[1]
        yaw = euler[2]

        pitch_bad = not (-self.max_incl < pitch < self.max_incl)
        roll_bad = not (-self.max_incl < roll < self.max_incl)
        altitude_bad = not (self.min_altitude < data_pose[2] < self.max_altitude)
        x_bad = not (-self.x_max < data_pose[0] < self.x_max)
        y_bad = not (-self.y_max < data_pose[1] < self.y_max)

        if altitude_bad or pitch_bad or roll_bad or x_bad or y_bad:
            rospy.loginfo("(Drone flight status is wrong) >> " + "[" + str(altitude_bad) + "," + str(pitch_bad) + "," +
                          str(roll_bad) + "," + str(x_bad) + "," + str(y_bad) + "]")
            done = True
            reward = -5
        else:
            reward, reach_goal, self.dist = self.get_reward(data_pose)
            if reach_goal:
                done = True

        return reward, done, self.dist
