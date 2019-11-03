#!/usr/bin/env python

'''
    Training code made by Ricardo Tellez <rtellez@theconstructsim.com>
    Based on many other examples around Internet
    Visit our website at www.theconstruct.ai
'''
import time
import numpy as np
import random
import time
import dqn_learning
import gym
from gym import wrappers
import matplotlib.pyplot as plt
import os
import pandas as pd

# ROS packages required
import rospy
import rospkg

# import our training environment
from drone_controller import DroneController


save_path = 'saved_models_rohit_'
PATH = os.path.dirname(os.path.abspath(__file__))

if __name__ == '__main__':

    rospy.init_node('learn_fly', anonymous=True)
    # Create the Gym environment
    env = gym.make('QuadcopterLiveShow-v0')
    rospy.loginfo("Gym environment done")

    # Set the logging system
    rospack = rospkg.RosPack()
    pkg_path = rospack.get_path('drone_application')
    outdir = pkg_path + '/training_results'
    env = wrappers.Monitor(env, outdir, force=True)
    rospy.loginfo("Monitor Wrapper started")

    # Loads parameters from the ROS param server
    # Parameters are stored in a yaml file inside the config directory
    # They are loaded at runtime by the launch file
    Alpha = rospy.get_param("/alpha")
    Gamma = rospy.get_param("/gamma")
    Epsilon = rospy.get_param("/epsilon")
    Replace_Target_Iter = rospy.get_param("/replace_target_iter")
    Memory_Size = rospy.get_param("/memory_size")
    Batch_Size = rospy.get_param("/batch_size")
    nepisodes = rospy.get_param("/nepisodes")

    # Initialises the algorithm that we are going to use for learning
    dqnlearn = dqn_learning.DQN(n_actions=env.action_space.n, n_features=env.observation_space.shape[0],
                                learning_rate=Alpha,
                                gamma=Gamma, e_greedy=1, replace_target_iter=Replace_Target_Iter,
                                memory_size=Memory_Size, batch_size=Batch_Size)
    initial_epsilon = dqnlearn.epsilon

    ON_TRAIN = True


    def train():
        save_dir = os.path.join(PATH, save_path)
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        os.chdir(save_dir)

        highest_reward = 0
        all_reward = []
        episodes = []
        step = 0
        # Starts the main training loop: the one about the episodes to do
        for x in range(nepisodes):
            rospy.loginfo("STARTING Episode #" + str(x))

            cumulated_reward = 0

            done = False

            # Initialize the environment and get first state of the robot
            s = env.reset()

            # for each episode, we test the robot for nsteps
            while not done:

                step += 1
                # Pick an action based on the current state
                a = dqnlearn.choose_action(s)

                # Execute the action in the environment and get feedback
                s_, r, done, info = env.step(a)

                cumulated_reward += r

                # Put s,a,r,s_ into the memory
                dqnlearn.preceive(s, a, r, s_,done)

                s = s_

            all_reward.append(cumulated_reward)
            episodes.append(x + 1)

        ################ Plotting rewards ##############
        plt.figure(1)
        plt.plot(episodes, all_reward, 'b')
        plt.savefig("Reward Graph.png")

        ################ Plotting costs ##############
        plt.figure(2)
        dqnlearn.plot_cost()
        plt.savefig("cost.png")

        data1 = {'Reward': all_reward}
        data1 = pd.DataFrame(data1, columns=['Reward'])
        data1.to_csv(r'./Reward.csv', index=None)

        dqnlearn.save()
        env.close()


    def eval():
        dqnlearn.restore()
        s = drone.reset()
        while True:
            a = dqnlearn.choose_action(s)
            s, r, done = drone.step(a)


    if ON_TRAIN:
        train()
    else:
        eval()
