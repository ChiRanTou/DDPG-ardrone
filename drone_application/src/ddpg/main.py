#!/usr/bin/env python
import gym
from gym import wrappers

import time
import numpy as np
# from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random
import tensorflow as tf
import os
from ddpg import DDPG
from drone_controller import DroneController
from ou_noise import OUNoise
import rospy
import rospkg
from actor_network import Actor_Network
import pandas as pd

timestr = time.strftime("%Y%m%d-%H%M%S")
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

    NEPISODES = rospy.get_param("/nepisodes")
    NSTEPS = rospy.get_param("/nsteps")

    # Train or not
    Running = ['Train', 'Play_Sim', 'Play_Real']
    go = Running[1]


    def train():
        save_dir = os.path.join(PATH, save_path)
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        os.chdir(save_dir)

        agent = DDPG(env)
        episode_reward = []
        episode = []
        on_place_all = []
        explore = 10000
        epsilon = 1


        for epi in range(NEPISODES):

            # receive initial observation state
            s = env.reset()
            s = np.asarray(s)
            on_place = 0
            total_reward = 0
            done = False
            step = 0
            while (done == False):

                step += 1

                epsilon -= 1.0 / explore

                # select action according to current policy and exploration noise
                a_o = agent.take_action(s)
                explore_noise = OUNoise(a_o, epsilon).noise()
                a = (np.add(a_o, explore_noise))[0]
                a = np.clip(a, -0.5, 0.5)
                s_, r, done, info = env.step(a)
                s_ = np.asarray(s_)
                if s_[3] < 0.45:
                    on_place += 1

                # add to replay buffer
                agent.preceive(s, a, r, s_, done)

                total_reward += r
                s = s_

            episode_reward.append(total_reward)
            episode.append(epi + 1)
            on_place_all.append(on_place)

            if ((epi + 1) % 25 == 0):
                a_model_name = '%d_actor_model.h5' % (epi + 1)
                c_model_name = '%d_critic_model.h5' % (epi + 1)
                rospy.loginfo("Saving Model...")
                agent.save(a_model_name, c_model_name)

            rospy.loginfo(("episode: " + str(epi + 1) + " | " + "Total Reward: " + str(total_reward)))

        ################ Plotting rewards ##############
        plt.figure(2)
        plt.plot(episode, episode_reward, 'b')
        plt.xlabel('Episodes')
        plt.ylabel('Reward')
        plt.savefig("Reward Graph.png")

        ################ Plotting loss ##############
        plt.figure(3)
        loss = agent.plot_loss()
        plt.savefig("Train Graph.png")

        ############### Plotting On_Goal vs Reward ##############
        plt.figure(4)
        plt.plot(episode, on_place_all)
        plt.xlabel('Episodes')
        plt.ylabel('Number of Steps Arrived Goal')
        plt.savefig("Step each Episode.png")

        data1 = {'Reward': episode_reward, 'Number of Arrive': on_place_all}
        data2 = {'Loss': loss}
        data1 = pd.DataFrame(data1, columns=['Reward', 'Number of Arrive'])
        data2 = pd.DataFrame(data2, columns=['Loss'])
        data1.to_csv(r'./Reward.csv', index=None)
        data2.to_csv(r'./Evaluation.csv',index=None)

        env.close()


    def play():
        agent = DDPG(env)
        episode_count = 10

        explore = 10000
        trained_path = 'trained_models_rohit_'
        save_dir = os.path.join(PATH, trained_path)
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        os.chdir(save_dir)
        epsilon = 1
        plot_reward = True
        mean_reward = []
        model_num = []
        std_reward = []
        save_path = 'Random_Track'
        load_dir = os.path.join(PATH, save_path)


        for i in range(25, 500, 25):

            actor_model_name = '%d_actor_model.h5' % i
            critic_model_name = '%d_critic_model.h5' % i
            filepath = os.path.join(load_dir, actor_model_name)
            actor = agent.restore(filepath)
            cumulative_reward = []
            model_num.append(i)
            best_reward = 0
            # x_store = []
            # y_store = []
            # z_store = []
            for epi in range(episode_count):

                # receive initial observation state
                s = env.reset()
                s = np.asarray(s)
                total_reward = 0
                done = False
                step = 0
                x_pos = []
                y_pos = []
                z_pos = []
                while (done == False):
                    step += 1

                    epsilon -= 1.0 / explore

                    # select action according to current policy and exploration noise
                    a = actor.predict(s.reshape(1, s.shape[0]))
                    a = np.clip(a, -0.25, 0.25)
                    # if step > 90 and step < 120:
                    #     a[0] = np.array((0.3, 0, 0))

                    s_, r, done, _ = env.step(a[0])
                    s_ = np.asarray(s_)

                    total_reward += r
                    s = s_

                    # x_pos.append(s[0])
                    # y_pos.append(s[1])
                    # z_pos.append(s[2])

                cumulative_reward.append(total_reward)

                rospy.loginfo("Episodes:" + str(epi) + " done...")

                # 3D plot
                # plt.figure(1)
                # plt.axes(projection='3d')
                # plt.plot(x_pos, y_pos, z_pos)
                # plt.xlim((-3, 3))
                # plt.ylim((-3, 3))
                # plotname = '%d_episodes_state.png' % i
                # plt.savefig(plotname)

            mean_reward.append(np.mean(cumulative_reward))
            mean_data = {"Mean Reward":mean_reward}
            mean_data = pd.DataFrame(mean_data, columns=['Mean Reward'])
            mean_data.to_csv(r'./MeanReward.csv', index=None)
            rospy.loginfo(("episode: " + str(i) + " | " + "Average Reward: " + str(mean_reward)))


        env.close()


    def real_play():
        pass


    if go == 'Train':
        train()
    elif go == 'Play_Sim':
        play()
    elif go == 'Play_Real':
        real_play()
