import gym
import keras.backend as K
import math
import tensorflow as tf
import numpy as np
import json
from keras.models import load_model
from replay_buffer import Replay_Buffer
from actor_network import Actor_Network
from critic_network import Critic_Network
from ou_noise import OUNoise
from keras.callbacks import TensorBoard

import rospy

GAMMA = 0.98
EXPLORE = 10000
EPSILON = 1
REPLAY_START_SIZE = 5000


class DDPG():
    def __init__(self, env):
        self.action_dim = env.action_space.shape[0]
        self.state_dim = env.observation_space.shape[0]

        self.config = tf.ConfigProto()
        self.config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=self.config)
        K.set_session(self.sess)

        self.actor = Actor_Network(env, self.sess)
        self.critic = Critic_Network(env, self.sess)
        self.replay_buffer = Replay_Buffer()
        self.loss = 0
        self.loss_his = []
        self.accuracy_his = []
        self.gamma = GAMMA
        self.writer = tf.summary.FileWriter("./logs", graph=tf.get_default_graph())


    def train(self):

        # sample from replay buffer
        batch = self.replay_buffer.sample_batch()
        states = np.asarray([e[0] for e in batch])
        actions = np.asarray([e[1] for e in batch])
        rewards = np.asarray([e[2] for e in batch])
        new_states = np.asarray([e[3] for e in batch])
        dones = np.asarray([e[4] for e in batch])
        y_t = np.asarray([e[1] for e in batch])

        target_q_values = self.critic.target_model.predict([new_states, self.actor.target_model.predict(new_states)])

        for k in range(len(batch)):
            if dones[k]:
                y_t[k] = rewards[k]
            else:
                y_t[k] = rewards[k] + self.gamma * target_q_values[k]

        self.loss += self.critic.model.train_on_batch([states, actions], y_t)
        self.loss_his.append(self.loss)
        # history = self.critic.model.fit([states, actions], y_t)
        # self.loss_his.append(history.history['loss'])
        #self.accuracy_his.append(history.history['acc'])

        a_for_grad = self.actor.model.predict(states)
        grads = self.critic.gradients(states, a_for_grad)
        self.actor.train(states, grads)
        self.actor.target_train()
        self.critic.target_train()

    def take_action(self, s):

        # a = self.actor.model.predict(s.reshape(1, s.shape[0]))
        a = self.actor.model.predict(s.reshape(1, s.shape[0]))

        return a

    def preceive(self, s, a, r, s_, done):

        self.replay_buffer.add(s, a, r, s_, done)

        if self.replay_buffer.count() > REPLAY_START_SIZE:
            self.train()
        # self.train()

    def save(self, a_model_name, c_model_name):

        self.actor.model.save(a_model_name)
        self.critic.model.save(c_model_name)

        self.actor.model.save_weights("actormodel.h5", overwrite=True)
        with open("actormodel.json", "w") as outfile:
            json.dump(self.actor.model.to_json(), outfile)

        self.critic.model.save_weights("criticmodel.h5", overwrite=True)
        with open("criticmodel.json", "w") as outfile:
            json.dump(self.critic.model.to_json(), outfile)

    def restore(self, dir):

        self.actor = load_model(dir)

        return self.actor

    def plot_loss(self):
        import matplotlib.pyplot as plt
        plt.plot(self.loss_his, 'b')
        # plt.plot(self.loss_valid, 'r')
        plt.ylabel('Loss')
        plt.xlabel('training steps')
        return self.loss_his
