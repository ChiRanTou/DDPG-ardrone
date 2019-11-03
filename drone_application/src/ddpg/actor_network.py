import gym
import numpy as np
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input
from keras.layers.merge import Add, Multiply
from keras.optimizers import Adam
import keras.backend as K
import tensorflow as tf
import pdb

from replay_buffer import Replay_Buffer

BATCH_SIZE = 64
TAU = 0.125
LR = 1e-4
BUFFER_SIZE = 20000
HIDDEN_LAYER_1 = 32
HIDDEN_LAYER_2 = 64
EPSILON = 1
EPSILON_DEACY = 0.995
GAMMA = 0.95


class Actor_Network(object):
    def __init__(self, env, sess, batch_size=BATCH_SIZE, tau=TAU, learning_rate=LR):
        self.env = env
        self.sess = sess

        self.obs_dim = self.env.observation_space.shape[0]
        self.act_dim = self.env.action_space.shape[0]

        # hyperparameters
        self.lr = learning_rate
        self.bs = batch_size
        self.eps = EPSILON
        self.eps_decay = EPSILON_DEACY
        self.gamma = GAMMA
        self.tau = tau
        self.buffer_size = BUFFER_SIZE
        self.hidden_layer_1 = HIDDEN_LAYER_1
        self.hidden_layer_2 = HIDDEN_LAYER_2

        # replay buuffer
        self.replay_buffer = Replay_Buffer(self.buffer_size)

        # create model
        self.model, self.weights, self.state = self.create_actor()
        self.target_model, self.target_weights, self.target_state = self.create_actor()

        # gradients
        self.action_gradient = tf.placeholder(tf.float32, [None, self.act_dim])
        self.params_grad = tf.gradients(self.model.output, self.weights,
                                        -self.action_gradient)  # negative for grad ascend
        grads = zip(self.params_grad, self.weights)

        # optimizer & run
        self.optimize = tf.train.AdamOptimizer(self.lr).apply_gradients(grads)
        self.sess.run(tf.initialize_all_variables())

        # self.writer = tf.summary.FileWriter("./logs", graph=tf.get_default_graph())
        # self.merge_op = tf.summary.merge_all()

    def create_actor(self):
        obs_in = Input(shape=[self.obs_dim])  # 5 states
        # pdb.set_trace()

        h1 = Dense(self.hidden_layer_1, activation='relu')(obs_in)
        h2 = Dense(self.hidden_layer_2, activation='relu')(h1)
        h3 = Dense(self.hidden_layer_2, activation='relu')(h2)

        out = Dense(self.act_dim, activation='tanh')(h3)

        model = Model(input=obs_in, output=out)

        # no loss function for actor apparently
        return model, model.trainable_weights, obs_in

    def train(self, states, action_grads):
        self.sess.run(self.optimize,
                      feed_dict={self.state: states, self.action_gradient: action_grads})

    def target_train(self):
        actor_weights = self.model.get_weights()
        actor_target_weights = self.target_model.get_weights()

        # update target network
        for i in range(len(actor_weights)):  # used to be xrange
            actor_target_weights[i] = self.tau * actor_weights[i] + (1 - self.tau) * actor_target_weights[i]

        self.target_model.set_weights(actor_target_weights)
