 # Deep-Q Learning Code

import numpy as np
import tensorflow as tf
import rospy
from collections import deque
import random

np.random.seed(1)
tf.set_random_seed(1)


# Deep Q Network off-Policy
class DQN:
    # Initialization
    def __init__(
            self,
            n_actions,
            n_features,
            learning_rate,  # alpha
            gamma,  # reward decay
            e_greedy,
            replace_target_iter,
            memory_size,
            batch_size,
    ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.LR = learning_rate
        self.gamma = gamma
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_min = 0.01
        self.epsilon_max = 1
        self.epsilon = e_greedy
        self.memory_count = 0

        # Total Learning Step
        self.learnStep_counter = 0

        # Initialize zero memory (s,a,r,s_)
        # Create a full zero matrix with row of memory size and column of information
        self.memory = deque()

        # Build the nets
        self._build_net()

        # Replace target net parameter
        target_params = tf.get_collection('target_net_params')
        evaluate_params = tf.get_collection('eval_net_params')
        # Update the target net value
        self.replace_target_op = [tf.assign(t, e) for t, e in zip(target_params, evaluate_params)]

        self.sess = tf.Session()

        self.sess.run(tf.global_variables_initializer())
        self.cost_his = []  # Record all the cost and plot later

    # Set up the deep network
    def _build_net(self):
        # ---------------- Evaluate Net-------------------------
        self.s = tf.placeholder(tf.float32, [None, self.n_features], name='s')  # Get current state observation
        self.q_target = tf.placeholder(tf.float32, [None, self.n_actions], name='Q_target')  # Get Value of Q target

        # Define variable for the neural network
        with tf.variable_scope('eval_net'):
            # Set default variable for the neural network
            c_names, n_l1, weighted_init, bias_init = ['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES], 64, \
                                                      tf.random_normal_initializer(0., 0.3), tf.constant_initializer(
                0.1)

            # Network layer 1
            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [self.n_features, n_l1], initializer=weighted_init, collections=c_names)
                b1 = tf.get_variable('b1', [n_l1], initializer=bias_init, collections=c_names)
                l1 = tf.nn.relu(tf.matmul(self.s, w1) + b1)

            # Network layer 2
            with tf.variable_scope('l2'):
                w2 = tf.get_variable('w2', [n_l1, self.n_actions], initializer=weighted_init, collections=c_names)
                b2 = tf.get_variable('b2', [self.n_actions], initializer=bias_init, collections=c_names)
                self.q_eval = tf.matmul(l1, w2) + b2

            # Calculate the loss
            with tf.variable_scope('loss'):
                self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval))

            with tf.variable_scope('train'):
                self.train_op = tf.train.RMSPropOptimizer(self.LR).minimize(self.loss)

        # -----------------Target Net-----------------------------
        self.s_ = tf.placeholder(tf.float32, [None, self.n_features], name='s_')  # Get next state observation

        # Define layer for target net
        with tf.variable_scope('target_net'):
            # Store the information in target net parameter in the target net
            c_names = ['target_net_params', tf.GraphKeys.GLOBAL_VARIABLES]

            # Network layer 1
            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [self.n_features, n_l1], initializer=weighted_init, collections=c_names)
                b1 = tf.get_variable('b1', [n_l1], initializer=bias_init, collections=c_names)
                l1 = tf.nn.relu(tf.matmul(self.s, w1) + b1)

            # Network layer 2
            with tf.variable_scope('l2'):
                w2 = tf.get_variable('w2', [n_l1, self.n_actions], initializer=weighted_init, collections=c_names)
                b2 = tf.get_variable('b2', [self.n_actions], initializer=bias_init, collections=c_names)
                self.q_next = tf.matmul(l1, w2) + b2

    # store the transition on the memory
    def store_transition(self, s, a, r, s_, done):

        transition = (s, a, r, s_, done)
        if self.memory_count < self.memory_size:
            self.memory.append(transition)
            self.memory_count += 1
        else:
            self.memory.popleft()
            self.memory.append(transition)

    # choose an action
    def choose_action(self, observation):
        observation = np.array(observation)
        #observation = observation[np.newaxis, :]
        # Find the maximum value for action if lower than epsilon
        # else random pick a value if bigger
        if np.random.uniform() < self.epsilon:
            return np.argmax(self.sess.run(self.q_eval, feed_dict={self.s: observation.reshape(-1, self.n_features)}))

        return np.random.randint(0, self.n_actions)

    # process learning
    def learn(self):
        # check if need to replace the target parameters
        if self.learnStep_counter % self.replace_target_iter:
            self.sess.run(self.replace_target_op)

        # batch sample memory from the storage memory
        # sample batch memory from all memory
        # sample batch memory from all memory
        if self.memory_count > self.batch_size:
            mini_batch = random.sample(self.memory, self.batch_size)
        else:
            mini_batch = random.sample(self.memory, self.memory_count)

        states = np.asarray([e[0] for e in mini_batch])
        actions = np.asarray([e[1] for e in mini_batch])
        rewards = np.asarray([e[2] for e in mini_batch])
        next_states = np.asarray([e[3] for e in mini_batch])
        dones = np.asarray([e[4] for e in mini_batch])

        # Get the q_next and q_eval from the momory
        q_next, q_eval = self.sess.run([self.q_next, self.q_eval],
                                       feed_dict={self.s_: next_states,
                                                  self.s: states})

        # Give q target value as same as q eval
        q_target = q_eval.copy()

        for k in range(len(mini_batch)):
            if dones[k]:
                q_target[k][actions[k]] = rewards[k]
            else:
                q_target[k][actions[k]] = rewards[k] + self.gamma * np.max(q_next[k])

        loss = self.sess.run(self.loss, feed_dict={self.s: states, self.q_target: q_target})
        self.cost_his.append(loss)
        self.sess.run(self.train_op, feed_dict={self.s: states, self.q_target: q_target})
        self.epsilon = self.epsilon_min + (self.epsilon_max - self.epsilon_min) * (np.exp(-0.01))
        # self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learnStep_counter += 1

    def preceive(self, s, a, r, s_, done):
        self.store_transition(s, a, r, s_, done)
        if self.memory_count > self.replace_target_iter:
            self.learn()

    def plot_cost(self):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.cost_his)), self.cost_his)
        plt.ylabel('Cost')
        plt.xlabel('training steps')

    def save(self):
        saver = tf.compat.v1.train.Saver()
        saver.save(self.sess, './params', write_meta_graph=False)

    def restore(self):
        saver = tf.compat.v1.train.Saver()
        saver.restore(self.sess, './params')
