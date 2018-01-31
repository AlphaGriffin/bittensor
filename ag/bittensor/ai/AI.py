#!/usr/bin/python3
"""
Bittensor.
by: AlphaGriffin
"""

__author__ = "Eric Petersen @Ruckusist"
__copyright__ = "Copyright 2018, The Alpha Griffin Project"
__credits__ = ["Eric Petersen", "Shawn Wilson", "@alphagriffin"]
__license__ = "***"
__version__ = "0.0.1"
__maintainer__ = "Eric Petersen"
__email__ = "ruckusist@alphagriffin.com"
__status__ = "Beta"

# ////////////////// | Imports | \\\\\\\\\\\\\\\#
# generic
import os, sys, time, datetime, collections
from collections import deque

# Tensorflow
import tensorflow as tf
import pandas as pd
import numpy as np
import random


# from stock_env import StockEnv
# from ag.bittensor.ai.stock_env import StockEnv


class Q_Trader(object):
    """
    This handles all the flowing tensors.
    """

    def __init__(self, options):
        self.options = options

        class RuntimeStats(): pass;  # dummy stats holder

        self.stats = RuntimeStats()
        # Runtime Controller Switch, for save_or_load
        self.running = False

        # Set Some Globals...
        self.replay_buffer = deque()
        self.time_step = 0
        self.epsilon = .09
        self.gamma = .09
        self.replay_size = 500
        self.state_dim = 40
        self.action_dim = 3
        self.stats.g_step = 0
        self.batch_size = 32

        # Start the Setup
        self.setup()
        self.session = tf.InteractiveSession(config=tf.ConfigProto(
                allow_soft_placement=True,
                )
        """
        WARNING! ACHTUNG!
        ALWAYS INIT GLOBALS TO ZERO WITH initializer...
        THEN LOAD A PREVIOUSLY SAVED STATE...
        """
        self.session.run(tf.global_variables_initializer())
        # AFTER SETUP RUN SAVE OR LOAD
        self.saver = tf.train.Saver()
        self.save_or_load()

    def setup(self):
        # Gotta have a global step man!
        self.global_step = tf.Variable(0,
            name='global_step', trainable=False)
        # Decay the learning rate exponentially based on the number of steps.
        self.lr = tf.train.exponential_decay(.095,
                                        self.global_step,
                                        .000005,
                                        0.87,
                                        staircase=True,
                                        name="Learn_decay")

        # input layer
        self.state_input = tf.placeholder('float', [None, 40])
        # self.state_input = tf.placeholder('float', [None, 60, 9])


        # 2018 way to init some fucking w and b
        sigma = 1
        weight_initializer = tf.variance_scaling_initializer(mode="fan_avg", distribution="uniform", scale=sigma)
        bias_initializer = tf.zeros_initializer()

        c = []
        for i in range(4):
            with tf.device('/gpu:{}'.format(i)):
                # TODO: make this iteritive
                # TODO: make the input and output dimensions fit the dataset automatic
                W0 = tf.Variable(weight_initializer([40, 1024]))
                B0 = tf.Variable(bias_initializer([1024]))
                W1 = tf.Variable(weight_initializer([1024, 512]))
                B1 = tf.Variable(bias_initializer([512]))
                W2 = tf.Variable(weight_initializer([512, 256]))
                B2 = tf.Variable(bias_initializer([256]))
                W3 = tf.Variable(weight_initializer([256, 128]))
                B3 = tf.Variable(bias_initializer([128]))
                W4 = tf.Variable(weight_initializer([128, 3]))
                B4 = tf.Variable(bias_initializer([3]))

                # Hidden layer
                hidden_1 = tf.nn.relu(tf.add(tf.matmul(self.state_input, W0), B0))
                hidden_2 = tf.nn.relu(tf.add(tf.matmul(hidden_1, W1), B1))
                hidden_3 = tf.nn.relu(tf.add(tf.matmul(hidden_2, W2), B2))
                hidden_4 = tf.nn.relu(tf.add(tf.matmul(hidden_3, W3), B3))

                # Q value layer
                final = tf.matmul(hidden_4, W4) + B4
                c.append(final)
        self.Q_value = tf.add_n(c)

        # Output variables
        self.action_input = tf.placeholder('float', [None, 3])
        self.y_input = tf.placeholder('float', [None])

        # Training Method
        Q_action = tf.reduce_sum(tf.multiply(self.Q_value, self.action_input), reduction_indices=1)
        self.cost = tf.reduce_mean(tf.square(self.y_input - Q_action))
        self.optimizer = tf.train.AdamOptimizer(self.lr).minimize(
            self.cost, global_step=self.global_step, colocate_gradients_with_ops=True)
        pass

    def egreedy_action(self, state):
        Q_value = self.Q_value.eval(feed_dict={
            self.state_input: [state]
        })
        Q_value = Q_value[0]

        if self.epsilon <= 0.1:
            epsilon_rate = 1
        else:
            epsilon_rate = 0.95
        if self.time_step > 200:
            self.epsilon = epsilon_rate * self.epsilon

        if random.random() <= self.epsilon:
            return random.randint(0, 2)
        else:
            return np.argmax(Q_value)

    def action(self, state):
        Q_value = self.Q_value(feed_dict={
            self.state_input: [state]
        })
        Q_value = Q_value[0]
        return np.argmax(Q_value)

    def save_or_load(self):
        folder = os.path.join(os.getcwd(), 'data', 'models', 'Q_Network')
        filename = '/Q_Trader_Network'
        if self.running:  # SAVE
            saver = self.saver
            saver.save(
                self.session,
                folder + filename,
                global_step=self.stats.g_step
            )
            return
        else:  # LOAD
            check_point = tf.train.get_checkpoint_state(folder)
            if check_point and check_point.model_checkpoint_path:
                self.saver.restore(self.session, check_point.model_checkpoint_path)
                print('Loaded Previously Saved State. Get current worked number of steps.')
                return
            else:
                print('No Previously Saved State. New model will be saved to:\n\t{}'.format(
                    folder + filename
                ))
        return


    def getLoss(self):
        """
        this shouldnt be that hard....
        """
        return self.stats.cost


    def train(self, state, action, reward, state_, done):
        self.time_step += 1
        one_hot_action = np.zeros(3)
        one_hot_action[action] = 1
        self.replay_buffer.append((state, one_hot_action, reward, state_, done))
        if len(self.replay_buffer) > self.replay_size:  # this is a rolling window
            self.replay_buffer.popleft()
        if len(self.replay_buffer) > 100:  # after 100 step ,pre  train
            self.training()


    def training(self):
        # get random  sample from replay buffer
        minibatch = random.sample(self.replay_buffer, self.batch_size)
        state_batch = [data[0] for data in minibatch]
        action_batch = [data[1] for data in minibatch]
        reward_batch = [data[2] for data in minibatch]
        next_state_batch = [data[3] for data in minibatch]

        # calcuate Q
        Y_batch = []
        next_Q = self.Q_value.eval(feed_dict={
            self.state_input: next_state_batch
        })

        # Build a batch
        for i in range(self.batch_size):
            done = minibatch[i][4]
        if done:
            Y_batch.append(reward_batch[i])
        else:
            Y_batch.append(reward_batch[i] + self.gamma * np.max(next_Q[i]))

        # Train on that built Batch. Boom.
        """ EXAMPLE
        self.optimizer.run(feed_dict ={
        self.y_input:Y_bach,
        self.action_input:action_bach,
        self.state_input:state_bach
        })
            """
        FEED = {
            self.y_input: Y_batch,
            self.action_input: action_batch,
            self.state_input: state_batch
        }


        # //////| Training Method |\\\\\\\#
        _, self.stats.cost, self.stats.g_step = self.session.run([
                self.optimizer,
                self.cost,
                self.global_step],
                feed_dict=FEED
        )
    pass
