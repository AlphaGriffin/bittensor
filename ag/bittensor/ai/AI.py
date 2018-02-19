#!/usr/bin/python3
"""
Bittensor.
by: AlphaGriffin
"""

__author__ = "Eric Petersen @Ruckusist"
__copyright__ = "Copyright 2018, The Alpha Griffin Project"
__credits__ = ["Eric Petersen", "Shawn Wilson", "@alphagriffin"]
__license__ = "***"
__version__ = "0.0.5"
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
        self.state_dim1 = None
        self.state_dim2 = None
        self.action_dim = 3
        self.stats.g_step = 0
        self.batch_size = 32
        self.num_gpus = self.options.num_gpus

    def reset_que(self):
        self.replay_buffer.clear()
        return True

    def set_state_dim(self, dim1, dim2, dim3, dim4):
        self.state_dim1 = dim1
        self.state_dim2 = dim2
        self.state_dim3 = dim3
        self.state_dim4 = dim4
        pass

    def preRun(self):
        self.setup_proper()
        # GPU SUPPORT IS BACK!!!
        gpu_config = tf.GPUOptions(per_process_gpu_memory_fraction=0.3333)
        try:
            self.session = tf.InteractiveSession(config=tf.ConfigProto(
                    # allow_soft_placement=True,
                    gpu_options=gpu_config
                   ))
            print('Started Session')
        except:
            print('Cant Start a session... Failing.')

        """
        WARNING! ACHTUNG!
        ALWAYS INIT GLOBALS TO ZERO WITH initializer...
        THEN LOAD A PREVIOUSLY SAVED STATE...
        """
        self.session.run(tf.global_variables_initializer())
        print('Initialized globals successfully')
        # AFTER SETUP RUN SAVE OR LOAD
        self.saver = tf.train.Saver()
        self.save_or_load()
        self.running = True
        print('finished model init')

    @property
    def IsRunning(self):
        return self.running

    def setup_OLD(self):
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
        print('setting up gpus')
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
        """BottleNeck
        try:
            with tf.device('/gpu:1'):
                self.Q_value = tf.add_n(c)
        except:
            self.Q_value = tf.add_n(c)
        """
        self.Q_value = tf.add_n(c)
        print('passed gpus setup')
        # Output variables
        self.action_input = tf.placeholder('float', [None, 3])
        self.y_input = tf.placeholder('float', [None])

        # Training Method
        Q_action = tf.reduce_sum(tf.multiply(
            self.Q_value, self.action_input), reduction_indices=1)
        self.cost = tf.reduce_mean(tf.square(self.y_input - Q_action))
        self.optimizer = tf.train.AdamOptimizer(self.lr).minimize(
            self.cost, global_step=self.global_step, colocate_gradients_with_ops=True)
        pass

    def setup_proper(self, num_gpus=4):
        # Gotta have a global step man!
        self.global_step = tf.Variable(0,
            name='global_step', trainable=False)
        # Decay the learning rate exponentially based on the number of steps.
        self.lr = tf.train.exponential_decay(.095,
                                        self.global_step,
                                        .00005,
                                        0.87,
                                        staircase=True,
                                        name="Learn_decay")

        # input layer
        self.state_input1 = tf.placeholder('float', [None, self.state_dim1])
        self.state_input2 = tf.placeholder('float', [None, self.state_dim2])
        self.state_input3 = tf.placeholder('float', [None, self.state_dim3])
        self.state_input4 = tf.placeholder('float', [None, self.state_dim4])
        # self.state_input = tf.placeholder('float', [None, 60, 9])

        sigma = 1.0
        weight_initializer = tf.variance_scaling_initializer(mode="fan_avg", distribution="uniform", scale=sigma)
        bias_initializer = tf.zeros_initializer()

        c = []
        print('starting gpus setup')
        for i, e in enumerate(zip(
                [self.state_dim1,self.state_dim2,self.state_dim3,self.state_dim4],
                [self.state_input1, self.state_input2, self.state_input3, self.state_input4])
                ):
            with tf.device('/gpu:{}'.format(i)):
                W0 = tf.Variable(weight_initializer([e[0], 256]))
                B0 = tf.Variable(bias_initializer([256]))
                W1 = tf.Variable(weight_initializer([256, 128]))
                B1 = tf.Variable(bias_initializer([128]))
                W2 = tf.Variable(weight_initializer([128, 3]))
                B2 = tf.Variable(bias_initializer([3]))

                hidden_1 = tf.nn.relu(tf.add(tf.matmul(e[1], W0), B0))
                hidden_2 = tf.nn.relu(tf.add(tf.matmul(hidden_1, W1), B1))
                # Q value layer
                final = tf.matmul(hidden_2, W2) + B2
                c.append(final)

        self.Q_value = tf.add_n(c)

        # Output variables
        self.action_input = tf.placeholder('float', [None, 3])
        self.y_input = tf.placeholder('float', [None])

        # Training Method
        Q_action = tf.reduce_sum(tf.multiply(self.Q_value, self.action_input), reduction_indices=1)
        self.cost = tf.reduce_mean(tf.square(self.y_input - Q_action))
        self.optimizer = tf.train.AdamOptimizer(self.lr).minimize(
            self.cost,
            global_step=self.global_step,
            colocate_gradients_with_ops=True
            )
        print('finished Model setup')
        pass

    def egreedy_action(self, state):
        Q_value = self.Q_value.eval(feed_dict={
            self.state_input1: [state[0]],
            self.state_input2: [state[1]],
            self.state_input3: [state[2]],
            self.state_input4: [state[3]],
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
        # print('trying to save or load')
        folder = os.path.join(os.getcwd(), 'data', 'models', 'Q_Network')
        filename = '\\Q_Trader_Network'
        if self.running:  # SAVE
            saver = self.saver
            saver.save(
                self.session,
                folder + filename,
                global_step=self.stats.g_step
            )
            print('SAVED!')
            return
        else:  # LOAD
            check_point = tf.train.get_checkpoint_state(folder)
            if check_point and check_point.model_checkpoint_path:
                self.saver.restore(self.session, check_point.model_checkpoint_path)
                # get global steps
                steps = self.session.run(self.global_step)
                print('Loaded Previously Saved State. Model Previous Steps: {}.'.format(steps))
                return
            else:
                print('No Previously Saved State Found.')
                #print('No Previously Saved State. New model will be saved to:\n\t{}'.format(
                #   folder + filename
                #), end='\r')
        print('finished save or load')
        return

    def getLoss(self):
        """
        this shouldnt be that hard....
        """
        return self.stats.cost

    def train(self, state, action, reward, state_):
        self.time_step += 1
        one_hot_action = np.zeros(self.action_dim)
        one_hot_action[action] = 1
        self.replay_buffer.append((state[0], state[1],state[2],state[3], one_hot_action, reward, state_[0], state_[1],state_[2],state_[3]))
        if len(self.replay_buffer) > self.replay_size:  # this is a rolling window
            self.replay_buffer.popleft()
        if len(self.replay_buffer) > 64:  # after 100 step ,pre  train
            self.training()
        else:
            # print('PreBuffering... {}'.format(len(self.replay_buffer)), end='\r')
            pass

    def training(self):
        # get random sample from replay buffer
        minibatch = random.sample(self.replay_buffer, self.batch_size)
        state_batch1 = [data[0] for data in minibatch]
        state_batch2 = [data[1] for data in minibatch]
        state_batch3 = [data[2] for data in minibatch]
        state_batch4 = [data[3] for data in minibatch]
        action_batch = [data[4] for data in minibatch]
        reward_batch = [data[5] for data in minibatch]
        next_state_batch1 = [data[6] for data in minibatch]
        next_state_batch2 = [data[7] for data in minibatch]
        next_state_batch3 = [data[8] for data in minibatch]
        next_state_batch4 = [data[9] for data in minibatch]

        # calcuate Q
        Y_batch = []
        next_Q = self.Q_value.eval(feed_dict={
            self.state_input1: next_state_batch1,
            self.state_input2: next_state_batch2,
            self.state_input3: next_state_batch3,
            self.state_input4: next_state_batch4
        })

        # Build a batch
        for i in range(self.batch_size):
            if i == self.batch_size - 1:
                Y_batch.append(reward_batch[i])
            else:
                Y_batch.append(reward_batch[i] + self.gamma * np.max(next_Q[i]))

        FEED = {
            self.y_input: Y_batch,
            self.action_input: action_batch,
            self.state_input1: state_batch1,
            self.state_input2: state_batch2,
            self.state_input3: state_batch3,
            self.state_input4: state_batch4,
        }


        # //////| Training Method |\\\\\\\#
        _, self.stats.cost, self.stats.g_step = self.session.run([
                self.optimizer,
                self.cost,
                self.global_step],
                feed_dict=FEED
        )

    def get_reward(self, action, state):
        trends = []
        for index, state_ in enumerate(state):
            x = state_.mean()
            d = x.mean()
            trends.append(d)

        trend = np.array(trends, dtype=np.float64).sum()
        # print('trend: {:.3f}'.format(trend))
        right_answer = 2  # do nothing... is usually the right answer
        if trend > 1:
            right_answer = 0  # buy if going up
        elif trend < -1:
            right_answer = 1  # sell if going down

        if action == right_answer:
            reward = 1
        else: reward = -1
        # print('action {}, right {}, reward {}'.format(action, right_answer, reward))
        return reward
