import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from keras.layers import LayerNormalization
import keras.layers as kl
import keras.src.layers.rnn as kslr

import numpy as np

class DQNetwork:
    def __init__(self, learning_rate, state_size,action_size,hidden_size,step_size, name):
        with tf.variable_scope(name):
            self.inputs_ = tf.placeholder(tf.float32, [None, step_size, state_size], name='inputs_')
            self.actions_ = tf.placeholder(tf.int32, [None], name='actions')
            one_hot_actions = tf.one_hot(self.actions_, action_size)

            self.targetQs_ = tf.placeholder(tf.float32, [None], name='target')
            ##########################################

            #self.rnn = kl.rnn.legacy_cells.BasicRNNCell(hidden_size)
            self.rnn = kl.SimpleRNNCell(hidden_size)
            self.rnn_out, self.state = tf.nn.dynamic_rnn(self.rnn, self.inputs_, dtype=tf.float32)

            self.reduced_out = self.rnn_out[:, -1, :]
            self.reduced_out = tf.reshape(self.reduced_out, shape=[-1, hidden_size])
            #########################################

            self.w1 = tf.Variable(tf.random_uniform([hidden_size, hidden_size]))
            self.b1 = tf.Variable(tf.constant(0.1, shape=[hidden_size]))
            self.h1 = tf.matmul(self.reduced_out, self.w1) + self.b1
            self.h1 = tf.nn.relu(self.h1)
            #self.h1 = tf.contrib.layers.layer_norm(self.h1)

            """input_layer_norm = kl.LayerNormalization(axis=-1, epsilon=1e-12, dtype=tf.float32)
            self.h1 = input_layer_norm(self.h1)"""

            self.w2 = tf.Variable(tf.random_uniform([hidden_size, hidden_size // 2]))
            self.b2 = tf.Variable(tf.constant(0.1, shape=[hidden_size // 2]))
            self.h2 = tf.matmul(self.h1, self.w2) + self.b2
            self.h2 = tf.nn.relu(self.h2)
            #self.h2 = tf.contrib.layers.layer_norm(self.h2)

            """input_layer_norm = kl.LayerNormalization(axis=-1, epsilon=1e-12, dtype=tf.float32)
            self.h2 = input_layer_norm(self.h2)"""

            self.w3 = tf.Variable(tf.random_uniform([hidden_size // 2, hidden_size // 4]))
            self.b3 = tf.Variable(tf.constant(0.1, shape=[hidden_size // 4]))
            self.h3 = tf.matmul(self.h2, self.w3) + self.b3
            self.h3 = tf.nn.relu(self.h3)
            #self.h3 = tf.contrib.layers.layer_norm(self.h3)
            input_layer_norm = kl.LayerNormalization(axis=-1, epsilon=1e-12, dtype=tf.float32)
            self.h3 = input_layer_norm(self.h3)

            self.w4 = tf.Variable(tf.random_uniform([hidden_size // 4, action_size]))
            self.b4 = tf.Variable(tf.constant(0.1, shape=[action_size]))
            self.h4 = tf.matmul(self.h3, self.w4) + self.b4
            self.h4 = tf.nn.relu(self.h4)
            #self.h4 = tf.contrib.layers.layer_norm(self.h4)
            input_layer_norm = kl.LayerNormalization(axis=-1, epsilon=1e-12, dtype=tf.float32)
            self.h4 = input_layer_norm(self.h4)

            #self.fc1 = tensorflow.contrib.layers.fully_connected(self.h4, hidden_size)
            #self.fc2 = tensorflow.contrib.layers.fully_connected(self.fc1, hidden_size)
            self.fc1 = tf.layers.dense(self.h4, hidden_size, activation=tf.nn.relu)
            self.fc2 = tf.layers.dense(self.fc1, hidden_size, activation=tf.nn.relu)

            #self.output = tf.contrib.layers.fully_connected(self.fc2, action_size,activation_fn=None)
            self.output = tf.layers.dense(self.fc2, action_size, activation=tf.nn.relu)


            self.Q = tf.reduce_sum(tf.multiply(self.output, one_hot_actions), axis=1)

            self.loss = tf.reduce_mean(tf.square(self.targetQs_ - self.Q))
            self.opt = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)
