"""
ActorNetwork.py
"""
__author__ = "py81.kim@gmail.com"
__credits__ = "https://github.com/puyo1999"

import tensorflow as tf
from keras.layers import Dense, Flatten, Input
import keras.backend as K
from keras import Model
from keras.optimizers import Adam
import numpy as np

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

class ActorNetwork:
	def __init__(self, sess, state_dim, action_dim):
		self.learning_rate = 0.0001
		self.state_dim, self.action_dim = state_dim, action_dim
		
		K.set_session(sess)
		self.sess = sess
		self.state_input, self.output, self.model = self.create_model()

		self.advantages = tf.placeholder(tf.float32, shape=[None, action_dim])
		model_weights = self.model.trainable_weights

		log_prob = tf.math.log(self.output + 10e-10)
		neg_log_prob = tf.multiply(log_prob, -1)

		actor_gradients = tf.gradients(neg_log_prob, model_weights, self.advantages)
		grads = zip(actor_gradients, model_weights)
		self.optimize = tf.train.AdamOptimizer(0.0001).apply_gradients(grads)


	def create_model(self):
		state_input = Input(shape=(self.state_dim,))
		state_h1 = Dense(24, activation='relu', kernel_initializer='he_uniform')(state_input)
		state_h2 = Dense(24, activation='relu', kernel_initializer='he_uniform')(state_h1)
		output = Dense(self.action_dim, activation='softmax', kernel_initializer='he_uniform')(state_h2)
		model = Model(inputs=state_input, outputs=output)
		adam = Adam(learning_rate=0.001)
		model.compile(loss='categorical_crossentropy', optimizer='adam')
		return state_input, output, model

	def train(self, X, y):

		self.sess.run(self.optimize, feed_dict={self.state_input:X, self.advantages:y})