"""
ActorNetwork.py
"""
__author__ = "py81.kim@gmail.com"
__credits__ = "https://github.com/puyo1999"

import keras.losses
import tensorflow as tf
from keras.layers import Dense, Flatten, Input
import keras.backend as K
from keras.models import Model
from keras.optimizers import Adam
import numpy as np

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


class ActorNetwork:
	def __init__(self, sess, action_dim, observation_dim, lr, memory):
		self.lr = lr
		self.action_dim, self.observation_dim = action_dim, observation_dim
		self.memory = memory

		self.state_input = tf.placeholder(tf.float32, shape=[None, observation_dim], name='inputs_')

		# setting the our created session as default session
		K.set_session(sess)
		self.sess = sess
		self.state_input, self.output, self.model, self.policy = self.create_model()


		# Placeholder for advantage values
		self.advantages = tf.placeholder(tf.float32, shape=[None, action_dim])
		model_weights = self.model.trainable_weights

		# Adding small number inside log to avoid log(0) = -infinity
		log_prob = tf.math.log(self.output + 10e-10)

		neg_log_prob = tf.multiply(log_prob, -1)

		actor_gradients = tf.gradients(neg_log_prob, model_weights, self.advantages)
		grads = zip(actor_gradients, model_weights)
		#self.optimize = tf.train.AdamOptimizer(self.lr).apply_gradients(grads)
		self.optimize = Adam(self.lr).apply_gradients(grads)


	def create_model(self):
		state_input = Input(shape=(self.action_dim, self.observation_dim))
		delta = Input(shape=[self.action_dim])
		#delta = Input(shape=[1])
		state_h1 = Dense(24, activation='relu', kernel_initializer='he_uniform')(state_input)
		state_h2 = Dense(24, activation='relu', kernel_initializer='he_uniform')(state_h1)
		output = Dense(self.action_dim, activation='softmax', kernel_initializer='he_uniform')(state_h2)

		def custom_loss(y_true, y_pred):
			out = K.clip(y_pred, 1e-8, 1 - 1e-8)
			log_lik = y_true * K.log(out)
			return K.sum(-log_lik * delta)

		policy = Model(inputs=[state_input], outputs=[output])
		model = Model(inputs=[state_input, delta], outputs=[output])
		adam = Adam(self.lr)
		#model.compile(loss='categorical_crossentropy', optimizer='adam')
		#model.compile(loss=keras.losses.CategoricalCrossentropy(), optimizer=adam)
		model.compile(loss=custom_loss, optimizer=adam)
		return state_input, output, model, policy

	def store_transition(self, s, a, r, s_):
		print('shape of s:{} a:{}'.format(np.shape(s), np.shape(a)))
		print('StoreTransition - s:{} a:{} r:{} s_:{}'.format(s, a, r, s_))
		transition = np.hstack(
			[list(s[0]), list(s[1]), list(s[2]), list(np.r_[a, r]), list(s_[0]), list(s_[1]), list(s_[2])])
		print('StoreTransition - transition:{}'.format(transition))
		self.memory.store(transition)
		#self.step_cnt += 1

	def train(self, X, y):

		self.sess.run(self.optimize, feed_dict={self.state_input:X, self.advantages:y})