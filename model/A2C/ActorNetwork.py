"""
ActorNetwork.py
"""
__author__ = "py81.kim@gmail.com"
__credits__ = "https://github.com/puyo1999"

import keras.losses
import tensorflow as tf
import tensorflow

from tensorflow import keras
from keras.layers import Dense, Flatten, Input
#import keras.backend as K
import tensorflow.keras.backend as K

from keras.models import Model
from keras.optimizers.legacy import Adam

import torch
import torch.nn.functional as F
from torch.distributions import Normal

import numpy as np

import tensorflow_probability as tfp
import tensorflow.compat.v1 as tf

class ActorNetwork(keras.Model):
	# action : 3, observation : 6
	def __init__(self, sess, action_dim, observation_dim, lr, memory):
		super(ActorNetwork, self).__init__()
		self.lr = lr
		self.action_dim, self.observation_dim = action_dim, observation_dim
		self.memory = memory
		self.log_std_min = -2
		self.log_std_max = 2

		self.relu = torch.nn.ReLU()

		self.action_dim = np.squeeze(action_dim)

		#self.state_input = tf.placeholder(tf.float32, shape=[None, observation_dim], name='inputs_')

		# setting the our created session as default session
		tf.keras.backend.set_session(sess)

		self.sess = sess
		#self.state_input, self.a, self.output, self.model, self.policy = self.create_model()
		self.state_input, self.output_, self.model = self.create_model()

		self.state_h1 = Dense(24, activation='relu', kernel_initializer='he_uniform')
		self.state_h2 = Dense(24, activation='relu', kernel_initializer='he_uniform')

		self.mu = Dense(self.action_dim, activation=None, kernel_initializer='he_uniform')
		self.log_std_linear = Dense(self.action_dim, activation=None)

		self.output_layer = Dense(self.action_dim, activation='softmax', kernel_initializer='he_uniform')


		# Placeholder for advantage values
		#self.advantages = tf.placeholder(tf.float32, shape=[None, action_dim])

		self.build(input_shape=[None, observation_dim])
		self.summary()

		'''
		model_weights = self.model.trainable_weights

		# Adding small number inside log to avoid log(0) = -infinity
		self.log_prob = tf.math.log(self.output_ + 10e-10)

		neg_log_prob = tf.multiply(self.log_prob, -1)

		actor_gradients = tf.gradients(neg_log_prob, model_weights, self.advantages)
		grads = zip(actor_gradients, model_weights)
		#self.optimize = tf.train.AdamOptimizer(self.lr).apply_gradients(grads)
		self.optimize = Adam(self.lr).apply_gradients(grads)
		'''

	def call(self, inputs, training=None, mask=None):
		x = self.state_h1(inputs)
		x = self.state_h2(x)

		return self.output_layer(x)

	'''
	def predict(self, state):
		""" Action prediction
        """
		print('@@ A2C predict state:{}'.format(state))
		return super(ActorNetwork, self).predict(state)
	'''

	# return super(Actor,self).predict(np.expand_dims(state, axis=0))
	def forward(self, state):
		x = self.state_h1(state)
		x = self.state_h2(x)

		mu = self.mu(x)
		log_std = self.log_std_linear(x)
		print(f'@ forward - mu:{mu}\nlog_std:{log_std}')
		log_std = np.clip(log_std, a_min=self.log_std_min, a_max=self.log_std_max)
		#log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
		return mu, log_std

	def get_action(self, state):
		"""
        returns the action based on a squashed gaussian policy. That means the samples are obtained according to:
        a(s,e)= tanh(mu(s)+sigma(s)+e)
        """
		mu, log_std = self.forward(state)
		std = log_std.exp()

		dist = Normal(0, 1)
		e = dist.sample()
		action = np.tanh(mu + e * std).cpu()
		return action[0]

	def get_output(self, input_data):
		x = self.d1(input_data)
		x = self.d2(x)
		a = self.a(x)
		return a

	def create_model(self):
		# state_input [3, 6]
		state_input = Input(shape=(self.action_dim, self.observation_dim))
		delta = Input(shape=[self.action_dim])
		#delta = Input(shape=[1])
		#d1 = Dense(24, activation='relu', kernel_initializer='he_uniform')
		d2 = Dense(24, activation='relu', kernel_initializer='he_uniform')
		a = Dense(self.action_dim, activation='softmax', kernel_initializer='he_uniform')

		state_h1 = Dense(24, activation='relu', kernel_initializer='he_uniform')(state_input)
		state_h2 = Dense(24, activation='relu', kernel_initializer='he_uniform')(state_h1)

		output_layer = Dense(self.action_dim, activation='softmax', kernel_initializer='he_uniform')(state_h2)

		def custom_loss(y_true, y_pred):
			out = K.clip(y_pred, 1e-8, 1 - 1e-8)
			log_lik = y_true * K.log(out)
			print('Actor Loss - {}'.format(K.sum(-log_lik * delta)))
			return K.sum(-log_lik * delta)

		policy = Model(inputs=[state_input], outputs=[output_layer])
		model = Model(inputs=[state_input, delta], outputs=[output_layer])
		adam = Adam(self.lr)

		#model.compile(loss='categorical_crossentropy', optimizer='adam')
		#model.compile(loss=keras.losses.CategoricalCrossentropy(), optimizer=adam)
		model.compile(loss=custom_loss, optimizer=adam, run_eagerly=True)

		#return state_input, a, output_layer, model, policy
		return state_input, output_layer, model

	def store_transition(self, s, a, r, s_):
		print('a2c @shape of s:{} a:{}'.format(np.shape(s), np.shape(a)))
		print('a2c @StoreTransition - s:{} a:{} r:{} s_:{}'.format(s, a, r, s_))
		transition = np.hstack(
			[(s[0]), (s[1]), (s[2]), (np.r_[a, r]), (s_[0]), (s_[1]), (s_[2])])
		print('StoreTransition - transition:{}'.format(transition))
		#self.memory.store(transition)

		error = 1010

		self.memory.add(transition, error=error)

		#self.step_cnt += 1

	@tf.function
	def actor_loss(self, probs, actions, td):
		print('@ actor_loss - probs:\n{}'.format(probs))
		print('@ actor_loss - actions:\n{}'.format(actions))
		print('@ actor_loss - td:\n{}'.format(td))

		dist = tfp.distributions.Categorical(probs=probs, dtype=tf.float32)
		log_prob = dist.log_prob(actions)
		loss = -log_prob * td
		return loss

		'''
		probability = []
		log_probability = []
		probs_unpacked = tf.unstack(probs)
		actions_unpacked = tf.unstack(actions)
		print('@ probs_unpacked:\n{}'.format(probs_unpacked))
		for pb, a in zip(probs_unpacked, actions_unpacked):
			dist = tfp.distributions.Categorical(probs=pb, dtype=tf.float32)
			log_prob = dist.log_prob(a)
			prob = dist.prob(a)
			probability.append(prob)
			log_probability.append(log_prob)

		p_loss = []
		e_loss = []
		#td = td.numpy()

		print('@ td:\n{}'.format(td))
		for pb, t, lpb in zip(probability, td, log_probability):
			t = tf.constant(t)
			policy_loss = tf.math.multiply(lpb, t)
			entropy_loss = tf.math.negative(tf.math.multiply(pb, lpb))
			p_loss.append(policy_loss)
			e_loss.append(entropy_loss)
		p_loss = tf.stack(p_loss)
		e_loss = tf.stack(e_loss)
		p_loss = tf.reduce_mean(p_loss)
		e_loss = tf.reduce_mean(e_loss)

		loss = -p_loss - 0.0001 * e_loss
		return loss
		'''

	def learn(self, batch, batch_size, feature_size):
		if self.prior:
			idx, w, transition = self.memory.sample(batch_size)
		else:
			transition = self.memory.sample(batch_size)
		print('@ actor learn - transition:\n{}\n'.format(transition))
		s = transition[:,:feature_size]

		if self.prior:
			#q_pred = self.q_eval_model.predict([s, np.ones((self.batch_size, 1))])
			#p = np.sum(np.abs(q_pred - q_target), axis=1)
			#errors = (q_preds - q_targets).abs().cpu().numpy()
			#print('##### Before memory.update() p:\n{}\n'.format(p))
			self.memory.update(idx, 0.001)

	'''
	def train(self, X, y):

		self.sess.run(self.optimize, feed_dict={self.state_input:X, self.advantages:y})
	'''