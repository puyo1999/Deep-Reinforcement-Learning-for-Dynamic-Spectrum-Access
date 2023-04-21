"""
CriticNetwork.py
"""

import tensorflow as tf
from keras.layers import Dense, Input
import keras.backend as K
from keras import Model
from keras.layers import Add, Multiply
from keras.optimizers import Adam

NUM_USERS = 3		# Total number of users

class CriticNetwork:
	def __init__(self, sess, state_dim, action_dim):
		self.state_dim = state_dim
		self.action_dim = action_dim
		K.set_session(sess)
		self.value_size = NUM_USERS
		self.model = self.create_model()
		#self.model = self.create_model2()
	
	def create_model(self):
		#state_input = Input(shape=self.observation_dim)
		#state_input = Input(shape=(self.state_dim,))
		state_input = Input(shape=(1,))
		state_h1 = Dense(24, activation='relu', kernel_initializer='he_uniform')(state_input)
		state_h2 = Dense(24, activation='relu', kernel_initializer='he_uniform')(state_h1)
		state_h3 = Dense(24, activation='relu', kernel_initializer='he_uniform')(state_h2)
		state_h4 = Dense(24, activation='relu', kernel_initializer='he_uniform')(state_h3)
		output = Dense(self.value_size, activation='linear', kernel_initializer='he_uniform')(state_h4)
		model = Model(inputs=state_input, outputs=output)
		model.compile(loss="mse", optimizer=Adam(learning_rate=0.0005))
		return model

	def create_model2(self):
		state_input = Input(shape=(self.state_dim,))
		state_h1 = Dense(24, activation='relu')(state_input)
		state_h2 = Dense(48)(state_h1)
		
		action_input = Input(shape=(self.action_dim,))
		action_h1    = Dense(48)(action_input)
		
		merged = Add()([state_h2, action_h1])
		merged_h1 = Dense(24, activation='relu')(merged)
		output = Dense(1, activation='relu')(merged_h1)
		model  = Model(inputs=[state_input, action_input], outputs=output)
		
		adam  = Adam(learning_rate=0.001)
		model.compile(loss="mse", optimizer=adam)
		return model		


