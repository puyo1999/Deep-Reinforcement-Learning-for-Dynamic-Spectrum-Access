"""
CriticNetwork.py
"""
import keras.losses
import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense, Input
import keras.backend as K
import tensorflow.keras.backend as K

from keras.models import Model
from keras.layers import Add, Multiply
from keras.optimizers.legacy import Adam

import tensorflow.compat.v1 as tf

#tf.disable_v2_behavior()
from tensorflow.python.framework.ops import enable_eager_execution
enable_eager_execution()

NUM_USERS = 3		# Total number of users

class CriticNetwork(keras.Model):
	def __init__(self, sess, action_dim, observation_dim, lr):
		super(CriticNetwork, self).__init__()
		self.action_dim, self.observation_dim = action_dim, observation_dim
		self.lr = lr
		tf.keras.backend.set_session(sess)

		#self.value_size = NUM_USERS
		self.value_size = 1

		self.state_h1 = Dense(24, activation='relu', kernel_initializer='he_uniform')
		self.state_h2 = Dense(24, activation='relu', kernel_initializer='he_uniform')
		self.state_h3 = Dense(24, activation='relu', kernel_initializer='he_uniform')
		self.state_h4 = Dense(24, activation='relu', kernel_initializer='he_uniform')
		self.output_layer = Dense(self.value_size, activation='linear', kernel_initializer='he_uniform')

		#self.output, self.d1, self.d2, self.d3, self.d4, self.v, self.model = self.create_model()
		#self.model = self.create_model2()

		self.build(input_shape=[self.action_dim, observation_dim])
		self.summary()

	def call(self, inputs, training=None, mask=None):
		x = self.state_h1(inputs)
		x = self.state_h2(x)
		x = self.state_h3(x)
		x = self.state_h4(x)
		return self.output_layer(x)

	def get_output(self, input_data):
		x = self.d1(input_data)
		x = self.d2(x)
		x = self.d3(x)
		x = self.d4(x)
		v = self.v(x)
		return v

	def create_model(self):
		state_input = Input(shape=(self.action_dim, self.observation_dim))
		#state_input = Input(shape=(self.state_dim,))
		#state_input = Input(shape=(1,))

		d1 = Dense(24, activation='relu', kernel_initializer='he_uniform')
		d2 = Dense(24, activation='relu', kernel_initializer='he_uniform')
		d3 = Dense(24, activation='relu', kernel_initializer='he_uniform')
		d4 = Dense(24, activation='relu', kernel_initializer='he_uniform')
		v = Dense(self.value_size, activation='linear', kernel_initializer='he_uniform')

		state_h1 = Dense(24, activation='relu', kernel_initializer='he_uniform')(state_input)
		state_h2 = Dense(24, activation='relu', kernel_initializer='he_uniform')(state_h1)
		state_h3 = Dense(24, activation='relu', kernel_initializer='he_uniform')(state_h2)
		state_h4 = Dense(24, activation='relu', kernel_initializer='he_uniform')(state_h3)
		output = Dense(self.value_size, activation='linear', kernel_initializer='he_uniform')(state_h4)
		model = Model(inputs=state_input, outputs=output)
		model.compile(loss=keras.losses.mse, optimizer=Adam(learning_rate=self.lr))
		return output, d1, d2, d3, d4, v, model

	def create_model2(self):
		state_input = Input(shape=(self.observation_dim,))
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

