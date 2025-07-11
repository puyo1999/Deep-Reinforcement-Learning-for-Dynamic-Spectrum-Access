"""
CriticNetwork.py
"""
import keras.losses
import tensorflow as tf
tf.compat.v1.enable_eager_execution()

from tensorflow.python.keras import backend as K

from keras.layers import Dense, Input, Add, Multiply
from keras.models import Model
from keras.optimizers import Adam

NUM_USERS = 3		# Total number of users

class CriticNetwork(keras.Model):
	def __init__(self, sess, action_dim, observation_dim, lr):
		super(CriticNetwork, self).__init__()
		self.action_dim, self.observation_dim = action_dim, observation_dim
		self.lr = lr
		K.set_session(sess)

		#self.value_size = NUM_USERS
		self.value_size = 1

		self.state_h1 = Dense(24, activation='relu', kernel_initializer='he_uniform')
		self.state_h2 = Dense(24, activation='relu', kernel_initializer='he_uniform')
		self.state_h3 = Dense(24, activation='relu', kernel_initializer='he_uniform')
		self.state_h4 = Dense(24, activation='relu', kernel_initializer='he_uniform')
		self.output_layer = Dense(self.value_size, activation='linear', kernel_initializer='he_uniform')

		self.state_input, self.output_, self.model = self.create_model()
		#self.model = self.create_model2()

		#self.build(input_shape=[self.action_dim, self.observation_dim])
		self.build(input_shape=[None, observation_dim])
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
		state_input = Input(shape=[self.action_dim, self.observation_dim])
		#state_input = Input(shape=[self.observation_dim, self.action_dim])
		#state_input = Input(shape=(self.state_dim,))
		#state_input = Input(shape=(1,))

		'''
		d1 = Dense(24, activation='relu', kernel_initializer='he_uniform')
		d2 = Dense(24, activation='relu', kernel_initializer='he_uniform')
		d3 = Dense(24, activation='relu', kernel_initializer='he_uniform')
		d4 = Dense(24, activation='relu', kernel_initializer='he_uniform')
		v = Dense(self.value_size, activation='linear', kernel_initializer='he_uniform')
		'''
		state_h1 = Dense(24, activation='relu', kernel_initializer='he_uniform')(state_input)
		state_h2 = Dense(24, activation='relu', kernel_initializer='he_uniform')(state_h1)
		state_h3 = Dense(24, activation='relu', kernel_initializer='he_uniform')(state_h2)
		state_h4 = Dense(24, activation='relu', kernel_initializer='he_uniform')(state_h3)
		output = Dense(self.value_size, activation='linear', kernel_initializer='he_uniform')(state_h4)

		def custom_loss(y_true, y_pred):
			out = K.clip(y_pred, 1e-8, 1 - 1e-8)
			log_lik = y_true * K.log(out)
			print('Critic Loss - {}'.format(K.sum(-log_lik * state_input)))
			return K.sum(-log_lik * state_input)

		model = Model(inputs=[state_input], outputs=[output])
		#model.compile(loss=keras.losses.MeanAbsoluteError(), optimizer='Adam', run_eagerly=False)
		model.compile(loss=keras.losses.MeanAbsoluteError(), optimizer='Adam')
		#return output, d1, d2, d3, d4, v, model
		return state_input, output, model

	def create_model2(self):
		state_input = Input(shape=(self.observation_dim,))
		state_h1 = Dense(24, activation='relu')(state_input)
		state_h2 = Dense(48)(state_h1)
		
		action_input = Input(shape=(self.action_dim,))
		action_h1 = Dense(48)(action_input)
		
		merged = Add()([state_h2, action_h1])
		merged_h1 = Dense(24, activation='relu')(merged)
		output = Dense(1, activation='relu')(merged_h1)
		model = Model(inputs=[state_input, action_input], outputs=output)
		
		adam = Adam(learning_rate=0.001)
		model.compile(loss="mse", optimizer='Adam')
		return model		

