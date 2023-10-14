import tensorflow as tf
from keras.layers import Dense, Flatten, Input
from keras.initializers import RandomUniform
import keras.backend as K
from keras.models import Model
from keras.optimizers import Adam
import keras
import numpy as np

output_init = RandomUniform(-3*10e-3, 3*10e-3)

class Critic(keras.Model):
    def __init__(self, obs_dim, act_dim):
        super(Critic, self).__init__()
        act_dim = np.squeeze(act_dim)

        self.o_dense1 = Dense(128, activation='relu', kernel_initializer=keras.initializers.RandomUniform(-1/np.sqrt(128),1/np.sqrt(128)))
        self.o_dense2 = Dense(128, activation='relu', kernel_initializer=keras.initializers.RandomUniform(-1/np.sqrt(128),1/np.sqrt(128)))

        self.output_layer = Dense(1, activation='linear', kernel_initializer=output_init)
        self.build(input_shape=[(None, obs_dim), (None, act_dim)])
        self.summary()

    def call(self, inputs, training=None, mask=None):
        obs, action = inputs
        z = tf.concat([obs, action], axis=1)

        x = self.o_dense1(z)
        x = self.o_dense2(z)
        return self.output_layer(x)
    

