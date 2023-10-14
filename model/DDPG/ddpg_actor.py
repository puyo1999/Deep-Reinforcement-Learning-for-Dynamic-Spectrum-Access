import keras.losses
import tensorflow as tf
from keras.layers import Dense, Flatten, Input
from keras.initializers import RandomUniform
import keras.backend as K
from keras.models import Model
from keras.optimizers import Adam
import numpy as np

#import tensorflow.compat.v1 as tf
#tf.disable_v2_behavior()

output_init = RandomUniform(-3*10e-3, 3*10e-3)

class Actor(keras.Model):
    def __init__(self, obs_dim, act_dim):
        super(Actor, self).__init__()
        act_dim = np.squeeze(act_dim)

        self.dense1 = Dense(128, activation='relu', kernel_initializer=keras.initializers.RandomUniform(-1/np.sqrt(128),1/np.sqrt(128)))
        self.dense2 = Dense(128, activation='relu', kernel_initializer=keras.initializers.RandomUniform(-1/np.sqrt(128),1/np.sqrt(128)))

        self.output_layer = Dense(act_dim, activation='linear', kernel_initializer=output_init)
        self.build(input_shape=[None, obs_dim])
        self.summary()

    def call(self, inputs, training=None, mask=None):
        x = self.dense1(inputs)
        x = self.dense2(x)
        return self.output_layer(x)

    def predict(self, state):
        """ Action prediction
        """
        print('@@ DDPG predict state:{}'.format(state))
        return super(Actor,self).predict(state)
        #return super(Actor,self).predict(np.expand_dims(state, axis=0))



