"""
ActorNetwork.py
"""
__author__ = "py81.kim@gmail.com"
__credits__ = "https://github.com/puyo1999"

import tensorflow as tf
import tensorflow

import numpy as np

from keras.layers import Dense
from keras import optimizers, losses
from keras.models import Model

import tensorflow.compat.v1 as tf

#tf.disable_v2_behavior()
from tensorflow.python.framework.ops import enable_eager_execution
enable_eager_execution()

num_action = 3

class ActorNetwork(Model):
    def __init__(self):
        super(ActorNetwork, self).__init__()
        self.layer_a1 = Dense(64, activation='relu')
        self.layer_a2 = Dense(64, activation='relu')
        self.logits = Dense(num_action, activation='softmax')

    def call(self, state):
        layer_a1 = self.layer_a1(state)
        layer_a2 = self.layer_a2(layer_a1)
        logits = self.logits(layer_a2)
        return logits


    def store_transition(self, s, a, r, s_):
        print('a2c @shape of s:{} a:{}'.format(np.shape(s), np.shape(a)))
        print('a2c @StoreTransition - s:{} a:{} r:{} s_:{}'.format(s, a, r, s_))
        transition = np.hstack(
            [(s[0]), (s[1]), (s[2]), (np.r_[a, r]), (s_[0]), (s_[1]), (s_[2])])
        print('StoreTransition - transition:{}'.format(transition))
        # self.memory.store(transition)

        # error = 1010
        error = 1007
        self.memory.add(transition, error=error)