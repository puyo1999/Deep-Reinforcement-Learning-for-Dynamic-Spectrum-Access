"""
ActorNetwork.py
"""
__author__ = "py81.kim@gmail.com"
__credits__ = "https://github.com/puyo1999"

import tensorflow as tf
import tensorflow

from keras.layers import Dense
from keras.models import Model

import tensorflow.compat.v1 as tf

#tf.disable_v2_behavior()
from tensorflow.python.framework.ops import enable_eager_execution
enable_eager_execution()

class CriticNetwork(Model):
    def __init__(self):
        super(CriticNetwork, self).__init__()
        self.layer_c1 = Dense(64, activation='relu')
        self.layer_c2 = Dense(64, activation='relu')
        self.value = Dense(1)

    def call(self, state):
        layer_c1 = self.layer_c1(state)
        layer_c2 = self.layer_c2(layer_c1)
        value = self.value(layer_c2)
        return value