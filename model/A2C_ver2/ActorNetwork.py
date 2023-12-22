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

