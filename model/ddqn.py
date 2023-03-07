"""
ddqn.py
"""
__author__ = "py81.kim@gmail.com"
__credits__ = "https://github.com/puyo1999"

import tensorflow as tf
import numpy as np
import keras.backend as K
from keras.models import Sequential, Model, clone_model
from keras.layers.core import Dense, Activation, Lambda
from keras.layers import Input, LSTM, TimeDistributed, RepeatVector, Reshape, Dropout, Bidirectional, Concatenate
#from keras.layers.normalization import BatchNormalization
from keras.optimizers import RMSprop

class DDQN:

    def __init__(self, learning_rate, state_size, action_size, name='DDQNetwork'):
        with tf.variable_scope(name):
            self.state_size = state_size
            self.action_size = action_size
            self.lr = learning_rate
            self.model = self._build_model()
            self.target_model = self._build_model()
            self.update_target_model()

    def _build_model(self):
        # Q-Evaluation Model
        HIDDEN_SIZE = 24

        def weight_loss_wrapper(input_tensor):
            def weight_loss(y_true, y_pred):
                return K.mean(K.square(y_true - y_pred) * input_tensor)
            return weight_loss

        def mse_loss(y_true, y_pred):
            return K.mean(K.square(y_true - y_pred))

        inputs = Input(shape=(self.feature_size,))
        if self.prior:
            weights = Input(shape=(1,))
        fc_1 = Dense(HIDDEN_SIZE, activation='relu')(inputs)
        fc_2 = Dense(len(self.actions))(fc_1)
        rmsp = RMSprop(lr=self.lr)
        if self.prior:
            self.q_eval_model = Model([inputs, weights], fc_2)
            self.q_eval_model.compile(loss=weight_loss_wrapper(weights), optimizer=rmsp)
        else:
            self.q_eval_model = Model(inputs, fc_2)
            self.q_eval_model.compile(loss=mse_loss, optimizer=rmsp)

        self.q_eval_model.summary()
        return self.q_eval_model

    def _copy_model(self):
        self.q_target_model = clone_model(self.q_eval_model)
        self.q_target_model.set_weights(self.q_eval_model.get_weights())

    def update_target_model(self):
        # copy weights from model to target_model
        self.target_model.set_weights(self.model.get_weights())

    def learn(self):
        if self.learning_cnt % self.replace_target_iter == 0:
            self._copy_model()
            if self.verbose: print('Copy model')

        if self.prior:
            idx, w, transition = self.memory.sample(self.batch_size)
        else:
            transition = self.memory.sample(self.batch_size)

        s = transition[:, :self.feature_size]
        s_ = transition[:, -self.feature_size:]
        r = transition[:, self.feature_size + 1]

        if self.prior:
            index = self.q_eval_model.predict([s_, np.ones((self.batch_size, 1))]).argmax(axis=1)
            max_q = self.q_target_model.predict([s_, np.ones((self.batch_size, 1))])[range(self.batch_size), index]
            q_predict = self.q_eval_model.predict([s, np.ones((self.batch_size, 1))])
        else:
            index = self.q_eval_model.predict(s_).argmax(axis=1)
            max_q = self.q_target_model.predict(s_)[range(self.batch_size), index]
            q_predict = self.q_eval_model.predict(s)

        q_target = np.copy(q_predict)
        q_target[range(self.batch_size), transition[:, self.feature_size].astype(np.int32)] = r + self.gamma * max_q

        if self.prior:
            q_pred = self.q_eval_model.predict([s, np.ones((self.batch_size, 1))])
            p = np.sum(np.abs(q_pred - q_target), axis=1)
            self.memory.update(idx, p)
            report = self.q_eval_model.fit([s, w], q_target, verbose=0)
        else:
            report = self.q_eval_model.fit(s, q_target, verbose=0)

        self.epsilon = self.epsilon_min + (self.epsilon - self.epsilon_min) * self.epsilon_decrease

        self.history.append(report.history['loss'])

        if self.verbose and not self.learning_cnt % 100:
            print('training', self.learning_cnt, ': loss', report.history['loss'])

        self.learning_cnt += 1