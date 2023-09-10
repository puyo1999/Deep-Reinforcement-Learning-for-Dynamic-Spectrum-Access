"""
ddqn.py
"""
__author__ = "py81.kim@gmail.com"
__credits__ = "https://github.com/puyo1999"

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import keras.backend as K
from keras.models import Sequential, Model, clone_model
from keras.layers.core import Dense, Activation, Lambda
from keras.layers import Input, LSTM, TimeDistributed, RepeatVector, Reshape, Dropout, Bidirectional, Concatenate
#from keras.layers.normalization import BatchNormalization
from keras.optimizers import RMSprop
import random

class DDQN:

    def __init__(self, feature_size, learning_rate, state_size, actions, action_size, step_size, prior, memory, gamma, epsilon=.5, epsilon_min=.1, epsilon_decrease=.9, name='DDQNetwork'):
        with tf.variable_scope(name):
            self.epsilon = epsilon
            self.epsilon_min = epsilon_min
            self.epsilon_decrease = epsilon_decrease
            self.prior = prior
            self.memory = memory
            self.gamma = gamma
            self.state_size = state_size
            self.actions = actions
            self.action_size = action_size
            self.step_size = step_size
            self.lr = learning_rate

            self.feature_size = feature_size
            self.replace_target_iter = 500
            self.batch_size = 32
            self.step_cnt = 0
            self.verbose = True
            self.learning_cnt = 0
            self.history = []

            self.q_eval_model = None
            self.q_target_model = None

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

        #inputs = tf.placeholder(tf.float32, [None,self.step_size, self.state_size], name='inputs')
        #actions = tf.placeholder(tf.int32, [None], name='actions')
        #one_hot_actions = tf.one_hot(self.actions, self.action_size)

        inputs = Input(shape=(self.feature_size,))
        #inputs = Input(shape=(1,))
        if self.prior:
            weights = Input(shape=(1,))
        fc_1 = Dense(HIDDEN_SIZE, activation='relu')(inputs)
        fc_2 = Dense(len(self.actions))(fc_1)
        rmsp = RMSprop(learning_rate=self.lr)
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

    def store_transition(self, s, a, r, s_):
        #print('shape of s:{} a:{}'.format(np.shape(s), np.shape(a)))
        #print('StoreTransition - s:{} a:{} r:{} s_:{}'.format(s,a,r,s_))
        transition = np.hstack([list(s[0]),list(s[1]), list(s[2]), list(np.r_[a, r]), list(s_[0]), list(s_[1]),list(s_[2])])
        #print('StoreTransition - transition:{}'.format(transition))
        self.memory.store(transition)
        self.step_cnt += 1

    def update_target_model(self):
        # copy weights from model to target_model
        self.target_model.set_weights(self.model.get_weights())

    def actor(self, observation):
        """
        if np.random.uniform(0, 1) < self.epsilon:
            action = np.random.choice(self.actions)
        else:
        """
        observation = np.array(observation)
        if self.prior:
            print("@ actor - prior case")
            print("!!! obs : ", observation)
            observation = tf.one_hot(observation, self.feature_size)
            print("!!! after one_hot obs : ", observation)
            q_value = self.q_eval_model.predict([observation, np.ones((1, 1))],steps=1)
            print("!!! q_value : ", q_value)
        else:
            q_value = self.q_eval_model.predict(observation)
        action = self.actions[q_value.argmax()]
        print('@ actor - action:{}'.format(action))
        return action

    def learn(self, memory, replay_memory, batch_size):
        if self.learning_cnt % self.replace_target_iter == 0:
            self._copy_model()
            if self.verbose:
                print('Copy model')

        #minibatch = random.sample(replay_memory, batch_size)
        #for sample in enumerate(minibatch):
            #s, a, r, s_ = sample

        '''
        transition = random.sample(replay_memory, batch_size)
        print("transition:")
        print(transition)
        s, a, r, s_ = map(lambda x: np.vstack(x).astype('float32'), np.transpose(transition))
        '''

        #저장할 때는 hstack 으로 아래처럼 저장
        #transition = np.hstack([list(s[0]),list(s[1]), list(s[2]), list(np.r_[a, r]), list(s_[0]), list(s_[1]),list(s_[2])])
        if self.prior:
            idx, w, transition = memory.sample(self.batch_size)
        else:
            transition = memory.sample(self.batch_size)
        print('Learn - shape of transition:{}'.format(np.shape(transition)))
        print('Learn - transition{}'.format(transition))

        s = transition[:, :self.feature_size]
        a = transition[:, self.feature_size + 2]
        r = transition[:, self.feature_size + 1]
        s_ = transition[:, -self.feature_size:]
        print('Learn - s:{} a:{} r:{} s_:{}'.format(s, a, r, s_))

        #s, a, r, s_ = map(lambda x: np.vstack(x).astype('float32'), np.transpose(transition))
        #print('Learn - s:{} a:{} r:{} s_:{}'.format(s, a, r, s_))
        '''s = [[0] * 6 for _ in range(3)]
        s_ = [[0] * 6 for _ in range(3)]
        a = [0 for i in range(3)]
        r = [0 for i in range(3)]
        s[0] = np.hsplit(transition, self.feature_size)
        s[1] = transition[self.feature_size:, :self.feature_size]
        s[2] = transition[self.feature_size*2:, :self.feature_size]
        a = transition[self.feature_size*3, 3]
        r = transition[self.feature_size*3 + 3, 3]
        s_[0] = transition[:self.feature_size*3 + 6, :self.feature_size]
        s_[1] = transition[:self.feature_size*4 + 6, :self.feature_size]
        s_[2] = transition[:, -self.feature_size:]
        print('Learn - s[0]:{} s[1]:{} s[2]:{} a:{} r:{} s_[0]:{} s_[1]:{} s_[2]:{}'.format(s[0],s[1],s[2], a, r,s_[0],s_[1], s_[2]))
        '''

        if self.prior:
            print("shape of s: ", np.shape(s))
            print("shape of a: ", np.shape(a))
            print("shape of r: ", np.shape(r))
            print("shape of s_: ", np.shape(s_))

            #s_ = np.reshape(s_,[144,4])
            #print("after reshape s_: ", np.shape(s_))
            # s_.shape is now (1,4)
            #s_ = tf.one_hot(s_, self.feature_size)
            #print("after reshape s_: ", np.shape(s_))

            #index = self.q_eval_model.predict([s_, np.ones((self.batch_size, 1))]).argmax(axis=1)
            index = self.q_eval_model.predict([s_, np.ones((self.batch_size*3, 1))]).argmax(axis=1)
            print('Learn - index:{}'.format(index))
            max_q = self.q_target_model.predict([s_, np.ones((self.batch_size*3, 1))])[range(self.batch_size), index]
            #max_q = np.max(index[0])

            q_predict = self.q_eval_model.predict([s, np.ones((self.batch_size*3, 1))])
        else:
            index = self.q_eval_model.predict(s_).argmax(axis=1)
            max_q = self.q_target_model.predict(s_)[range(self.batch_size), index]
            q_predict = self.q_eval_model.predict(s)

        print("shape of index: ", np.shape(index))
        print("shape of max_q: ", np.shape(max_q))
        print("shape of q_predict: ", np.shape(q_predict))

        #max_q = np.reshape(max_q,[32,3])

        #print("q_predict : ", q_predict)
        q_target = np.copy(q_predict)
        #print("q_target : ", q_target)
        #q_target[range(self.batch_size*3), s.astype(np.int32)] = r + self.gamma * max_q
        q_target[range(self.batch_size), transition[:, self.feature_size].astype(np.int32)] = r + self.gamma * max_q

        if self.prior:
            q_pred = self.q_eval_model.predict([s, np.ones((self.batch_size*3, 1))])
            p = np.sum(np.abs(q_pred - q_target), axis=1)
            memory.update(idx, p)
            report = self.q_eval_model.fit([s, w], q_target, verbose=0)
        else:
            report = self.q_eval_model.fit(s, q_target, verbose=0)

        self.epsilon = self.epsilon_min + (self.epsilon - self.epsilon_min) * self.epsilon_decrease

        self.history.append(report.history['loss'])

        if self.verbose and not self.learning_cnt % 100:
            print('training', self.learning_cnt, ': loss', report.history['loss'])

        self.learning_cnt += 1

    def append_sample(self, state, action, reward, next_state, done):
        target = self.model.predict(state)
        old_val = target[0][action]

        if done:
            target[0][action] = reward
        else:
            t = self.target_model.predict(next_state)[0]
            target[0][action] = reward + self.gamma * np.amax(t)

        #target_val = self.target_model(Variable(torch.FloatTensor(next_state))).data
        target_val = self.target_model.predict(next_state)[0]
        if done:
            target[0][action] = reward
        else:
            target[0][action] = reward + self.discount_factor * np.amax(target_val)

        error = abs(old_val - target[0][action])
        print("@ append_sample, error : ", error)
        #memory.add(error, (state, action, reward, next_state, done))
