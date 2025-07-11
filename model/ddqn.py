"""
ddqn.py
"""
__author__ = "py81.kim@gmail.com"
__credits__ = "https://github.com/puyo1999"

import tensorflow as tf
import tensorflow.compat.v1 as tf1
'''
tf1.enable_eager_execution()
tf1.disable_v2_behavior()
from tensorflow.python.framework.ops import enable_eager_execution
enable_eager_execution()
'''

#ddqn
from tensorflow.python.framework.ops import disable_eager_execution
# A2C 때에는 막아야 합니다 !
#disable_eager_execution()

import sys
# Increase the recursion limit
sys.setrecursionlimit(2000)

import numpy as np
import keras.backend as K
from keras.models import Sequential, Model, clone_model
from keras.layers import Dense, Activation, Lambda, Layer
from keras.layers import Input, LSTM, TimeDistributed, RepeatVector, Reshape, Dropout, Bidirectional, Concatenate
#from keras.layers.normalization import BatchNormalization
#from keras.optimizers import RMSprop
from keras.optimizer_v1 import RMSprop
from keras.optimizer_v1 import Adam

import random
from py_lab.lib import logger
logger = logger.get_logger(__name__)

import torch
import torch.nn.functional as F

class CustomLayer(Layer):
    def __init__(self, **kwargs):
        super(CustomLayer, self).__init__(**kwargs)

    def call(self, inputs):
        y_pred, input_tensor = inputs
        return tf.reduce_mean(tf.square(y_pred) * input_tensor)
        #return tf.reshape(input_tensor, (-1, 1))

    def call_(self, inputs):
        return tf.reshape(inputs, (-1, 1))

# Instantiate the custom layer
custom_layer = CustomLayer()

class DDQN:

    def __init__(self, feature_size, learning_rate, state_size, actions, action_size, step_size, prior, memory, gamma, epsilon=.5, epsilon_min=.1, epsilon_decrease=.9, name='DDQNetwork'):
        with tf1.variable_scope(name):
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

            #self.q_eval_model = None
            #self.q_target_model = None

            self.q_eval_model = self.build_model()
            #self.q_eval_model = self._build_model()
            self.q_target_model = self._build_model()

            self.optimizer = Adam(self.lr)

            self.update_target_model()

    def _huber_loss(self, y_true, y_pred, clip_delta=1.0):
        error = y_true - y_pred
        cond  = K.abs(error) <= clip_delta

        squared_loss = 0.5 * K.square(error)
        quadratic_loss = 0.5 * K.square(clip_delta) + clip_delta * (K.abs(error) - clip_delta)

        return K.mean(tf.where(cond, squared_loss, quadratic_loss))

    def build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss=self._huber_loss, optimizer='Adam')
        return model

    def _build_model(self):
        # Q-Evaluation Model
        HIDDEN_SIZE = 4

        def weight_loss_wrapper(input_tensor):
            def weight_loss(y_true, y_pred):
                return K.mean(K.square(y_true - y_pred) * input_tensor)
            return weight_loss

        def mse_loss(y_true, y_pred):
            return K.mean(K.square(y_true - y_pred))

        #inputs = tf.placeholder(tf.float32, [None,self.step_size, self.state_size], name='inputs')
        #actions = tf.placeholder(tf.int32, [None], name='actions')
        #one_hot_actions = tf.one_hot(self.actions, self.action_size)

        #inputs = Input(shape=(self.feature_size,))
        inputs = Input(shape=(6,))
        if self.prior:
            #weights = Input(shape=(self.feature_size,))
            #weights = Input(shape=(6,))
            weights = Input(shape=(1,))
        fc_1 = Dense(HIDDEN_SIZE, activation='relu')(inputs)
        #fc_2 = Dense(len(self.actions))(fc_1)
        if self.prior:
            #self.q_eval_model = Model([inputs, weights], fc_2)
            custom_layer = CustomLayer()([inputs, weights])
            custom_layer = tf.reshape(custom_layer, (-1, 1))
            fc_2 = Dense(self.action_size)(custom_layer)
            self.q_eval_model = Model([inputs,weights], fc_2)
            self.q_eval_model.compile(loss=weight_loss_wrapper(weights))
        else:
            fc_2 = Dense(self.action_size)(fc_1)  # action_size : 3
            self.q_eval_model = Model(inputs, fc_2)
            self.q_eval_model.compile(loss=mse_loss, optimizer='RMSProp')


        self.q_eval_model.summary()
        return self.q_eval_model

    #@tf.function
    def learn_loss(self, predictions, targets):
        #self.optimizer.zero_grad()
        loss = self.mse_loss(predictions, targets)
        #loss.backward()
        #self.optimizer.step()
        logger.error(f'@ learn_loss : {loss}')
        return loss

    def my_mse_loss(self, y, y_pred):
        logger.error(f'@ my_mse_loss - y.mean : {np.mean(y, axis=1)}')
        logger.error(f'@ my_mse_loss - y_pred.mean : {np.mean(y_pred, axis=1)}')
        mse = np.mean(np.square(y - y_pred))
        rmse = np.sqrt(mse)
        #return (np.mean(y, axis=1)-np.mean(y_pred, axis=1))**2
        return rmse

    def mse_loss(self, y_true, y_pred):
        #err = y_true - y_pred
        a = K.ones_like(y_true)  # use Keras instead so they are all symbolic
        logger.error(f'@ mse_loss - a : {a}')

        y_true = tf.cast(y_true, tf.float32)

        loss = K.mean(K.square(y_pred - y_true)) + a
        #loss = tf.math.reduce_mean(tf.math.square(err))
        logger.error(f'@ mse_loss - loss : {loss}')
        return loss

    def _copy_model(self):
        self.q_target_model = clone_model(self.q_eval_model)
        self.q_target_model.set_weights(self.q_eval_model.get_weights())

    def store_transition(self, s, a, r, s_):
        #logger.info('shape of s:{} a:{}'.format(np.shape(s), np.shape(a)))
        #logger.info('StoreTransition - s:{} a:{} r:{} s_:{}'.format(s,a,r,s_))
        #transition = np.hstack([list(s[0]), list(s[1]), list(s[2]), list(np.r_[a, r]), list(s_[0]), list(s_[1]), list(s_[2])])
        transition = np.hstack([list(s[0]), list(s[1]), list(s[2]), list(s[3]), list(np.r_[a, r]), list(s_[0]), list(s_[1]), list(s_[2]), list(s_[3])])
        logger.error(f'@@ StoreTransition - transition:{transition}')
        #self.memory.store(transition)
        error = 1007
        self.memory.add(transition, error=error)
        self.step_cnt += 1

    def update_target_model(self):
        # copy weights from model to target_model
        self.q_target_model.set_weights(self.q_eval_model.get_weights())

    def choose_action(self, observation):
        logger.error(f"@ ddqn / choose_action - step_cnt : {self.step_cnt}")
        temp_epsilon = 0.5 * ( 1/self.step_cnt )
        logger.error(f"@ ddqn / choose_action - temp_epsilon : {temp_epsilon}")
        observation = np.array(observation)
        logger.error(f"@ ddqn / choose_action - obs array : {observation}")
        observation = tf.one_hot(observation, 6)
        logger.error(f"!!! after one_hot obs : {observation}")
        #if np.random.random() > self.epsilon:
        if np.random.uniform(0,1) > temp_epsilon:
            #state = torch.tensor([observation],dtype=torch.float).to(self.q_eval_model.device)
            actions = self.q_eval_model.predict([observation, np.ones((1,1))], steps=1)
            logger.error(f'@ ddqn / choose_action - actions : {actions}')
            action = np.argmax(actions).item()
        else:
            action = np.random.choice(self.actions)

        logger.error(f'@ ddqn / choose_action - action:{action}')
        return action

    def actor(self, observation):
        """
        if np.random.uniform(0, 1) < self.epsilon:
            action = np.random.choice(self.actions)
        else:
        """
        observation = np.array(observation)
        if self.prior:
            logger.info("@ actor - prior case")
            logger.info("!!! obs : ", observation)
            #observation = tf.one_hot(observation, self.feature_size)
            observation = tf.one_hot(observation, 6)
            logger.info("!!! after one_hot obs : ", observation)
            q_value = self.q_eval_model.predict([observation, np.ones((1, 1))],steps=1)
            logger.info("!!! q_value : ", q_value)
        else:
            q_value = self.q_eval_model.predict(observation)
        action = self.actions[q_value.argmax()]
        logger.info('@ actor - action:{}'.format(action))
        return action

    def learn_(self, cur_state, action, next_state, discnt_rewards):
        if self.learning_cnt % self.replace_target_iter == 0:
            self._copy_model()
            if self.verbose:
                logger.error('Copy model')
        logger.error(f'cur_state : {cur_state}')
        logger.error(f'next_state : {next_state}')

        cur_state = np.array(cur_state)
        next_state = np.array(next_state)

        adjusted_input = np.ones((1, 1))

        cur_state = cur_state.reshape((6, 6))
        #next_state = next_state.reshape((6, 6))
        next_state = next_state.reshape((-1, 6))

        cur_state = np.resize(cur_state, [1, 6])
        next_state = np.resize(next_state, [1, 6])
        index = self.q_eval_model.predict([cur_state, adjusted_input]).argmax(axis=1)
        # Predict the index
        logger.error(f'eval predict - index:{index}')

        max_q = self.q_target_model.predict([next_state, adjusted_input]).argmax(axis=1)
        logger.error(f'target predict - max_q:{max_q}')
        q_predict = self.q_eval_model.predict([cur_state, adjusted_input])

        q_target = np.copy(q_predict)
        #q_target = np.array(q_target)
        q_target[0][action] = discnt_rewards + self.gamma * max_q
        logger.error(f"q_predict.shape : {q_predict.shape}")
        logger.error(f"q_target.shape : {q_target.shape}")

        report = self.q_eval_model.fit([cur_state, adjusted_input], q_target, epochs=1, verbose=0)
        rmse = np.sqrt(report.history["loss"])
        logger.error(f'report history loss : {report.history["loss"]}, rmse : {rmse}')
        return rmse


        #loss = self.my_mse_loss(q_predict, q_target)
        #logger.error(f'mse_loss : {loss}')
        #return loss


    def learn(self, memory, replay_memory, batch_size):
        if self.learning_cnt % self.replace_target_iter == 0:
            self._copy_model()
            if self.verbose:
                logger.info('Copy model')

        #minibatch = random.sample(replay_memory, batch_size)
        #for sample in enumerate(minibatch):
            #s, a, r, s_ = sample
        minibatch = random.sample(replay_memory, batch_size)

        for index, sample in enumerate(minibatch):
            cur_state, action, reward, next_state = sample

        logger.info(f'Learn - shape of sample : {np.shape(sample)}')
        logger.info(f'Learn - sample : {sample}')
        logger.info(f'Learn - cur_state : {cur_state}')
        logger.info(f'Learn - next_state : {next_state}')
        '''
        transition = random.sample(replay_memory, batch_size)
        logger.info("transition:")
        logger.info(transition)
        s, a, r, s_ = map(lambda x: np.vstack(x).astype('float32'), np.transpose(transition))
        '''

        #저장할 때는 hstack 으로 아래처럼 저장
        #transition = np.hstack([list(s[0]),list(s[1]), list(s[2]), list(np.r_[a, r]), list(s_[0]), list(s_[1]),list(s_[2])])
        if self.prior:
            idx, w, transition = memory.sample(self.batch_size)
        else:
            transition = memory.sample(self.batch_size)
        logger.info(f'Learn - shape of transition : {np.shape(transition)}')
        transition = np.array(transition[0])

        logger.info(f'Learn - transition : {transition}')
        s = transition[:36]
        a = transition[36:40]
        r = transition[40:44]
        s_ = transition[44:80]
        logger.info(f'Learn - s:{s} a:{a} r:{r} s_:{s_}')
        #s, a, r, s_ = map(lambda x: np.vstack(x).astype('float32'), np.transpose(transition))
        #logger.info('Learn - s:{} a:{} r:{} s_:{}'.format(s, a, r, s_))
        '''
        s = [[0] * 6 for _ in range(3)] # cannot reshape array of size 24 into shape (6,6)
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
        logger.info('Learn - s[0]:{} s[1]:{} s[2]:{} a:{} r:{} s_[0]:{} s_[1]:{} s_[2]:{}'.format(s[0],s[1],s[2], a, r,s_[0],s_[1], s_[2]))
        '''

        if self.prior:
            logger.info("shape of s: ", np.shape(s))
            logger.info("shape of a: ", np.shape(a))
            logger.info("shape of r: ", np.shape(r))
            logger.info("shape of s_: ", np.shape(s_))

            #s_ = np.reshape(s_,[144,4])
            #logger.info("after reshape s_: ", np.shape(s_))
            # s_.shape is now (1,4)
            #s_ = tf.one_hot(s_, self.feature_size)
            #logger.info("after reshape s_: ", np.shape(s_))

            logger.error(f'cur_state : {cur_state}')
            logger.error(f'next_state : {next_state}')

            cur_state = np.array(cur_state)
            next_state = np.array(next_state)

            #adjusted_input = np.ones((6, 6))
            adjusted_input = np.ones((1, 1))

            if np.shape(cur_state) != np.shape(next_state):
                next_state = next_state.reshape((4, 6))
            else:
                next_state = next_state.reshape((6, 6))
            cur_state = cur_state.reshape((6, 6))


            #adjusted_input = adjusted_input.reshape((6, 6))
            cur_state = np.resize(cur_state, [1, 6])
            next_state = np.resize(next_state, [1, 6])
            s_ = np.tile(s_, (3, 1))

            #index = self.q_eval_model.predict([s_, np.ones((self.batch_size, 1))]).argmax(axis=1)
            #index = self.q_eval_model.predict([s_, np.ones((self.batch_size*3, 1))]).argmax(axis=1)
            index = self.q_eval_model.predict([cur_state, adjusted_input]).argmax(axis=1)
            # Predict the index
            logger.error(f'eval predict - index:{index}')


            #max_q = self.q_target_model.predict([s_, np.ones((self.batch_size*3, 1))])[range(self.batch_size*3), index]
            max_q = self.q_target_model.predict([next_state, adjusted_input]).argmax(axis=1)
            #max_q = np.max(index[0])
            logger.error(f'target predict - max_q:{max_q}')

            s = np.tile(s, (3, 1))
            #q_predict = self.q_eval_model.predict([s, np.ones((self.batch_size*3, 1))])
            q_predict = self.q_eval_model.predict([cur_state, adjusted_input])
        else:
            index = self.q_eval_model.predict(s_).argmax(axis=1)
            max_q = self.q_target_model.predict(s_)[range(self.batch_size), index]
            q_predict = self.q_eval_model.predict(s)

        logger.info("shape of index: ", np.shape(index))
        logger.info("shape of max_q: ", np.shape(max_q))
        logger.info("shape of q_predict: ", np.shape(q_predict))

        #max_q = np.reshape(max_q,[32,3])
        #max_q = max_q.reshape(3, -1).mean(axis=0)
        logger.error(f'## max_q : {max_q}\n')
        #logger.info("q_predict : ", q_predict)
        q_target = np.copy(q_predict)
        logger.error(f"## q_target : {q_target}\n")
        #q_target = np.tile(q_target, (3,1))
        #q_target[range(self.batch_size*3), s.astype(np.int32)] = r + self.gamma * max_q
        #q_target[range(self.batch_size), transition[:, self.feature_size].astype(np.int32)] = r + self.gamma * max_q
        #q_target = r + self.gamma * max_q
        #q_target = np.tile(q_target, (6, 1))

        batch_index = np.arange(6, dtype=np.int32)
        #q_target[0, batch_index] = r[:6] + self.gamma * max_q

        # q_target 업데이트
        #q_target = np.tile(q_target, (6, 1))  # q_target의 크기를 입력 데이터에 맞추기 위해 조정

        # q_target 업데이트

        #q_target[batch_index, :] = r[:6].reshape(-1, 1) + self.gamma * max_q.reshape(-1, 1)
        #q_target[batch_index] = r[:6].reshape(-1, 1) + self.gamma * max_q.reshape(-1, 1)
        logger.error(f'{np.shape(cur_state)} {np.shape(adjusted_input)}, {np.shape(q_target)}')
        report = self.q_eval_model.fit([cur_state, adjusted_input], q_target, epochs=1, verbose=0)

        if self.prior:
            #q_pred = self.q_eval_model.predict([s, np.ones((self.batch_size*3, 1))])
            q_pred = self.q_eval_model.predict([cur_state, adjusted_input])
            q_pred = np.array(q_pred)
            if q_pred.shape != q_target.shape:
                q_target = q_target[:6]
            q_target = np.array(q_target)
            #q_target = q_target[:6]
            logger.info("@ q_pred : ", q_pred)
            logger.info("@ q_target : ", q_target)
            p = np.sum(np.abs(q_pred - q_target), axis=1)
            #memory.update(idx, p)
            memory.update(index, p)

            w = np.repeat(w, 3)
            logger.info("@ shape of s: ", np.shape(s))
            logger.info("@ shape of w: ", np.shape(w))
            logger.info("@ shape of q_target: ", np.shape(q_target))

            q_target = q_target.reshape((1, 6))
            # Ensure target is a numpy array
            '''
            if isinstance(q_target, np.ndarray):
                q_target = q_target[0]  # Use the first element if target is an array
            else:
                q_target = np.array([q_target])  # Convert to array if target is not an array
            '''

            # Adjust the shape of target if needed
            #q_target = np.tile(q_target, (6, 1))  # Adjust the shape to (6, 1)

            logger.error(f'q_pred shape: {np.shape(q_pred)}')
            logger.error(f'q_target shape: {np.shape(q_target)}')

            # Calculate the target values correctly

            #if q_target.shape[0] > 0 and action < q_target.shape[1]:
            q_target[0][action] = reward + self.gamma * max_q
            logger.error(f"target_Q_values : {q_target}")
            #else:
                #logger.error("배열 크기 이상")

            #report = self.q_eval_model.fit([s, w], q_target, verbose=0)
            report = self.q_eval_model.fit([cur_state, adjusted_input], q_target, epochs=1, verbose=0)
            #loss = self.q_eval_model.evaluate([cur_state, adjusted_input], q_target)
            loss = self.my_mse_loss(q_pred, q_target)
            logger.error(f'my_mse_loss : {loss}')
        else:
            report = self.q_eval_model.fit(s, q_target, verbose=0)

        self.epsilon = self.epsilon_min + (self.epsilon - self.epsilon_min) * self.epsilon_decrease

        self.history.append(report.history['loss'])

        if self.verbose and not self.learning_cnt % 100:
            logger.info('training', self.learning_cnt, ': loss', report.history['loss'])

        self.learning_cnt += 1


        logger.error(f'ddqn, learn - learning_cnt : {self.learning_cnt}')
        logger.error(f'ddqn, learn - report history loss : {report.history["loss"]}')
        logger.error(f'ddqn, learn - loss : {loss}')
        return loss

    def append_sample(self, state, action, reward, next_state, done):
        target = self.q_eval_model.predict(state)
        old_val = target[0][action]

        if done:
            target[0][action] = reward
        else:
            t = self.q_target_model.predict(next_state)[0]
            target[0][action] = reward + self.gamma * np.amax(t)

        #target_val = self.target_model(Variable(torch.FloatTensor(next_state))).data
        target_val = self.q_target_model.predict(next_state)[0]
        if done:
            target[0][action] = reward
        else:
            target[0][action] = reward + self.discount_factor * np.amax(target_val)

        error = abs(old_val - target[0][action])
        logger.info("@ append_sample, error : ", error)
        #memory.add(error, (state, action, reward, next_state, done))
