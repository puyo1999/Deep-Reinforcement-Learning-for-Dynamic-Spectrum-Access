#import tensorflow.compat.v1 as tf
#tf.disable_v2_behavior()

import tensorflow as tf
import keras
import keras as K
import keras.backend as KB
from keras.layers import Input, Dense, Flatten
from keras.optimizers import Adam
import torch
import gym
import numpy as np
import random as rand
from model.base import Algorithm
import logging
logger = logging.getLogger(__name__)

LOSS_CLIPPING = 0.1

from tensorflow.python.framework.ops import enable_eager_execution
enable_eager_execution()

import tensorflow.compat.v1 as tf


class PPOAgent(Algorithm):
#class PPOAgent(object):
    def __init__(self, env, sess, memory, action_n, state_dim, training_batch_size):
        #self.env = gym.make('CartPole-v1')
        self.env = env
        # setting the our created session as default session
        tf.keras.backend.set_session(sess)
        self.sess = sess
        self.action_n = action_n
        self.state_dim = state_dim
        self.value_size = 1

        self.node_num = 32
        self.learning_rate_actor = 0.0005
        self.learning_rate_critic = 0.0005
        self.epochs_cnt = 5

        self.discount_rate = 0.98
        self.smooth_rate = 0.95
        self.episode_num = 500

        #CONSTANTS
        self.TRAINING_BATCH_SIZE = training_batch_size
        self.TARGET_UPDATE_ALPHA = 0.95
        self.GAMMA = 0.99
        self.GAE_LAMBDA = 0.95
        self.CLIPPING_LOSS_RATIO = 0.1
        self.ENTROPY_LOSS_RATIO = 0.001
        self.TARGET_UPDATE_ALPHA = 0.9

        self.model_actor = self.build_model_actor()
        self.model_critic = self.build_model_critic()

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.7)

        self.model_actor_old = self.build_model_actor()
        self.model_actor_old.set_weights(self.model_actor.get_weights())

        self.dummy_advantage = np.zeros((1,1))
        self.dummy_old_prediction = np.zeros((1, self.action_n))

        self.advantage_input = tf.placeholder(dtype=tf.float32, shape=[None, 1], name='advantage_input')
        self.old_prediction = tf.placeholder(dtype=tf.float32, shape=[None, 6], name='old_prediction_input')

        self.memory = memory


    '''class MyModel(keras.Model):
        def train_step(self, data):
            in_datas, out_action_probs = data
            states, action_matrixs, advantages = in_datas[0], in_datas[1], in_datas[2]
            with tf.GradientTape() as tape:
                y_pred = self(states, training=True)
                new_policy = K.max(action_matrixs*y_pred, axis=-1)
                old_policy = K.max(action_matrixs*out_action_probs, axis=-1)
                r = new_policy/(old_policy)
                clipped = K.clip(r, 1-LOSS_CLIPPING, 1+LOSS_CLIPPING)
                loss = -K.minimum(r*advantages, clipped*advantages)
            trainable_vars = self.trainable_variables
            gradients = tape.gradient(loss, trainable_vars)
            self.optimizer.apply_gradients(zip(gradients, trainable_vars))
    '''

    def build_model_actor(self):

        #state = Input(shape=self.state_dim, name='state')
        #state = Input(shape=(1,), name='state')

        #state = Input(shape=[self.action_n, self.state_dim])
        state = Input(shape=[self.state_dim, self.action_n + 2 + 1])
        advantage = Input(shape=[1,], name='advantage_input')
        old_prediction = K.layers.Input(shape=[self.action_n,], name='old_prediction_input')

        #state = Flatten()([state, advantage, old_prediction])
        dense = Dense(self.node_num, activation='relu', name='dense1')(state)
        dense = Dense(self.node_num, activation='relu', name='dense2')(dense)
        policy = Dense(self.action_n, activation='softmax', name='actor_output_layer')(dense)
        logger.error(f'@ build_model_actor - policy :{policy}')

        '''
        def ppo_loss(y_true, y_pred):
            return K.mean(K.square(y_true - y_pred))
        '''
        #model = K.Model(inputs=[state, advantage, old_prediction], outputs=policy)
        model = K.Model(inputs=[state], outputs=[policy])

        model.compile(
            optimizer='Adam',
            loss=self.ppo_loss(advantage=advantage, old_pred=old_prediction)
            #loss=self.ppo_loss,
        )
        model.summary()
        return model

    def get_action_prob(self, inputs, training=None, mask=None):
        print(f'@ get_action_prob - inputs:\n{inputs}')
        dense = Dense(self.node_num, activation='relu', name='dense1')(inputs)
        dense = Dense(self.node_num, activation='relu', name='dense2')(dense)
        policy = Dense(self.action_n, activation='softmax', name='actor_output_layer')(dense)
        logger.error(f'policy : {policy}')
        return policy

    def build_model_critic(self):

        #state = Input(shape=(self.state_dim), name='state_input')
        state = Input(shape=[self.state_dim, self.action_n + 2 + 1], name='state_input')

        #state = Input(shape=(1,), name='state_input')
        dense = Dense(32, activation='relu')(state)
        dense = Dense(32, activation='relu')(dense)

        V = K.layers.Dense(1, name='actor_output_layer')(dense)

        model = K.Model(inputs=[state], outputs=V)
        model.compile(
            optimizer='Adam',
            loss='mean_squared_error',
        )
        model.summary()
        return model

    def ppo_loss_by_torch(self, predictions, old_predictions, rewards, advantages):
        # Calculate the log probabilities of the predicted actions
        log_probs = torch.log(predictions)
        old_log_probs = torch.log(old_predictions)

        ratio = log_probs / (old_log_probs + 1e-10)
        clip_ratio = K.backend.clip(ratio, min_value=1 - self.CLIPPING_LOSS_RATIO,
                                    max_value=1 + self.CLIPPING_LOSS_RATIO)
        logger.info(f'@ ppo_loss - ratio:\n{ratio}')
        surrogate1 = ratio[0] * advantages
        surrogate2 = clip_ratio[0] * advantages
        surrogate1 = tf.cast(surrogate1, dtype='float32')

        entropy_loss = (log_probs[0] * K.backend.log(log_probs[0] + 1e-10))
        ppo_loss = -K.backend.mean(K.backend.minimum(surrogate1, surrogate2) + self.ENTROPY_LOSS_RATIO * entropy_loss)
        logger.info(f'@ ppo_loss - ppo_loss:\n{ppo_loss}')
        '''
        # Calculate the surrogate loss using the advantages
        surrogate_loss = -advantages * log_probs

        # Calculate the clipping loss
        clip_loss = torch.clamp(log_probs - old_log_probs, min=-0.2, max=0.2)
        clip_loss = clip_loss * advantages

        # Return the sum of the surrogate and clipping losses
        return surrogate_loss + clip_loss
        '''
        return ppo_loss

    def choose_action_(self, state):
        assert isinstance(state, np.ndarray), "state must be numpy.ndarry"
        logger.error(f'@ choose_action_ : state:\n{state}')
        #state = np.reshape(state, [-1, self.dic_agent_conf["STATE_DIM"][0]])
        #state = np.expand_dims(state, axis=0)
        state = state.reshape((1, 4, 9))
        logger.error(f'@ after reshape state:\n{state}')

        prob = self.model_actor.predict_on_batch([state, self.dummy_advantage, self.dummy_old_prediction]).flatten()
        logger.error(f'before prob:\n{prob}')
        logger.error(f'before prob shape : {prob.shape}')
        prob = prob[:6]
        logger.error(f'after prob:\n{prob}')
        logger.error(f'after prob shape : {prob.shape}')
        action = np.random.choice(self.action_n, p=prob)
        return action

    def choose_action(self, state):
        logger.error(f'state:\n{state}')
        probs = self.model_actor.predict(state)[0]
        logger.error(f'probs:\n{probs}')
        probs = probs[0][:6]
        probs = np.ravel(probs)
        logger.error(f'probs shape : {probs.shape}')
        action = np.random.choice(self.action_n, p=probs)
        return action

    def get_action_(self, state):
        policy = self.model_actor.predict(state, batch_size=1).flatten()
        return np.random.choice(self.action_n, 1, p=policy)[0]

    def discount_rewards(self, rewards):
        discounted_rewards = np.zeros_like(rewards)
        running_add = 0
        for t in reversed(range(len(rewards))):
            running_add = running_add * self.gamma + rewards[t]
            discounted_rewards[t] = running_add
        return discounted_rewards

    def surrogate_loss(self, old_probs, advantages, chosen_probs):
        ratio = chosen_probs / old_probs
        clipped_ratio = tf.clip_by_value(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio)
        surrogate_loss = tf.minimum(ratio * advantages, clipped_ratio * advantages)
        return -tf.reduce_mean(surrogate_loss)

    def train_on_batch(self, states, actions, advantages):
        old_probs = self.actor.predict(states)
        chosen_probs = tf.reduce_sum(old_probs * tf.one_hot(actions, self.action_n), axis=1)
        with tf.GradientTape() as tape:
            loss = self.surrogate_loss(old_probs, advantages, chosen_probs)
        grads = tape.gradient(loss, self.actor.trainable_variables)
        self.actor.optimizer.apply_gradients(zip(grads, self.actor.trainable_variables))

    def compute_loss(self, old_policy, new_policy, actions, gaes):
        gaes = tf.reshape(gaes, [100])
        gaes = tf.stop_gradient(gaes)
        old_log_p = tf.math.log(
            tf.reduce_sum(old_policy * actions))
        old_log_p = tf.stop_gradient(old_log_p)
        log_p = tf.math.log(tf.reduce_sum(
            new_policy * actions))
        ratio = tf.math.exp(log_p - old_log_p)
        clipped_ratio = tf.clip_by_value(
            ratio, 1 - clip_ratio, 1 + clip_ratio)
        surrogate = -tf.minimum(ratio * gaes, clipped_ratio * gaes)
        return tf.reduce_mean(surrogate)

    def ppo_loss_with_GT(self, old_probs, states, actions, advantages, values, next_values, returns):
     action_masks = tf.one_hot(actions, self.action_n)
     probs = tf.nn.softmax(self.model_actor(states))

     # Ensure the number of elements match
     if tf.size(action_masks) == tf.size(probs):
         action_masks = tf.reshape(action_masks, probs.shape)
     else:
         print("The number of elements in action_masks and probs do not match.")


     # One-hot encode actions
     one_hot_actions = tf.one_hot(actions, self.action_n)

     # Expand dimensions to make it rank 3
     one_hot_actions = tf.expand_dims(one_hot_actions, axis=0)

     # 패딩을 추가하여 크기를 [1, 9, 6]으로 변경
     #paddings = tf.constant([[0, 0], [0, 5], [0, 0]])
     #padded_one_hot_actions = tf.pad(one_hot_actions, paddings, "CONSTANT")

     # Perform element-wise multiplication
     chosen_probs = tf.reduce_sum(probs * one_hot_actions, axis=1)
     # Now perform the element-wise multiplication
     #chosen_probs = tf.reduce_sum(action_masks * probs)

     old_chosen_probs = tf.reduce_sum(old_probs * one_hot_actions, axis=1)
     #old_chosen_probs = action_masks * old_probs

     logger.info("# chosen_probs shape:", chosen_probs.shape)
     logger.info("# old_chosen_probs shape:", old_chosen_probs.shape)

     max_len = max(chosen_probs.shape[0], old_chosen_probs.shape[0])
     # Tile the smaller tensor to match the larger tensor's shape
     if chosen_probs.shape[0] < max_len:
         chosen_probs = tf.tile(chosen_probs, [max_len // chosen_probs.shape[0] + 1])[:max_len]
     if old_chosen_probs.shape[0] < max_len:
         old_chosen_probs = tf.tile(old_chosen_probs, [max_len // old_chosen_probs.shape[0] + 1])[:max_len]

     # Print new shapes
     logger.error(f"New chosen_probs shape: {chosen_probs.shape}")
     logger.error(f"New old_chosen_probs shape: {old_chosen_probs.shape}")

     ratio = chosen_probs / (old_chosen_probs + 1e-10)
     clipped_ratio = tf.clip_by_value(ratio, 1.0 - self.CLIPPING_LOSS_RATIO, 1.0 + self.CLIPPING_LOSS_RATIO)
     logger.error(f"clipped_ratio: {clipped_ratio}")

     surrogate1 = ratio * advantages
     surrogate2 = clipped_ratio * advantages
     logger.error(f"surrogate1: {surrogate1}")
     logger.error(f"surrogate2: {surrogate2}")
     policy_loss = -tf.reduce_mean(tf.minimum(surrogate1, surrogate2))

     value_loss = tf.reduce_mean(tf.square(values - returns))

     entropy_loss = -tf.reduce_mean(tf.reduce_sum(probs * tf.math.log(probs + 1e-10), axis=1))

     # Assuming you are using TensorFlow 1.x

     total_loss = tf.reduce_mean(tf.reduce_sum(policy_loss + 0.5 * value_loss - 0.01 * entropy_loss))
     total_loss_array = self.sess.run(total_loss)

     # Log the array
     #logger.error(f'*********** total_loss : {total_loss_array} **********')

     #total_loss = policy_loss + 0.5 * value_loss - 0.01 * entropy_loss
     #total_loss = tf.reduce_mean(tf.reduce_sum( policy_loss + 0.5 * value_loss - 0.01 * entropy_loss))

     logger.error(f'*********** total_loss : {total_loss} **********')
     #logger.error(f'*********** total_loss_array : {total_loss_array} **********')
     return total_loss, total_loss_array

    #@tf.function
    def ppo_loss(self, advantage, old_pred):
        """
        :param advantage:
        :param old_prediction:
        :return:
        """
        def loss(y_true, y_pred):
            prob = y_true * y_pred
            old_prob = y_true * old_pred
            ratio = prob / (old_prob + 1e-10)
            clip_ratio = K.backend.clip(ratio, min_value=1-self.CLIPPING_LOSS_RATIO, max_value=1+self.CLIPPING_LOSS_RATIO)
            surrogate1 = ratio * advantage
            surrogate2 = clip_ratio * advantage
            entropy_loss = (prob * K.backend.log(prob+1e-10))
            ppo_loss = -K.backend.mean(K.backend.minimum(surrogate1, surrogate2)+self.ENTROPY_LOSS_RATIO * entropy_loss)
            logger.error(f'@ ppo_loss - ppo_loss:\n{ppo_loss}')
            return ppo_loss
        return loss

    def ppo_loss_new(self, advantages, old_predictions, predictions):
        # Clip the predicted actions to ensure stability
        predictions = tf.clip_by_value(predictions, 1e-8, 1-1e-8)
        old_predictions = tf.clip_by_value(old_predictions, 1e-8, 1-1e-8)

        # Calculate the ratio of the new and old predictions
        ratio = predictions[0] / old_predictions[0]
        logger.info(f'@@ ppo_loss_new - ratio :\n{ratio}\n')
        clipped = KB.clip(ratio, 1 - LOSS_CLIPPING, 1 + LOSS_CLIPPING)
        logger.info(f'@@ ppo_loss_new - clipped :\n{clipped}\n')

        # Calculate the PPO loss using the ratio and advantages
        loss = KB.minimum(ratio * advantages, clipped * advantages)
        loss = -tf.reduce_mean(loss)

        logger.info(f'@@ ppo_loss_new - loss :\n{loss}\n')
        return loss

    def make_gae(self):
        gae = 0
        mask = 0
        gae_cumulated = []
        logger.error(f'@@ make_gae - memory.cnt_samples:{self.memory.cnt_samples}')

        for i in reversed(range(self.memory.cnt_samples)):
            if i >= 100:
                continue
            logger.info(f'@@ make_gae - idx:{i}')
            #mask = 0 if self.memory.batch_done[i] else 1 # mask = 1-done
            mask = 1
            logger.error(f'@@ make_gae - batch_r[{i}]:{self.memory.batch_r[i]}')
            logger.info(f'@@ make_gae - batch_s[{i}]:{self.memory.batch_s[i]}')
            logger.info(f'@@ make_gae - batch_s_[{i}]:{self.memory.batch_s_[i]}')
            v = self.get_v(self.memory.batch_s[i])
            v_ = self.get_v(self.memory.batch_s_[i])
            logger.error(f'@@ make_gae - shape of batch_r:{np.shape(self.memory.batch_r[i])}')
            logger.error(f'@@ make_gae - shape of v_:{np.shape(v_)}')
            if np.shape(v_) == (4,1):
                delta = self.memory.batch_r[i] + self.GAMMA * self.get_v(self.memory.batch_s_[i]).reshape(4,) * mask - v
            else: # (1,4,1)
                delta = self.memory.batch_r[i] + self.GAMMA * v_ * mask - v
            logger.error(f'@@ make_gae - shape of delta:{np.shape(delta)}')
            logger.error(f'@@ make_gae - delta :\n{delta}\n')
            gae = delta + self.GAMMA * self.GAE_LAMBDA * mask * gae
            logger.error(f'@@ make_gae - shape of gae:{np.shape(gae)}')
            logger.error(f'@@ make_gae - gae :\n{gae}\n')
            logger.error(f'@@ make_gae - shape of gae+v:{np.shape(gae+v)}')
            logger.error(f'@@ make_gae - gae+v :\n{gae+v}\n')

            mean_gae = np.mean(gae+v, axis=(1, 2))
            logger.error(f'@@ make_gae - shape of mean_gae:{np.shape(mean_gae)}')
            logger.error(f'@@ make_gae - mean_gae :\n{mean_gae}\n')
            #self.memory.batch_gae_r.append(gae + v)
            self.memory.batch_gae_r.append(mean_gae)

            gae_cumulated.append(mean_gae)

        self.memory.batch_gae_r.reverse()
        # [101] 모양으로 변경
        gae_cumulated = torch.tensor(gae_cumulated)
        gae_cumulated = gae_cumulated.squeeze()
        self.memory.GAE_CALCULATED_Q = True
        logger.error(f'@@ make_gae - shape of gae_cumulated:{np.shape(gae_cumulated)}')
        logger.error(f'@@ make_gae - gae_cumulated : {gae_cumulated}')
        return gae_cumulated

    def update_target_network(self):
        alpha = self.TARGET_UPDATE_ALPHA
        actor_weights = np.array(self.model_actor.get_weights(), dtype=object)
        actor_target_weights = np.array(self.model_actor_old.get_weights(), dtype=object)
        new_weights = alpha*actor_weights + (1-alpha)*actor_target_weights
        self.model_actor_old.set_weights(new_weights)

    '''
    def choose_action(self, state):
        """chooses an action within the action space given a state.
        The action is chosen by random with the weightings accoring to the probability
        params:
            :state: np.array of the states with state_dim length
        """
        assert isinstance(state, np.ndarray)
        # reshape for predict_on_batch which requires 2d-arrays
        # state = np.reshape(state, [-1, self.state_dim[0]])
        # the probability list for each action is the output of the actor network given a state
        prob = self.model_actor.predict_on_batch([state, self.dummy_advantage, self.dummy_old_prediction]).flatten()
        # action is chosen by random with the weightings according to the probability
        logger.info(f'prob:\n{prob}\n')
        prob_needed = prob[:6]
        action = np.random.choice(self.action_n, p=prob_needed)
        return action
    '''

    def store_transition(self, s, a, r, s_):
        #self.memory.add((s, a, r, s_, done))
        self.memory.store((s,a,r,s_))
        self.memory.store_each(s, a, r, s_, False)

    def get_v(self, state):
        """Returns the value of the state.
        Basically, just a forward pass though the critic networtk
        """
        logger.error(f'@ get_v - state :\n{state}') # [4,9]
        s = np.reshape(state,(self.state_dim, self.action_n + 3))
        s = np.expand_dims(s, axis=0)
        logger.error(f'@ after reshape get_v - s :\n{s}')
        v = self.model_critic.predict_on_batch(s)
        logger.error(f'@ get_v - v :\n{v}')
        return v

    def get_prediction(self, state):
        return self.model_actor.predict_on_batch()

    def get_old_prediction(self, state):
        """Makes an old prediction (an action) given a state on the actor_old_network.
        This is for the train_network --> ppo_loss
        """
        #state = np.expand_dims(state, axis=0)
        return self.model_actor_old.predict_on_batch(state)

    def learn_(self, states, actions, next_states, discnt_rewards):
        with tf.GradientTape() as tape1, tf.GradientTape() as tape2:
            action_prob = self.model_actor(states, training=True)
            critic = self.model_critic(states, training=True)

            print(f'@ learn_ - action_prob:\n{action_prob}\ncritic:\n{critic}')
            print(f'@ learn_ - action_prob[0]:\n{action_prob[0]}')
            action = self.get_action_(action_prob)

            print(f'@ learn_ - action: {action}')

            # rewards 를 discounted factor 로 다시 계산.
            returns = []
            discounted_sum = 0
            for r in discnt_rewards[::-1]:
                discounted_sum = r + self.discount_factor * discounted_sum
                returns.insert(0, discounted_sum)
            # Normalize
            returns = np.array(returns)
            returns = (returns - np.mean(returns)) / (np.std(returns) + self.eps)
            returns = returns.tolist()

            next_critic = self.model_critic(next_states, training=True)
            print(f'@ learn_ - next_critic:\n{next_critic}\ndiscnt_rewards:\n{discnt_rewards}')
            print(f'@ learn_ - returns[0] :\n{returns[0]}')
            advantage = returns[0] - critic[0,0]

            print(f'@ learn_ - advantage :\n{advantage}')
            print(f'@ learn_ - critic[0,0] :\n{critic[0,0]}')
            print(f'@ learn_ - action_prob[0][0] :\n{action_prob[0][0]}')
            # [ [prob, prob, ... ] ]형식으로 입력이 들어옴
            #a_loss = -tf.math.log(action_prob[0][0]) * np.transpose( advantage[:5][:3] )
            a_loss = -tf.math.log(action_prob) * np.transpose(advantage)

            critic = np.squeeze(critic)
            critic = np.mean(critic)
            print(f'@ learn_ - critic:\n{critic}\n')

            c_loss = huber_loss(critic, discnt_rewards)
            print(f'actor loss :\n{a_loss}')
            print(f'critic loss :\n{c_loss}')

            #Compute the Q value estimate of the target network
            Q_target = self.critic_target(next_states, self.actor_target(next_states))
            Q_target = np.squeeze(Q_target)
            #Q_target = np.swapaxes(Q_target, 1, 2)
            Q_target = np.mean(Q_target)

            #Compute Y
            Y = discnt_rewards + (self.gamma * Q_target)
            #Compute Q value estimate of critic
            Q = self.critic(states, action)

            Q = np.squeeze(Q)
            #Q = np.swapaxes(Q, 1, 2)
            Q = np.mean(Q)

            #Calculate TD errors
            TD_errors = (Y - Q)

            print(f'@ learn_ - TD_errors:\n{TD_errors}')

            # Backpropagation
            grads1 = tape1.gradient(a_loss, self.actor.trainable_variables)
            #grads2 = tape2.gradient(c_loss, self.critic.trainable_variables)

            trainable_vars = self.critic.trainable_variables
            grads2 = tape2.gradient(c_loss, trainable_vars)

        self.a_opt.apply_gradients(zip(grads1, self.actor.trainable_variables))
        #self.c_opt.apply_gradients(zip(grads2, self.critic.trainable_variables))

        return a_loss, c_loss, TD_errors

    ## TODO:: GradientTape 에서 loss 계산해서 저장하기

    def train_ppo_with_GT(self, states, actions, rewards, next_states):
        action_probs, old_action_probs = [], []
        logger.error(f'@ train_ppo_with_GT - states:\n{states}')
        #states = states.reshape(-1, 4)
        #logger.error(f'@ train_ppo_with_GT - after reshape states:\n{states}')

        #actors = self.model_actor(states)
        #logger.error(f'### actors shape : {actors.shape}')

        # Reshape to add an additional dimension
        states = np.expand_dims(states, axis=0)  # Now shape is (1, 4, 9)
        action_probs = self.model_actor.predict(states)

        #action_probs = tf.nn.softmax(self.model_actor(states))#.numpy()
        logger.error(f'### action_probs :\n{action_probs}')

        #ex_states = np.pad(states, (0, 12 - states.shape[0]), 'constant')
        #ex_states = np.expand_dims(ex_states, axis=0)
        values = self.model_critic(states)
        logger.error(f'### values shape : {values.shape}')

        #old_action_probs.append(action_probs)
        old_action_probs = action_probs

        with tf.GradientTape(persistent=True) as tape:
            logger.error(f'################ with GT ################\n')
            #expand_states = np.pad(states, (0, 12 - states.shape[0]), 'constant')
            #expand_states = np.expand_dims(expand_states, axis=0)  # 배치 차원 추가

            #values = tf.squeeze(self.model_critic(expand_states))
            #values = self.model_critic(expand_states)
            #logger.error(f'### values shape : {values.shape}')

            #expand_next_states = np.pad(next_states, (0, 12 - next_states.shape[0]), 'constant')
            #expand_next_states = np.expand_dims(expand_next_states, axis=0)  # 배치 차원 추가



            #next_states = next_states.reshape(-1, 4)
            #next_values = tf.squeeze(self.model_critic(expand_next_states))
            # Add an extra dimension to next_states
            next_states = np.expand_dims(next_states, axis=0)
            next_values = self.model_critic(next_states)
            logger.error(f'### next_values shape : {next_values.shape}')

            advantages = self.make_gae()
            advantages = advantages[:6]
            #advantages = tf.reshape(advantages, [6, 1])
            advantages = tf.reshape(advantages, (-1, 1))

            advantages = tf.cast(advantages, tf.float32)
            logger.error(f'### advantages shape : {np.shape(advantages)}')
            '''
            mean_advantages = [
                sum(sum(inner_list) for inner_list in outer_list) / (len(outer_list) * len(outer_list[0])) for
                outer_list in advantages]
            logger.error(f"mean_advantages shape: {np.shape(mean_advantages)}")
            '''

            #curr_P = self.model_actor(states, training=True)
            #loss = self.compute_loss(old_action_probs, curr_P, actions, advantages)

            # Check if the number of elements matches the target shape
            logger.error(f'### tf.size(values): {tf.size(values)}')

            if tf.size(values) == 100:
                logger.error("Non-Error: The tensor size is [100].")
                values = tf.reshape(values, [100])
                
                returns = advantages + values
                logger.error(f'### returns : {np.shape(returns)}')  #(100, 3, 4)
            else:
                logger.error("Error: The tensor cannot be reshaped to the target shape [100].")
                returns = advantages

            logger.error(f'### np.shape(returns) : {np.shape(returns)}')  #(100, 3, 4)
            '''
            mean_returns = [
                sum(sum(inner_list) for inner_list in outer_list) / (len(outer_list) * len(outer_list[0])) for
                outer_list in returns]
            logger.info("mean_returns shape:", np.shape(mean_returns))
            '''

            #probs = tf.nn.softmax(self.model_actor(states))

            loss, total_loss = self.ppo_loss_with_GT(old_action_probs, states, actions, advantages, values, returns, rewards)
            #loss = self.ppo_loss(advantages, old_action_probs)
            #loss = self.ppo_loss_with_GT(old_action_probs, action_probs, values, old_values, next_values, actions, rewards)
            logger.error(f'after ppo_loss_with_GT ***** total_loss : {total_loss} *****')

            batch_old_prediction = self.get_old_prediction(states)
            # advantages = tf.cast(advantages, tf.float32)
            batch_old_prediction = tf.cast(batch_old_prediction, tf.float32)
            batch_old_prediction = tf.reshape(batch_old_prediction, (-1, 6))

            batch_a_final = np.zeros(shape=(1,))
            batch_a_final[:] = 1



            print("States shape:", states.shape)
            print("Advantages shape:", advantages.shape)
            print("Batch old prediction shape:", batch_old_prediction.shape)

            '''feed_dict = {
                self.advantage_input: advantages,
                self.old_prediction: batch_old_prediction,
                # other placeholders
            }'''
        advantage_np = advantages.eval(session=self.sess)
        batch_old_prediction_np = batch_old_prediction.eval(session=self.sess)

        init = tf.global_variables_initializer()

        if not isinstance(loss, tf.Tensor):
            raise TypeError(f"Expected loss to be a tensor, but got {type(loss)}")


        # Calculate gradients
        grads = tape.gradient(loss, self.model_actor.trainable_variables + self.model_critic.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model_actor.trainable_variables + self.model_critic.trainable_variables))

        batch_v = self.get_v(states)
        batch_advantage = rewards - batch_v
        # Convert dtype if necessary
        batch_advantage = batch_advantage.astype(np.float32)
        # Reshape to match placeholder shape
        batch_advantage = np.reshape(batch_advantage, (-1, 1))

        batch_old_prediction = self.get_old_prediction(states)

        #advantages = tf.cast(advantages, tf.float32)
        batch_old_prediction = tf.cast(batch_old_prediction, tf.float32)

        #batch_a_final = np.zeros(shape=(len(actions), self.action_n))
        #batch_a_final[:, actions.flatten()] = 1
        batch_a_final = np.zeros(shape=(1,))
        batch_a_final[:] = 1

        logger.error(f"Advantages shape: {advantages.shape}")
        logger.error(f"Advantages dtype: {advantages.dtype}")

        logger.error(f"batch_advantage shape: {batch_advantage.shape}")
        logger.error(f"batch_advantage dtype: {batch_advantage.dtype}")

        batch_old_prediction = tf.reshape(batch_old_prediction, (-1, 6))
        logger.error(f"batch_old_prediction shape: {batch_old_prediction.shape}")
        logger.error(f"batch_old_prediction dtype: {batch_old_prediction.dtype}")
        '''
        feed_dict = {
            self.advantage_input: batch_advantage,
            self.old_prediction: batch_old_prediction,
            # other placeholders
        }
        '''
        #self.model_actor.fit(x=[states, advantages, batch_old_prediction], y=batch_a_final, epochs=2, steps_per_epoch=32, verbose=0)
        #self.model_actor.fit(x=[states, batch_advantage, batch_old_prediction], y=batch_a_final, epochs=2, steps_per_epoch=6, verbose=0)
        #self.model_critic.fit(x=states, y=rewards, epochs=2, steps_per_epoch=6, verbose=0)
        #self.memory.clear()
        #self.update_target_network()

        return total_loss

    def train_network_(self):
        n = self.memory.cnt_samples
        discounted_r = []
        if self.memory.batch_done[-1]:
            v = 0
        else:
            v = self.get_v(self.memory.batch_s_[-1])
        for r in self.memory.batch_r[::-1]:
            v = r + self.GAMMA * v
            discounted_r.append(v)
        discounted_r.reverse()

        batch_s, batch_a, batch_discounted_r = np.vstack(self.memory.batch_s), \
                     np.vstack(self.memory.batch_a), \
                     np.vstack(discounted_r)

        batch_v = self.get_v(batch_s)
        batch_advantage = batch_discounted_r - batch_v
        batch_old_prediction = self.get_old_prediction(batch_s)

        batch_a_final = np.zeros(shape=(len(batch_a), self.n_actions))
        batch_a_final[:, batch_a.flatten()] = 1
        # print(batch_s.shape, batch_advantage.shape, batch_old_prediction.shape, batch_a_final.shape)
        self.model_actor.fit(x=[batch_s, batch_advantage, batch_old_prediction], y=batch_a_final, verbose=0)
        self.model_critic.fit(x=batch_s, y=batch_discounted_r, epochs=2, verbose=0)
        self.memory.clear()
        self.update_target_network()

    def train_network(self, states, actions, rewards, next_states):
        """
        1. Get GAE rewards
        2. reshape batches s,a,gae_r baches
        3. get value of state
        4. calc advantage
        5. get "old" predicition (of target network)
        6. fit actor and critic network
        7. soft update target "old" network
        :return:
        """

        #logger.error(f"@ train_network - GAE_CALCULATED_Q : {self.memory.GAE_CALCULATED_Q}")
        #if not self.memory.GAE_CALCULATED_Q:
        batch_gae_r = self.make_gae()
        batch_gae_r = batch_gae_r[:6]
        #batch_gae_r = tf.reshape(batch_gae_r, [6, 1])
        #batch_gae_r = tf.cast(batch_gae_r, tf.float32)
        logger.error(f'### batch_gae_r shape : {np.shape(batch_gae_r)}')
        #states,actions,rewards,gae_r,next_states,dones = self.memory.get_batch_each(self.TRAINING_BATCH_SIZE)

        # create np array batches for training
        batch_s = np.vstack(states)
        batch_a = np.vstack(actions)
        batch_r = np.vstack(rewards)
        #batch_gae_r = np.vstack(gae_r)
        # get values of states in batch
        batch_v = self.get_v(batch_s)
        logger.error(f'@@ train_network -\nbatch_s :\n{batch_s}\n'
              f'batch_gae_r :\n{batch_gae_r}\n'
              f'batch_v :\n{batch_v}\n')
        # calc advantages. required for actor loss.
        batch_advantage = batch_gae_r - batch_v
        batch_advantage = K.utils.normalize(batch_advantage)
        batch_advantage = torch.mean(batch_advantage, dim=0)
        logger.error(f'@@ train_network - batch_advantage shape:\n{np.shape(batch_advantage)}')
        logger.error(f'@@ train_network - batch_advantage :\n{batch_advantage}')

        # calc old_prediction. Required for actor loss.
        batch_old_prediction = np.zeros(shape=(len(batch_a), 1))
        batch_old_prediction = self.get_old_prediction(batch_s)
        # one-hot the actions. Actions will be the target for actor.
        logger.error(f'@@ train_network -\nlen(batch_a) : {len(batch_a)}\nself.action_n : {self.action_n}')
        #batch_a_final = np.zeros(shape=(len(batch_a), self.action_n))
        batch_a_final = np.zeros(shape=(1, len(batch_a)))
        #batch_a_final = np.zeros(shape=(len(batch_a), 1))
        logger.error(f'@@ train_network -\nbatch_a :\n{batch_a}\nbatch_a.flatten() :\n{batch_a.flatten()}\n')

        #commit training
        batch_s = np.expand_dims(batch_s, axis=0)
        logger.error(f'BEFORE fit : batch_s -\n{batch_s}\n batch_a_final -\n{batch_a_final}')
        batch_advantage = np.reshape(batch_advantage, (-1, 1))
        logger.error(f'BEFORE fit : batch_advantage -\n{batch_advantage}\n')

        batch_old_prediction = np.reshape(batch_old_prediction, (-1, 6))

        batch_advantage_tf = tf.convert_to_tensor(batch_advantage, dtype=tf.float32)
        # PyTorch 텐서를 float32로 변환
        #batch_old_prediction = batch_old_prediction.to(torch.float32)
        #batch_old_prediction_np = batch_old_prediction.numpy()
        batch_old_prediction_tf = tf.convert_to_tensor(batch_old_prediction, dtype=tf.float32)

        logger.error(f'BEFORE fit : batch_old_prediction_tf -\n{batch_old_prediction_tf}\n')

        #batch_a_final[:, batch_a.flatten()] = 1
        batch_a_final = batch_a_final[:1]
        self.model_actor.fit(x=[batch_s, batch_advantage_tf, batch_old_prediction_tf], y=batch_a_final, epochs=1, steps_per_epoch=6, verbose=0)
        self.model_critic.fit(x=batch_s, y=batch_gae_r[0], epochs=1, steps_per_epoch=6, verbose=0)
        #soft update the target network(aka actor_old).

        predicted_actions = self.model_actor.predict(batch_s)
        logger.info(f'@@ train_network -\npredicted_actions :\n{predicted_actions}')
        self.update_target_network()

        batch_old_prediction = batch_old_prediction.flatten()
        batch_predictions = self.model_actor.predict_on_batch(batch_s).flatten()
        logger.info(f'@@ train_network -\nbatch_s :\n{batch_s}\n'
              f'batch_advantage :\n{batch_advantage}\n'
              f'batch_old_prediction :\n{batch_old_prediction}\n'
              f'batch_a_final : \n{batch_a_final}\n'
              f'batch_predictions : \n{batch_predictions}\n')

        loss = self.ppo_loss_new(batch_advantage, batch_old_prediction, batch_predictions)
        '''
        old_predictions = torch.tensor(batch_old_prediction)
        predictions = torch.tensor(batch_predictions)
        advantages = torch.tensor(batch_advantage)
        loss = self.ppo_loss_by_torch(predictions, old_predictions, rewards, advantages)
        logger.info(f'@@ ppo learn - loss: {loss}')
        return loss
        '''

        #loss = self.ppo_loss(batch_advantage, batch_old_prediction)
        logger.error(f'### ppo_loss:{loss}')
        return loss
