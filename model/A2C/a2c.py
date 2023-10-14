"""
a2c.py
"""
__author__ = "py81.kim@gmail.com"
__credits__ = "https://github.com/puyo1999"

import tensorflow as tf
tf.executing_eagerly()

import keras
import numpy as np
import tensorflow_probability as tfp
import keras.losses as kls

import logging

logger = logging.getLogger(__name__)

from util.ou_noise import OUNoise
from util.memory_buffer import Memory
from model.A2C.ActorNetwork import ActorNetwork
from model.A2C.CriticNetwork import CriticNetwork

huber_loss = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.SUM)


class A2C:
    def __init__(self, sess, act_dim, obs_dim, memory, gamma=0.99, actor_lr=1e-4, critic_lr=5e-4):
        self.gamma = gamma
        self.noise = OUNoise(act_dim * 2)
        self.a_opt = tf.keras.optimizers.Adam(learning_rate=actor_lr)
        self.c_opt = tf.keras.optimizers.RMSprop(learning_rate=critic_lr)
        self.actor = ActorNetwork(sess, action_dim=act_dim, observation_dim=obs_dim, lr=actor_lr, memory=memory)
        self.critic = CriticNetwork(sess, action_dim=act_dim, observation_dim=obs_dim, lr=critic_lr)

    def act(self, state):
        print('@@@ a2c act state:{} shape(state):{} \n'.format(state, np.shape(state)))
        print('@@@ a2c act noise(): {}\n'.format(self.noise.noise()))
        print('@@@ a2c act np.clip: {}\n'.format(np.clip(state + self.noise.noise(), 0, 2)))
        return np.clip(state + self.noise.noise(), 0, 2)
        '''
        prob = self.actor(np.array([state]))
        prob = prob.numpy()
        dist = tfp.distributions.Categorical(probs=prob, dtype=tf.float32)
        action = dist.sample()
        return int(action.numpy()[0])
        '''

    def actor_loss(self, probs, actions, td):

        probability = []
        log_probability = []

        probs_unpacked = tf.unstack(probs)
        actions_unpacked = tf.unstack(actions)

        for pb, a in zip(probs_unpacked, actions_unpacked):
        #for pb, a in zip(probs, actions):
            dist = tfp.distributions.Categorical(probs=pb, dtype=tf.float32)
            log_prob = dist.log_prob(a)
            prob = dist.prob(a)
            probability.append(prob)
            log_probability.append(log_prob)

        # print(probability)
        # print(log_probability)
        p_loss = []
        e_loss = []
        td = td.numpy()
        print(f'@@@ td:\n{td}')

        for pb, t, lpb in zip(probability, td, log_probability):
            t = tf.constant(t)
            policy_loss = tf.math.multiply(lpb, t)
            entropy_loss = tf.math.negative(tf.math.multiply(pb, lpb))
            p_loss.append(policy_loss)
            e_loss.append(entropy_loss)
        p_loss = tf.stack(p_loss)
        e_loss = tf.stack(e_loss)
        p_loss = tf.reduce_mean(p_loss)
        e_loss = tf.reduce_mean(e_loss)
        # print(p_loss)
        # print(e_loss)
        loss = -p_loss - 0.0001 * e_loss
        # print(loss)
        return loss

    def compute_loss(self, action_probs, values, returns):
        """Computes the combined Actor-Critic loss."""
        advantage = returns - values

        action_log_probs = tf.math.log(action_probs)
        actor_loss = -tf.math.reduce_sum(action_log_probs * advantage)

        critic_loss = huber_loss(values, returns)

        return actor_loss + critic_loss

    def learn(self, states, actions, discnt_rewards):
        discnt_rewards = tf.reshape(discnt_rewards, (len(discnt_rewards),))

        with tf.GradientTape() as tape1, tf.GradientTape() as tape2:
            p = self.actor(states, training=True)
            v = self.critic(states, training=True)
            print(f'@@ learn @@\ndiscnt_rewards:\n{discnt_rewards}')
            print(f'p:\n{p}')
            print(f'v:\n{v}')

            p = tf.reshape(p, (15,))
            v = tf.reshape(v, (5,))
            print(f'reshape p:\n{p}')
            print(f'reshape v:\n{v}')
            td = tf.math.subtract(discnt_rewards, v)

            td = td.numpy()
            '''
            advantage = discnt_rewards - v
            action_log_probs = tf.math.log(p)
            print(f'action_log_probs:\n{action_log_probs}')
            print(f'advantage:\n{advantage}')
            a_loss = -tf.math.reduce_sum(action_log_probs * advantage)
            '''

            #a_loss = self.actor_loss(p, actions, td)
            a_loss = self.actor.actor_loss(p, actions, td)
            c_loss = 0.5 * kls.mean_squared_error(discnt_rewards, v)

            #returns = get_expected_return(rewards, gamma)
            #loss = self.compute_loss(actions, values, discnt_rewards)

        grads1 = tape1.gradient(a_loss, self.actor.trainable_variables)
        grads2 = tape2.gradient(c_loss, self.critic.trainable_variables)
        self.a_opt.apply_gradients(zip(grads1, self.actor.trainable_variables))
        self.c_opt.apply_gradients(zip(grads2, self.critic.trainable_variables))


        return a_loss, c_loss


