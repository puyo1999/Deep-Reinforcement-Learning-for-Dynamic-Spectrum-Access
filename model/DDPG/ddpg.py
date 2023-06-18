"""
ddpg.py
"""
__author__ = "py81.kim@gmail.com"
__credits__ = "https://github.com/puyo1999"

import tensorflow as tf
import keras
import numpy as np
from util.ou_noise import OUNoise
from util.memory_buffer import Memory
from model.DDPG.ddpg_actor import Actor
from model.DDPG.ddpg_critic import Critic

class DDPG:
    def __init__(self, env, obs_dim, act_dim, memory, steps, gamma=0.99, buffer_size=1e6, batch_size=32, tau=0.001, name='DDPG'):
        self.env = env
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.steps = steps
        self.gamma = gamma
        self.buf_size = buffer_size
        self.batch_size = batch_size
        self.step_cnt = 0
        self.tau = tau

        self.noise = OUNoise(act_dim*2)
        self.memory = memory

        self.actor = Actor(obs_dim, act_dim)
        self.actor_target = Actor(obs_dim, act_dim)
        self.critic = Critic(obs_dim, act_dim)
        self.critic_target = Critic(obs_dim, act_dim)

        self.soft_target_update(tau=1)  #

        self.loss_fn = keras.losses.MeanSquaredError()
        self.actor_optimizer = keras.optimizers.Adam(lr=0.0001)
        self.critic_optimizer = keras.optimizers.Adam(lr=0.001)

    def store_transition(self, s, a, r, s_):
        print('shape of s:{} a:{}'.format(np.shape(s), np.shape(a)))
        print('StoreTransition - s:{} a:{} r:{} s_:{}'.format(s,a,r,s_))
        transition = np.hstack([list(s[0]),list(s[1]), list(s[2]), list(np.r_[a, r]), list(s_[0]), list(s_[1]),list(s_[2])])
        print('StoreTransition - transition:{}'.format(transition))
        self.memory.store(transition)
        self.step_cnt += 1

    def learn(self, ob, ac, next_ob, reward, done = False):
        next_ac = tf.clip_by_value(self.actor_target(next_ob), self.env.action_space.low, self.env.action_space.high)

        q_target = self.critic_target([next_ob, next_ac])
        y = reward + (1-done) * self.gamma * q_target

        with tf.GradientTape() as tape_c:
            q = self.critic([ob, ac])
            q_loss = self.loss_fn(y, q)
        grads_c = tape_c.gradient(q_loss, self.critic.trainable_weights)

        with tf.GradientTape() as tape_a:
            a = self.actor([ob, ac])
            q_for_grad = -tf.reduce_mean(self.actor([ob,a]))
        grads_a = tape_a.gradient(q_for_grad, self.actor.trainable_weights)

        self.critic_optimizer.apply_gradient(grads_c)
        self.actor_optimizer.apply_gradient(grads_a)

    def policy_action(self, s):
        """ Use the actor to predict value
        """
        #return self.actor.predict(s)[0]
        return self.actor.predict(s)[0]

    def get_action(self, ob, train_mode=True):
        if train_mode:
            print('@@@ get_action ob:{} shape(ob):{} \n'.format(ob, np.shape(ob)))
            print('@@@ get_action noise():{}\n'.format(self.noise.noise()))
            return np.clip(ob + self.noise.noise(), 0, 2)
            action = self.actor(ob[np.newaxis])[0]
            #return np.clip(ob + self.noise.noise(), -self.act_dim, +self.act_dim)
            #return np.clip(self.actor(ob[np.newaxis])[0] + self.noise.noise(), 0, 2)
            #return np.clip(self.actor(ob) + self.noise.noise(), self.env.action_space.low, self.env.action_space.high)
        else:
            return np.clip(self.actor(ob[np.newaxis])[0], self.env.action_space.low, self.env.action_space.high)

    def soft_target_update(self, tau=None):
        tau = self.tau if tau is None else tau
        actor_tmp = tau * np.array(self.actor.get_weights()) + (1. - tau)*np.array(self.actor_target.get_weights())
        critic_tmp = tau * np.array(self.critic.get_weights()) + (1. - tau)*np.array(self.critic_target.get_weights())
        self.actor_target.set_weights(actor_tmp)
        self.critic_target.set_weights(critic_tmp)
