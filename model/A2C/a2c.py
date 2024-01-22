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
from keras.optimizers.legacy import Adam

import torch
import math

import logging

logger = logging.getLogger(__name__)

from util.ou_noise import OUNoise
from util.memory_buffer import Memory
from model.A2C.ActorNetwork import ActorNetwork
from model.A2C.CriticNetwork import CriticNetwork


huber_loss = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.SUM)


class A2C:
    def __init__(self, env, sess, act_dim, obs_dim, memory, gamma=0.99, actor_lr=1e-4, critic_lr=1e-5):
        self.env = env
        self.gamma = gamma
        self.discount_factor = 0.8
        self.act_dim = act_dim
        self.eps = np.finfo(np.float32).eps.item()  # Smallest number such that 1.0 + eps != 1.0

        self.noise = OUNoise(act_dim * 2)
        self.a_opt = tf.keras.optimizers.Adam(learning_rate=actor_lr)
        self.c_opt = tf.keras.optimizers.RMSprop(learning_rate=critic_lr)
        self.c2_opt = tf.keras.optimizers.RMSprop(learning_rate=critic_lr)
        self.actor = ActorNetwork(sess, action_dim=act_dim, observation_dim=obs_dim, lr=actor_lr, memory=memory)
        self.critic = CriticNetwork(sess, action_dim=act_dim, observation_dim=obs_dim, lr=critic_lr)
        self.critic2 = CriticNetwork(sess, action_dim=act_dim, observation_dim=obs_dim, lr=critic_lr)

        self.actor_target = ActorNetwork(sess, action_dim=act_dim, observation_dim=obs_dim, lr=actor_lr, memory=memory)
        self.critic_target = CriticNetwork(sess, action_dim=act_dim, observation_dim=obs_dim, lr=critic_lr)

        self.memory = memory
        self.actor.compile(optimizer=Adam(learning_rate=actor_lr))
        self.critic.compile(optimizer=Adam(learning_rate=critic_lr))

    def act(self, state):
        print('@@@ a2c act state:{} shape(state):{} \n'.format(state, np.shape(state)))
        print('@@@ a2c act noise(): {}\n'.format(self.noise.noise()))
        print('@@@ a2c act np.clip: {}\n'.format(np.clip(state + self.noise.noise(), 0, 2)))
        return np.clip(state + self.noise.noise(), 0, 2)


    def act_(self, state):
        prob = self.actor(np.array([state]))
        prob = prob.numpy()
        dist = tfp.distributions.Categorical(probs=prob, dtype=tf.float32)
        action = dist.sample()
        return int(action.numpy()[0])

    def act_noise(self, state, add_noise=True):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).unsqueeze(0).float()
        action = self.actor.get_action(state)
        return action

    def choose_action(self, observation):
        state = tf.convert_to_tensor([observation])
        actions, _ = self.actor.sample_normal(state, reparameterize=False)

    def get_action_(self, state):
        policy = self.actor.predict(state, batch_size=1).flatten()
        return np.random.choice(self.act_dim, 1, p=policy)[0]

    def get_action(self, action_prob):
        # [[확률 형식으로 출력]]
        # [0]을 넣어 줌
        # print("policy = ", policy)
        return np.random.choice(self.act_dim, 1, p = np.squeeze(action_prob))[0]

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

        return actor_loss, critic_loss

    def learn_experience(self, idxs, weights, experiences):
        print(f'@ learn_experience - shape of experiences :\n{np.shape(experiences)}')

        idx = idxs
        weight = weights
        print(f'@ learn_experience - idx :{idx}\nweight:{weight}')

        states = np.zeros((3,6))
        next_states = np.zeros((3,6))

        states[0] = experiences[:(3*2)]
        states[1]= experiences[(3*2):(3*2)*2]
        states[2] = experiences[(3*2)*2:(3*2)*3]

        actions = experiences[(3*2)*3:((3*2)*3)+3]
        rewards = experiences[((3*2)*3)+3:((3*2)*3)+6]
        print(f'@ learn_experience - states :\n{states}\n')
        print(f'@ learn_experience - rewards :\n{rewards}\n')
        next_states[0] = experiences[(3*2)*4:(3*2)*5]
        next_states[1] = experiences[(3*2)*5:(3*2)*6]
        next_states[2] = experiences[(3*2)*6:(3*2)*7]

        print(f'@@ learn_experience - states :\n{states}\nactions :\n{actions}\nrewards :\n{rewards}\nnext_states :\n{next_states}\n')

        # print(actions.shape)
        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        next_action, log_pis_next = self.actor.forward(next_states)

        Q_target_next = self.critic_target(next_states, next_action)

        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (
                    self.gamma * (Q_target_next * log_pis_next.mean(1)))

        # Compute critic loss
        p = self.actor(states, training=True)
        Q = self.critic(states, actions)
        Q_2 = self.critic2(states, actions)
        td_error = Q_targets - Q  # ,reduction="none"
        td_error2 = Q_targets - Q_2
        c_loss = 0.5 * np.mean((pow(td_error, 2) * weight))
        c2_loss = 0.5 * np.mean((pow(td_error2, 2) * weight))
        prios = abs(((td_error + td_error2) / 2.0 + 1e-5))

        print(f'@ learn_experience - idx: {idx}, prios: {prios}\nprios.numpy(): {prios.numpy()}')
        #self.memory.update_priorities(idx, prios.numpy())
        self.memory.update_priority(idx, prios)

        '''
        actionprob_history = []
        with tf.GradientTape() as tape:
            action_prob = self.actor(states, training=True)
            critic = self.critic(states, training=True)
            print(f'@ learn_experience - action_prob:{action_prob}\ncritic:{critic}\n')
        '''

        a_loss = self.actor.actor_loss(p, actions, prios)

        actor_loss = np.mean(a_loss)
        critic_loss = abs((c_loss + c2_loss) / 2.0)
        print(f'##### learn_experience - actor_loss:{actor_loss}\n##### critic_loss:{critic_loss}\n')
        return actor_loss, critic_loss

    def learn_(self, states, actions, next_states, discnt_rewards):
        with tf.GradientTape() as tape1, tf.GradientTape() as tape2:
            action_prob = self.actor(states, training=True)
            critic = self.critic(states, training=True)

            print(f'@ learn_ - action_prob:\n{action_prob}\ncritic:\n{critic}')
            print(f'@ learn_ - action_prob[0][0]:\n{action_prob[0][0]}')
            action = self.get_action(action_prob[0][0])

            print(f'@ learn_ - action: {action}')

            # rewards 를 discounted factor로 다시 계산.
            returns = []
            discounted_sum = 0
            for r in discnt_rewards[::-1]:
                discounted_sum = r + self.discount_factor * discounted_sum
                returns.insert(0, discounted_sum)
            # Normalize
            returns = np.array(returns)
            returns = (returns - np.mean(returns)) / (np.std(returns) + self.eps)
            returns = returns.tolist()
            print(f'returns :\n{returns}')

            next_critic = self.critic(next_states, training=True)

            print(f'@ learn_ - next_critic:\n{next_critic}\ndiscnt_rewards:\n{discnt_rewards}')

            advantage = returns[0] - critic[0,0]
            #advantage = discnt_rewards + self.gamma * next_critic - critic
            #advantage = discnt_rewards - self.discount_factor * next_critic
            #advantage = discnt_rewards - critic[0,0]
            print(f'advantage :\n{advantage}')
            # [ [prob, prob, ... ] ]형식으로 입력이 들어옴
            print(f'action_prob[0][0, action] :\n{action_prob[0][0, action]}')
            a_loss = -tf.math.log(action_prob[0][0, action]) * advantage
            c_loss = huber_loss(critic[0,0], discnt_rewards)
            print(f'actor loss :\n{a_loss}')
            print(f'critic loss :\n{c_loss}')

            #Compute the Q value estimate of the target network
            Q_target = self.critic_target(next_states, self.actor_target(next_states))
            #Compute Y
            Y = discnt_rewards + (self.gamma * Q_target)
            #Compute Q value estimate of critic
            Q = self.critic(states, action)
            #Calculate TD errors
            TD_errors = (Y - Q)

            print(f'@ learn_ - TD_errors:\n{TD_errors}')

            # Backpropagation
            grads1 = tape1.gradient(a_loss, self.actor.trainable_variables)
            grads2 = tape2.gradient(c_loss, self.critic.trainable_variables)

            self.a_opt.apply_gradients(zip(grads1, self.actor.trainable_variables))
            self.c_opt.apply_gradients(zip(grads2, self.critic.trainable_variables))

        return a_loss, c_loss, TD_errors

    def learn(self, states, actions, discnt_rewards):
        discnt_rewards = tf.reshape(discnt_rewards, (len(discnt_rewards),))

        with tf.GradientTape() as tape1, tf.GradientTape() as tape2, tf.GradientTape() as tape3:
            p = self.actor(states, training=True)
            v = self.critic(states, training=True)
            v2 = self.critic2(states, training=True)
            '''
            value = self.critic(states, training=True)
            current_policy_actions, log_probs = self.actor.sample_normal(states, reparameterize=False)
            print(f'@@ learn @@\nlog_probs:\n{log_probs}')
            log_probs = tf.squeeze(log_probs, 1)
            critic_value = tf.squeeze(
                tf.math.minimum(v, v2), 1)
            value_target = critic_value - log_probs
            value_loss = 0.5 * keras.losses.MSE(value, value_target)
            print(f'value_loss:\n{value_loss}')
            '''
            print(f'@@ learn @@\ndiscnt_rewards:\n{discnt_rewards}')
            print(f'p:\n{p}')
            print(f'v:\n{v}')
            print(f'v2:\n{v2}')
            p = tf.reshape(p, (15,))
            v = tf.reshape(v, (5,))
            v2 = tf.reshape(v2, (5,))
            print(f'reshape p:\n{p}')
            print(f'reshape v:\n{v}')
            print(f'reshape v2:\n{v2}')

            td1 = abs(discnt_rewards - v)
            td2 = abs(discnt_rewards - v2)
            print(f'td1:{td1}\n')
            print(f'td2:{td2}\n')

            td = abs((td1+td2)/2.0 + 1e-5)
            print(f'td:{td}\n')
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
            c2_loss = 0.5 * kls.mean_squared_error(discnt_rewards, v2)
            print(f'## a_loss:{a_loss}\n## c_loss:{c_loss}\n## c2_loss:{c2_loss}')
            actor_loss = np.mean(a_loss)
            critic_loss = abs((c_loss + c2_loss) / 2.0)
            #returns = get_expected_return(rewards, gamma)
            #loss = self.compute_loss(actions, values, discnt_rewards)

        grads1 = tape1.gradient(a_loss, self.actor.trainable_variables)
        grads2 = tape2.gradient(c_loss, self.critic.trainable_variables)
        grads3 = tape3.gradient(c2_loss, self.critic2.trainable_variables)
        self.a_opt.apply_gradients(zip(grads1, self.actor.trainable_variables))
        self.c_opt.apply_gradients(zip(grads2, self.critic.trainable_variables))
        self.c2_opt.apply_gradients(zip(grads3, self.critic2.trainable_variables))

        print(f'##### actor_loss:{actor_loss}\n##### critic_loss:{critic_loss}\n')
        return actor_loss, critic_loss
