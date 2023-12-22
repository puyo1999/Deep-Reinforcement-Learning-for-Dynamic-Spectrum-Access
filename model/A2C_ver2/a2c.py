"""
a2c.py
"""
__author__ = "py81.kim@gmail.com"
__credits__ = "https://github.com/puyo1999"

import numpy as np
import tensorflow as tf
from model.A2C_ver2.ActorNetwork import ActorNetwork
from model.A2C_ver2.CriticNetwork import CriticNetwork
from keras import optimizers, losses

class A2C_ver2:
    def __init__(self):
        # hyper parameters
        self.lr = 0.001
        self.lr2 = 0.005
        self.df = 0.99
        self.en = 0.001

        self.actor_model = ActorNetwork()
        self.actor_opt = optimizers.Adam(learning_rate=self.lr)

        self.critic_model = CriticNetwork()
        self.critic_opt = optimizers.Adam(learning_rate=self.lr2)

    def actor_loss(self, states, actions, advantages):
        policy = self.actor_model(tf.convert_to_tensor(np.vstack(states), dtype=tf.float32))
        print(f"@ actor_loss - policy:{policy}")
        print(f"@ actor_loss - actions:{actions}")

        # SparseCategoricalCrossentropy = ce loss with not one-hot encoded output
        # from_logits = True  =>  cross_entropy with soft_max

        entropy = losses.categorical_crossentropy(policy, policy, from_logits=False)
        #ce_loss = losses.SparseCategoricalCrossentropy(from_logits=False)
        ce_loss = losses.CategoricalCrossentropy(from_logits=False)
        # policy_loss = ce_loss(actions, policy, sample_weight=np.array(advantages))  # same way
        log_pi = ce_loss(actions, policy)
        policy_loss = log_pi * np.array(advantages)
        policy_loss = tf.reduce_mean(policy_loss)

        return policy_loss - self.en * entropy

    def critic_loss(self, states, rewards, dones):
        last_state = states[-1]
        if dones == True :
            reward_sum = 0
        else :
            reward_sum = self.critic_model(tf.convert_to_tensor(last_state[None, :], dtype=tf.float32))
        discounted_rewards = []
        for reward in rewards[::-1]:
            reward_sum = reward + self.df * reward_sum
            discounted_rewards.append(reward_sum)
        discounted_rewards.reverse()
        discounted_rewards = tf.convert_to_tensor(np.array(discounted_rewards)[:, None], dtype=tf.float32)
        values = self.critic_model(tf.convert_to_tensor(np.vstack(states), dtype=tf.float32))
        error = tf.square(values - discounted_rewards)*0.5
        error = tf.reduce_mean(error)
        return error

    def learn(self, states, actions, rewards, next_states, dones):
        critic_variable = self.critic_model.trainable_variables
        with tf.GradientTape() as tape_critic:
            tape_critic.watch(critic_variable)
            critic_loss = self.critic_loss(states, rewards, dones)

        # gradient descent will be applied automatically
        critic_grads = tape_critic.gradient(critic_loss, critic_variable)
        self.critic_opt.apply_gradients(zip(critic_grads, critic_variable))


        advantages = self.compute_advantages(states, rewards, dones)
        actor_variable = self.actor_model.trainable_variables
        with tf.GradientTape() as tape:
            tape.watch(actor_variable)
            actor_loss = self.actor_loss(states, actions, advantages)

        actor_grads = tape.gradient(actor_loss, actor_variable)
        self.actor_opt.apply_gradients(zip(actor_grads, actor_variable))

        # self.train_loss(tf.reduce_mean(actor_loss))
        # self.train_loss_c(tf.reduce_mean(critic_loss))
        self.train_loss = tf.reduce_mean(actor_loss)
        self.train_loss_c = tf.reduce_mean(critic_loss)
        return (actor_loss + critic_loss)

    def compute_advantages(self, states, rewards, dones):
        last_state = states[-1]
        if dones == True:
            reward_sum = 0
        else:
            reward_sum = self.critic_model(tf.convert_to_tensor(last_state[None, :], dtype=tf.float32))
        discounted_rewards = []
        for reward in rewards[::-1]:
            reward_sum = reward + self.df * reward_sum
            discounted_rewards.append(reward_sum)
        discounted_rewards.reverse()
        values = self.critic_model(tf.convert_to_tensor(np.vstack(states), dtype=tf.float32))
        advantages = discounted_rewards - values
        return advantages

    def get_action(self, state):
        policy = self.actor_model.predict(state, batch_size=1).flatten()
        return np.random.choice(self.action_size, 1, p=policy)[0]