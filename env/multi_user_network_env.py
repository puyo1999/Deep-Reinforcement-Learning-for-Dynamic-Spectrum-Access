import numpy as np
import random
import sys
import os
from py_lab.lib import logger
logger = logger.get_logger(__name__)

"""
TIME_SLOTS = 1000
NUM_CHANNELS = 8
NUM_USERS = 20
ATTEMPT_PROB = 0.6
GAMMA = 0.90
"""

class env_network:
    def __init__(self,num_users,num_channels,attempt_prob):
        self.ATTEMPT_PROB = attempt_prob
        self.NUM_USERS = num_users
        self.NUM_CHANNELS = num_channels
        self.REWARD = 1

        self.DOUBLE_REWARD = 2
        self.channel_pattern = np.zeros([self.NUM_CHANNELS+1],np.int32)

        #self.channel_alloc_freq =

        self.action_space = np.arange(self.NUM_CHANNELS+1)
        self.users_action = np.zeros([self.NUM_USERS],np.int32)
        self.users_observation = np.zeros([self.NUM_USERS],np.int32)
        #logger.info('@@ EnvNetwork init - users_observation:{}'.format(self.users_observation))
    def reset(self):
        pass
    def sample(self):
        x = np.random.choice(self.action_space, size=self.NUM_USERS)
        logger.info(f'##### MuEnv @@@@@ sample - action_space:{self.action_space}')
        #logger.info('@@ EnvNetwork - action by sampling:{}'.format(x))
        return x
    def step(self,action):
        logger.info(f'action.size: {action.size}, NUM_USER: {self.NUM_USERS}')
        assert (action.size) == self.NUM_USERS, "action and user should have same dim action.size:\n{}".format(action.size)
        channel_alloc_frequency = np.zeros([self.NUM_CHANNELS + 1],np.int32)  #0 for no chnnel access
        obs = []
        reward = np.zeros([self.NUM_USERS])
        j = 0
        for each in action:
            prob = random.uniform(0,1)
            logger.info(f'step - prob: {prob}')
            if prob <= self.ATTEMPT_PROB:
                self.users_action[j] = each  # action
                channel_alloc_frequency[each] += 1
            j += 1
        logger.info(f'step - original channel_alloc_frequency: {channel_alloc_frequency}')

        for i in range(1,len(channel_alloc_frequency)):
            if channel_alloc_frequency[i] > 1:
                channel_alloc_frequency[i] = 0

        #logger.info channel_alloc_frequency
        logger.info(f'step - modified channel_alloc_frequency: {channel_alloc_frequency}')
        for i in range(len(action)):
            
            self.users_observation[i] = channel_alloc_frequency[self.users_action[i]]
            logger.info('step - observation[{}] : {}'.format(i, self.users_observation[i]))
            if self.users_action[i] == 0:   # accessing no channel
                self.users_observation[i] = 0
            if self.users_observation[i] == 1:
                logger.info('step - reward : 1 !!!')
                reward[i] = 1
            elif self.users_observation[i] == 2:
                reward[i] = 2
            obs.append((self.users_observation[i],reward[i]))
        residual_channel_capacity = channel_alloc_frequency[1:]
        residual_channel_capacity = 1-residual_channel_capacity
        logger.info(f'step - residual_channel_capacity: {residual_channel_capacity}')
        obs.append(residual_channel_capacity)
        logger.critical(f'step - obs: {obs}')
        return obs

    def close(self):
        pass



