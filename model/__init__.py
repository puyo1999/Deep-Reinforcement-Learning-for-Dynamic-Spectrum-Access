# The agent module

from torch.utils.tensorboard import SummaryWriter
import numpy as np
import os
import pandas as pd
import pydash as ps
import torch
import warnings

from py_lab.lib import logger, util
logger = logger.get_logger(__name__)

class Agent:
    '''
    Agent abstraction; implements the API to interface with Env in SLM Lab
    Contains algorithm, memory, body
    '''

    def __init__(self, spec, global_nets=None):
        self.spec = spec
        self.agent_spec = spec['agent'][0]  # idx 0 for single-agent
        self.name = self.agent_spec['name']
        assert not ps.is_list(global_nets), f'single agent global_nets must be a dict, got {global_nets}'
        # set components
        #self.body = body
        #body.agent = self
        #MemoryClass = getattr(memory, ps.get(self.agent_spec, 'memory.name'))
        #self.body.memory = MemoryClass(self.agent_spec['memory'], self.body)
        #AlgorithmClass = getattr(algorithm, ps.get(self.agent_spec, 'algorithm.name'))
        #self.algorithm = AlgorithmClass(self, global_nets)

        logger.info(util.self_desc(self))

class Body:
    def __init__(self, env, spec, aeb=(0,0,0)):
        self.agent = None
        self.env = env
        self.spec = spec

        self.a, self.e, self.b = self.aeb = aeb

        # variables set during init_algorithm_params
        self.explore_var = np.nan  # action exploration: epsilon or tau
        self.entropy_coef = np.nan  # entropy for exploration

        # debugging/logging variables, set in train or loss function
        self.loss = np.nan
        self.mean_entropy = np.nan
        self.mean_grad_norm = np.nan

        # total_reward_ma from eval for model checkpoint saves
        self.best_total_reward_ma = -np.inf
        self.total_reward_ma = np.nan

        # dataframes to track data for analysis.analyze_session
        # track training data per episode
        self.train_df = pd.DataFrame(columns=[
            'epi', 't', 'wall_t', 'opt_step', 'frame', 'fps', 'total_reward', 'total_reward_ma', 'loss', 'lr',
            'explore_var', 'entropy_coef', 'entropy', 'grad_norm'])

        # in train@ mode, override from saved train_df if exists
        if util.in_train_lab_mode() and self.spec['meta']['resume']:
            train_df_filepath = util.get_session_df_path(self.spec, 'train')
            if os.path.exists(train_df_filepath):
                self.train_df = util.read(train_df_filepath)
                self.env.clock.load(self.train_df)

        # track eval data within run_eval. the same as train_df except for reward
        if self.spec['meta']['rigorous_eval']:
            self.eval_df = self.train_df.copy()
        else:
            self.eval_df = self.train_df

        # the specific agent-env interface variables for a body
        self.observation_space = self.env.hdmi_observation
        self.action_space = self.env.action_space
        #self.state_dim = self.observable_dim['state']
        self.action_dim = self.env.hdmi_action
        #self.is_discrete = self.env.is_discrete
        # set the ActionPD class for sampling action
        #self.action_type = policy_util.get_action_type(self.action_space)
        #self.action_pdtype = ps.get(spec, f'agent.{self.a}.algorithm.action_pdtype')
        #if self.action_pdtype in (None, 'default'):
            #self.action_pdtype = policy_util.ACTION_PDS[self.action_type][0]
        #self.ActionPD = policy_util.get_action_pd_cls(self.action_pdtype, self.action_type)