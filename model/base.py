from abc import ABC, abstractmethod
import numpy as np
from py_lab.lib import logger, util

logger = logger.get_logger(__name__)

class Algorithm(ABC):
    '''Abstract Algorithm class to define the API methods'''

    def __init__(self, agent, global_nets=None):
        '''
        @param {*} agent is the container for algorithm and related components, and interfaces with env.
        '''
        self.agent = agent
        self.algorithm_spec = agent.agent_spec['algorithm']
        self.name = self.algorithm_spec['name']
        self.memory_spec = agent.agent_spec['memory']
        self.net_spec = agent.agent_spec['net']
        self.body = self.agent.body
        self.init_algorithm_params()
        self.init_nets(global_nets)
        logger.info(util.self_desc(self, omit=['algorithm_spec', 'name', 'memory_spec', 'net_spec', 'body']))
