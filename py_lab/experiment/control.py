from py_lab.lib import util
from py_lab.lib.logger import logger
#from py_lab.experiment import analysis
from config.spec import spec_util
from env import make_env
from model import Agent, Body
import pydash as ps
import os
import subprocess

def make_agent_env(spec, global_nets=None):
    '''Helper to create agent and env given spec'''
    print(f'@ make_agent_env')
    env = make_env(spec)
    body = Body(env, spec)
    agent = Agent(spec, global_nets=global_nets)
    return agent, env


def mp_run_session(spec, global_nets, mp_dict):
    '''Wrap for multiprocessing with shared variable'''
    session = Session(spec, global_nets)
    metrics = session.run()
    mp_dict[session.index] = metrics

class Session:

    def __init__(self, spec, global_nets=None):
        self.spec = spec
        self.index = self.spec['meta']['session']
        util.set_random_seed(self.spec)
        util.set_cuda_id(self.spec)
        #util.set_logger(self.spec, logger, 'session')
        spec_util.save(spec, unit='session')

        self.agent, self.env = make_agent_env(self.spec, global_nets)
        if ps.get(self.spec, 'meta.rigorous_eval'):
            with util.ctx_lab_mode('eval'):
                self.eval_env = make_env(self.spec)
        else:
            self.eval_env = self.env
        print(f'util.self_desc(self) : {util.self_desc(self)} ')

    def run_rl(self):
        logger.info(f'Running RL loop for trial {self.spec["meta"]["trial"]} session {self.index}')
        state = self.env.reset()
        done = False

        os.system('python ../train.py')

    def close(self):
        '''Close session and clean up. Save agent, close env.'''
        self.agent.close()
        self.env.close()
        self.eval_env.close()
        #torch.cuda.empty_cache()
        logger.info(f'Session {self.index} done')

    def run(self):
        self.run_rl()
        #metrics = analysis.analyze_session(self.spec, self.agent.body.eval_df, 'eval')
        #self.agent.body.log_metrics(metrics['scalar'], 'eval')
        #self.close()
        #return metrics

    def get_agent_env(self):
        return self.env