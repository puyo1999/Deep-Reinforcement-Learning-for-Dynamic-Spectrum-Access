# the environment module

from env.mbr_detection_env import mbr_env
from env.multi_user_network_env import env_network
from py_lab.lib import logger
logger = logger.get_logger(__name__)


def make_env(spec):
    logger.info(f"num_hdmi :{spec['env'][0]['num_hdmi']}")
    env = mbr_env(spec['env'][0]['num_hdmi'], spec['env'][0]['num_dt'],
                  spec['env'][0]['num_mf'], spec['env'][0]['num_cs'],
                  spec['env'][0]['attempt_prob'])
    return env