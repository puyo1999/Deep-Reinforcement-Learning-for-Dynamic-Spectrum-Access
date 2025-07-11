import logging
import yaml

import os
print(os.getcwd())

with open('D:/Research Files/DRLforDSA/Deep-Reinforcement-Learning-for-Dynamic-Spectrum-Access/config/config.yaml') as f:
    config = yaml.safe_load(f)

logger = logging.getLogger(__name__)

log_level_info = config['log_level_info']

if log_level_info:
    logger.setLevel(logging.DEBUG) # Debug 레벨 이상의 로그를 Handler들에게 전달해야 합니다.
else:
    logger.setLevel(logging.WARNING) # Warning 레벨 이상의 로그를 Handler들에게 전달해야 합니다.

#formatter = logging.Formatter('%(asctime)s:%(module)s:%(levelname)s:%(message)s', '%Y-%m-%d %H:%M:%S')
formatter = logging.Formatter('%(module)s:%(levelname)s:%(lineno)d:%(message)s')


# INFO 레벨 이상의 로그를 콘솔에 출력하는 Handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# DEBUG 레벨 이상의 로그를 `debug.log`에 출력하는 Handler
file_debug_handler = logging.FileHandler('debug.log')
file_debug_handler.setLevel(logging.DEBUG)
file_debug_handler.setFormatter(formatter)
logger.addHandler(file_debug_handler)

# ERROR 레벨 이상의 로그를 `error.log`에 출력하는 Handler
file_error_handler = logging.FileHandler('error.log')
file_error_handler.setLevel(logging.ERROR)
file_error_handler.setFormatter(formatter)
logger.addHandler(file_error_handler)

def get_logger(__name__):
    '''Create a child logger specific to a module'''
    return logging.getLogger(__name__)
