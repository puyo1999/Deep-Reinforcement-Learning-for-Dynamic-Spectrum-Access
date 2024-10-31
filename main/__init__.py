import os

os.environ['PY_ENV'] = os.environ.get('PY_ENV') or 'development'
ROOT_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), '../../Deep-Reinforcement-Learning-for-Dynamic-Spectrum-Access'))

print(f'ROOT_DIR: {ROOT_DIR}')

# valid lab_mode in SLM Lab
EVAL_MODES = ('enjoy', 'eval')
TRAIN_MODES = ('search', 'train', 'dev')