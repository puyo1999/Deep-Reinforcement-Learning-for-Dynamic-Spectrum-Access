import yaml

with open('D:/Research Files/DRLforDSA/Deep-Reinforcement-Learning-for-Dynamic-Spectrum-Access/config/config.yaml') as f:
    config = yaml.safe_load(f)

TIME_SLOTS = config['time_slots']
NUM_CHANNELS = config['num_channels']                               # Total number of channels
NUM_CS = config['num_cs']                                       # Total number of codesets
NUM_HDMI = config['num_hdmi']
NUM_USERS = config['num_users']                                 # Total number of users
ATTEMPT_PROB = config['attempt_prob']                               # attempt probability of ALOHA based  models
BATCH_SIZE = config['batch_size']
PRETRAIN_LEN = config['pretrain_length']
TRAIN_FREQ = config['training_frequency']
TO_TRAIN = config['to_train']
STEP_SIZE = config['step_size']
ACTOR_LR = config['actor_lr']
CRITIC_LR = config['critic_lr']

MINIMUM_REPLAY_MEMORY = 32

MEMORY_SIZE = config['memory_size']

batch_size = BATCH_SIZE                         # Num of batches to train at each time_slot
pretrain_length = PRETRAIN_LEN            # this is done to fill the deque up to batch size before training
hidden_size = 128                       # Number of hidden neurons
learning_rate = 1e-4                    # learning rate

explore_start = .02                     # initial exploration rate
explore_stop = .01                      # final exploration rate
decay_rate = config['decay_rate']                      # rate of exponential decay of exploration

GAMMA = config['gamma']
REWARD_DISCOUNT = config['reward_discount']
TYPE = config['type']              # DL algorithm
WITH_PER = config['with_per']
GRAPH_DRAWING = config['graph_drawing']

dic_env_conf = {
    "NUM_CS": 5,
    "NUM_CHANNELS": NUM_CHANNELS,
    "NUM_HDMI": NUM_HDMI,
    "NUM_USERS": NUM_USERS,
    "NUM_DT": 6, # device type : STB, OTT, BD, Game
    "NUM_MF": 10, # manufacturer
}