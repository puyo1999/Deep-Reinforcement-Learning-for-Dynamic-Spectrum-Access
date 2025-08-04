import keras.src.losses
import tensorflow as tf
tf.executing_eagerly()
tf.config.run_functions_eagerly(True)

tf.data.experimental.enable_debug_mode()

### A2C or A2C_ver2
# Enable eager execution
tf.compat.v1.enable_eager_execution()

from config.setup import *

from env.multi_user_network_env import env_network
from env.mbr_detection_env import mbr_env
from keras.losses import huber
from util.memory_buffer import Memory
from util.prioritized_memory import PerMemory
from util.ppo_memory import PPOMemory
from util.parser import Parser
from util.utils import get_states_user, get_actions_user, get_rewards_user, get_next_states_user
from util.utils import state_generator
from util.utils import draw_res2, draw_multi_algorithm, draw_losses
from model.A2C.a2c import A2C
from model.A2C_ver2.a2c import A2C_ver2
from model.DDPG.ddpg import DDPG
from model.drqn import QNetwork
from model.dqn import DQNetwork
from model.ddqn import DDQN
from model.PPO.ppo import PPO

from model.base import Algorithm

import numpy as np
import random
import sys
from collections import deque
from multiprocessing import Queue, Lock
import tensorflow.compat.v1 as tf

import yaml

from py_lab.lib import logger
from py_lab.experiment import control
from py_lab.experiment.control import Session
from config.spec import spec_util
from util import path_util

with open('./config/config.yaml') as f:
    config = yaml.safe_load(f)

args = None
args = Parser.parse_args(args)

args.env = config['env']


load_spec = path_util.read('file/ppo_mbr/ppo_mbr_spec.json')
print(f'## load_spec : ***** {load_spec} *****')

NUM_HDMI = dic_env_conf["NUM_HDMI"]
NUM_DT = dic_env_conf["NUM_DT"]
NUM_MF = dic_env_conf["NUM_MF"]
NUM_CS = dic_env_conf["NUM_CS"]
#NUM_CS = load_spec['env'][0]['num_cs']
print(f'NUM_CS : {NUM_CS}')

#TIME_SLOTS = 100000                            # number of time-slots to run simulation
#TIME_SLOTS = 1000
#TIME_SLOTS = args.time_slot
TIME_SLOTS = config['time_slots']


if args.env == "network":
    NUM_CHANNELS = dic_env_conf["NUM_CHANNELS"]
elif args.env == "mbr":
    NUM_CHANNELS = dic_env_conf["NUM_CS"]
else:
    NUM_CHANNELS = config['num_channels']                       # Total number of channels

if args.env == "network":
    NUM_USERS = dic_env_conf["NUM_USERS"]
elif args.env == "mbr":
    NUM_USERS = dic_env_conf["NUM_HDMI"]
#NUM_USERS = config['num_users']                                 # Total number of users

ATTEMPT_PROB = config['attempt_prob']                               # attempt probability of ALOHA based  models
BATCH_SIZE = config['batch_size']
PRETRAIN_LEN = config['pretrain_length']
STEP_SIZE = config['step_size']
actor_lr = config['actor_lr']
critic_lr = config['critic_lr']

MINIMUM_REPLAY_MEMORY = 32

#memory_size = 1000                      #size of experience replay deque
args.memory_size = config['memory_size']

batch_size = BATCH_SIZE                         # Num of batches to train at each time_slot
pretrain_length = PRETRAIN_LEN            # this is done to fill the deque up to batch size before training
hidden_size = 128                       # Number of hidden neurons
learning_rate = 1e-4                    # learning rate
'''
actor_lr = 1e-4
critic_lr = 5e-4             # critic learning rate
'''
explore_start = .02                     # initial exploration rate
explore_stop = .01                      # final exploration rate
decay_rate = config['decay_rate']                      # rate of exponential decay of exploration
#gaemma = .99                             # discount  factor

from py_lab.lib import logger
logger = logger.get_logger(__name__)


logger.info('########## config.yaml start ##########')
logger.info(config)
logger.info('########## config.yaml end ##########')
#logger.info(fconfig)

args.gamma = config['gamma']
args.reward_discount = config['reward_discount']
args.type = config['type']              # DL algorithm
#args.type = load_spec['agent'][0]['net']['type']
print(f'args.type : {args.type}')
args.with_per = config['with_per']
args.with_ere = config['with_ere']
args.graph_drawing = config['graph_drawing']

if args.type == "A2C" or args.type == "A2C_ver2" or args.type == "PPO" or args.type == "DDQN":
    from tensorflow.python.framework.ops import enable_eager_execution
    enable_eager_execution()
else:
    tf.disable_v2_behavior()

if args.graph_drawing:
    data1 = np.genfromtxt("a2c_scores.txt", delimiter=",")
    logger.info(f"@ shape of data1 : {np.shape(data1)}")

    data2 = np.load("a2c_scores.npy")
    logger.info(f"@ shape of data2 : {np.shape(data2)}")

    data3 = np.load("drqn_scores.npy")
    data4 = np.load("ddqn_scores.npy")
    logger.info(f"@ shape of data3 : {np.shape(data3)}")
    logger.info(f"@ shape of data4 : {np.shape(data4)}")
    draw_multi_algorithm(data1, data2, data3, data4)

    exit()

'''
batch_size = args.batch_size
pretrain_length = batch_size  # this is done to fill the deque up to batch size before training
hidden_size = args.hidden  # Number of hidden neurons
learning_rate = args.lr  # learning rate
explore_start = args.explore_start  # initial exploration rate
explore_stop = args.explore_stop  # final exploration rate
decay_rate = args.decay_rate  # rate of exponential decay of exploration
gamma = args.gamma  # discount  factor
logger.info(f"+++ batch_size : ", batch_size)
logger.info(f"+++ pretrain_length : ", pretrain_length)
logger.info(f"+++ hidden_size : ", hidden_size)
logger.info(f"+++ learning_rate : ", learning_rate)
logger.info(f"+++ explore_start : ", explore_start)
logger.info(f"+++ explore_stop : ", explore_stop)
logger.info(f"+++ decay_rate : ", decay_rate)
logger.info(f"+++ gamma : ", gamma)
'''

step_size = STEP_SIZE
#step_size = 1 + 2 + 2                   #length of history sequence for each datapoint  in batch
if args.env == "mbr":
    # NUM_CHANNELS = NUM_CS = 5
    state_size = 2 * (NUM_CHANNELS + 1)     #length of input (2 * k + 2)   :k = NUM_CHANNELS
else:
    state_size = 2 * (NUM_CHANNELS + 1)     #length of input (2 * k + 2)   :k = NUM_CHANNELS
action_size = NUM_CHANNELS + 1            #length of output  (k+1)
alpha = 0                               #co-operative fairness constant
beta = 1                                #Annealing constant for Monte - Carlo

# reseting default tensorflow computational graph
tf.reset_default_graph()
sess = tf.Session()

session_instance = Session(load_spec)

# initializing the environment
if args.env == "network":
    env = env_network(NUM_USERS,NUM_CHANNELS,ATTEMPT_PROB)
elif args.env == "mbr":
    env = mbr_env(NUM_HDMI, NUM_DT, NUM_MF, NUM_CS, ATTEMPT_PROB)
    #env = session_instance.get_agent_env()
else:
    env = env_network(NUM_USERS,NUM_CHANNELS,ATTEMPT_PROB)

logger.info(f'##### ENV setup #####\n')
# to sample random actions for each user
action = env.sample()
logger.info(f'### action:{action}')

# mbr action 은 6개 코드셋 중에서 1개 골라서 hdmi1, 2, 3, 4 에 적용
#

obs = env.step(action)
logger.info(f'### obs :\n{obs}\n')

state = state_generator(action, obs)
reward = [i[1] for i in obs[:NUM_USERS]]
logger.info(f'### Before init Deep Q Network #####\n')

logger.info(f'### state :\n{state}\n')
logger.info(f'### reward :\n{reward}\n')

#this is experience replay buffer(deque) from which each batch will be sampled and fed to the neural network for training
if args.with_per:
    #memory = PerMemory(mem_size=args.memory_size, feature_size=NUM_USERS*2, prior=True)
    memory = PerMemory(mem_size=args.memory_size, feature_size=2 * (NUM_CHANNELS + 1), prior=True)
else:
    '''
    if args.type == "PPO":
        memory = PPOMemory()
    else:
    '''
    memory = Memory(max_size=args.memory_size)

#initializing deep Q network model
if args.type == "DQN":
    logger.info(f"##### init DQN #####")
    mainQN = DQNetwork(name='DQN',hidden_size=hidden_size,learning_rate=learning_rate,step_size=step_size,state_size=state_size,action_size=action_size, memory=memory)
elif args.type == "DRQN":
    logger.info(f"##### init DRQN #####")
    mainQN = QNetwork(name='DRQN',hidden_size=hidden_size,learning_rate=learning_rate,step_size=step_size,state_size=state_size,action_size=action_size, memory=memory)
elif args.type == "A2C":
    logger.info(f"##### A2C #####")
    #a2c = A2C(env=env, sess=sess, act_dim=action_size, obs_dim=NUM_USERS * 2, actor_lr=actor_lr, critic_lr=critic_lr, memory=memory)
    if args.env == "network":
        a2c = A2C(env=env, sess=sess, act_dim=action_size, obs_dim=state_size, memory=memory, actor_lr=actor_lr, critic_lr=critic_lr, prior=args.with_per)
    elif args.env == "mbr":
        a2c = A2C(env=env, sess=sess, act_dim=action_size, obs_dim=NUM_HDMI, actor_lr=actor_lr, critic_lr=critic_lr, memory=memory, prior=args.with_per)
elif args.type == "PPO":                                                                                                 
    #ppo = PPO(env=env, memory=memory, action_n=action_size, state_dim=state_size, training_batch_size=batch_size)
    if args.env == "network":
        ppo = PPO(env=env, sess=sess, memory=memory, action_n=action_size, state_dim=state_size, training_batch_size=batch_size)
    elif args.env == "mbr":
        # action_size = 6, NUM_HDMI = 4
        ppo = PPO(env=env, sess=sess, memory=memory, action_n=action_size, state_dim=NUM_HDMI, training_batch_size=batch_size)

elif args.type == "A2C_ver2":
    logger.info(f"##### A2C_ver2 #####")
    a2c = A2C_ver2(name='A2C_ver2')

elif args.type == "DDQN":
    if args.env == "network":
        mainQN = DDQN(name='DDQN', sess=sess, feature_size=NUM_USERS*2, learning_rate=learning_rate, state_size=state_size, actions=range(action_size), action_size=action_size, step_size=step_size, prior=args.with_per, memory=memory, gamma=args.gamma)
    elif args.env == "mbr":
        # action_size = 6, NUM_HDMI = 4, NUM_USER = 3
        mainQN = DDQN(name='DDQN', sess=sess, feature_size=NUM_USERS*2, learning_rate=learning_rate, state_size=NUM_HDMI, actions=range(action_size), action_size=action_size, step_size=step_size, prior=args.with_per, memory=memory, gamma=args.gamma)

elif args.type == "DDPG":
    #mainQN = DDPG(name= 'main', env=env, obs_dim=env.action_space.shape, act_dim=env.action_space.shape, memory=memory, steps=10000)
    mainQN = DDPG(name='DDPG', env=env, obs_dim=NUM_USERS*2, act_dim=action_size, memory=memory, steps=10000)
    #mainQN = DDPG(name='main', env=env, obs_dim=NUM_USERS, act_dim=action_size, memory=memory, steps=10000)


replay_memory = deque(maxlen=100)

#this is our input buffer which will be used for  predicting next Q-values
if args.type == "A2C" or args.type == "A2C_ver2" or args.type == "PPO":
    #history_input = deque(maxlen=action_size)
    history_input = deque(maxlen=NUM_HDMI)
else:
    history_input = deque(maxlen=step_size)

### TODO :
# 불확실성 추정 함수
'''
def predict_with_uncertainty(f, x, n_iter=100):
    result = np.zeros((n_iter,) + x.shape)
    for i in range(n_iter):
        result[i] = f(x, training=True)
    prediction = result.mean(axis=0)
    uncertainty = result.std(axis=0)
    return prediction, uncertainty
'''
###
def compute_importance(state, reward, gamma=0.99):
    # 바로 다음 리워드를 감가율만으로 예측?
    cur_tot_len = len(replay_memory)
    logger.info(f'cur_tot_len : {cur_tot_len}')

    # 세번째 값들만 추출
    reward_values = [t[2] for t in replay_memory]
    logger.info(f'reward_values : {reward_values}')
    logger.info(f'sum(reward_values[0]) : {sum(reward_values[0])}')
    logger.info(f'len(reward_values[0]) : {len(reward_values[0])}')
    # 평균 계산
    avg_reward = sum(reward_values[0]) / len(reward_values[0])

    print(f'avg_reward : {avg_reward}')  # Output: 3.0

    q_value = avg_reward * gamma
    next_q_value = np.max(avg_reward)
    td_target = avg_reward + gamma * next_q_value
    td_error = abs(td_target - q_value)
    return td_error

step = 0
start_train = False

##############################################
############### pretrain loop ################
##############################################
logger.info(f'############### pretrain loop start ################')
for ii in range(pretrain_length*step_size*5):
    done = False
    action = env.sample()
    obs = env.step(action)      # obs is a list of tuple with [[(ACK,REW) for each user] ,CHANNEL_RESIDUAL_CAPACITY_VECTOR]
    next_state = state_generator(action, obs)
    reward = [i[1] for i in obs[:NUM_USERS]]

    logger.info(f'@@@@@ Pretrain step: {step} @@@@@')

    logger.info(f"@ shape of state : {np.shape(state)}")

    logger.info(f'state : {state}\n')
    logger.info(f'action : {action}\n')
    logger.info(f'reward : {reward}\n')
    logger.info(f'next_state : {next_state}\n')

    if args.with_per:
        #memory.add(td_error, (state, action, reward, next_state))
        if args.type == "A2C":
            a2c.actor.store_transition(state, action, reward, next_state)
        elif args.type == "A2C_ver2":
            transition = np.hstack(
                [list(state[0]), list(state[1]), list(state[2]), list(np.r_[action, reward]), list(next_state[0]), list(next_state[1]), list(next_state[2])])

            #memory.store(transition)
            memory.add(transition, 1.0)
        elif args.type == "PPO":
            ppo.store_transition_per2(state, action, reward, next_state)
        else:
            mainQN.store_transition(state, action, reward, next_state)
    #elif args.with_ere:
        #a2c.add()
    else:
        if args.type == "PPO":
            # TODO:: reward 에 가중치 곱해보는데, 동일한 코드셋, 채널을 선택한 사용자를 분모로 넣어서,
            # TODO:: 그 에 비례해서 discount factor 계산해보자
            memory.add((state, action, reward, next_state))
            memory.store_each(state, action, reward, next_state, False)
            #ppo.store_transition(state, action, reward, next_state)
            if not ppo.memory.GAE_CALCULATED_Q:
                ppo.make_gae()
        elif args.type == "A2C":
            memory.add((state, action, reward, next_state))
            memory.store_each(state, action, reward, next_state, False)
            if args.with_per:
                if not memory.GAE_CALCULATED_Q:
                    a2c.make_gae()

        else:
            memory.add((state, action, reward, next_state))
            #memory.store_each(state, action, reward, next_state, False)
            #a2c.make_gae()

    #replay_memory.append((state, np.argmax(action), reward, next_state))
    replay_memory.append((state, action, reward, next_state))

    ## TODO:: importance sampling 기법 적용 가능할지 작업 추가 해보자.
    ## TODO:: 이걸로 네트워크 자원 할당 퍼포먼스 관련 국내 저널 한편 가능할지는 실험 필요
    #compute_importance(state, reward, 0.99)

    state = next_state
    history_input.append(state)
    logger.info(f'@@ Pretrain step:{step} after store_transition')

    if step >= args.memory_size:
        if not start_train:
            start_train = True
            if args.type == "A2C":
                a2c.memory.GAE_CALCULATED_Q = True
            elif args.type == "PPO":
                ppo.memory.GAE_CALCULATED_Q = True
            logger.info(f'@@ Now start_train:{start_train}')
            break
        #if args.type == 'A2C':
            #actor.learn(batch, batch_size, feature_size=NUM_USERS*2)
    step += 1

interval = 1       # debug interval

################################
## TODO:: compute_importance 적용을 해서 replay memory 에 저장할 수 있을까?

# After above pretraining, importance calculation method is applied
#
#replay_memory.append()
#for ii in range(replay_memory.len()):

#def compute_importance(state, action, reward, next_state):
###


################################

# saver object to save the checkpoints of the DQN to disk
if args.type != "A2C" and args.type != "A2C_ver2" and args.type != "PPO" and args.type != "DDQN":
    saver = tf.train.Saver()
else:
    saver = tf.train.Checkpoint()

#initializing the session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.7

#initializing the session

if args.type != "A2C" and args.type != "A2C_ver2" and args.type != "PPO" and args.type != "DDQN":
    sess = tf.Session(config=config)
else:
    sess = tf.Session()

#initialing all the tensorflow variables
if args.type != "A2C" and args.type != "A2C_ver2" and args.type != "PPO" and args.type != "DDQN":
    logger.critical(f'sess.run(tf.global_variables_initializer()')
    sess.run(tf.global_variables_initializer())

############################### train_ddpg ###############################
def train_ddpg(replay_memory, batch_size):
    global_steps = 0
    epochs = 0
    rewards_list = []
    minibatch = random.sample(replay_memory, batch_size)
    for index, sample in enumerate(minibatch):
        cur_state, action, reward, next_state = sample

        #mainQN.learn(cur_state, action, next_state, reward, done=False)

    if args.with_per:
        idx, weights, transition = memory.sample(batch_size)
    else:
        #cur_state, action, reward, next_state = sample
        transition = memory.sample(batch_size, step_size)
    logger.info(f': {transition}')
    a = np.split(transition, 6)
    ob = a[0]
    ac = a[3]
    reward_ = a[4]
    next_ob = a[5]
    mainQN.learn(ob, ac, next_ob, reward_, done=False)

    #mainQN.learn(*map(lambda x: np.stack(x).astype('float32'), np.transpose(transition)))
    mainQN.soft_target_update()

    mainQN.actor.save_weights("./ddpg_actor/actor", overwrite=True)
############################### train_ddpg ###############################

############################### train_ddqn ###############################
def train_ddqn(replay_memory, batch_size):
    minibatch = random.sample(replay_memory, batch_size)
    for index, sample in enumerate(minibatch):
        cur_state, action, reward, next_state = sample

    if args.with_per:
        idx, weights, batches = memory.sample(batch_size)
    else:
        #cur_state, action, reward, next_state = sample
        batches = memory.sample(batch_size, step_size)

    #for state, action, reward, next_state, done in minibatch:
    target = mainQN.model.predict(state)[0]
    if time_step == TIME_SLOTS:
        target[0][action] = reward
    else:
        # a = self.model.predict(next_state)[0]
        t = mainQN.target_model.predict(next_state)[0]
        #target[0][action] = reward + mainQN.gamma * np.amax(t)
        target[0][action] = reward + mainQN.gamma * np.argmax(t)
    mainQN.model.fit(state, target, epochs=1, verbose=0)
    if mainQN.epsilon > mainQN.epsilon_min:
        mainQN.epsilon *= mainQN.epsilon_decay

    if args.with_per:
        p = np.sum(np.abs(t - target), axis=1)
        memory.update(idx, p)
############################### train_ddqn ###############################

# list of total rewards
total_rewards = []

# list mean reward
all_means = []
means = [0]

# cumulative reward
cum_r = [0]

# cumulative collision
cum_collision = [0]

# loss list
loss_list = [0]

#mse_loss = mse()
actor_losses = []
critic_losses = []

max_reward = 10000
episode_reward = 0
max_timestep = 5

def is_target(x):
    return x > max_reward

################################################################################
########                      main simulation loop                      ########
lock = Lock()

s_queue = Queue()
a_queue = Queue()
n_s_queue = Queue()
r_queue = Queue()

def preprocess1(states, actions, next_states, rewards, gamma, s_queue, a_queue, n_s_queue, r_queue):
    discnt_rewards = []
    sum_reward = 0
    rewards.reverse()
    for r in rewards:
      sum_reward = r + gamma * sum_reward
      discnt_rewards.append(sum_reward)
    discnt_rewards.reverse()
    states = np.array(states, dtype=np.float32)
    actions = np.array(actions, dtype=np.int32)
    next_states = np.array(next_states, dtype=np.float32)
    discnt_rewards = np.array(discnt_rewards, dtype=np.float32)
    #exp = np.array([states, actions, discnt_rewards],dtype=np.object)
    lock.acquire()

    s_queue.put(states)
    a_queue.put(actions)
    n_s_queue.put(next_states)
    r_queue.put(discnt_rewards)

    logger.info(f's_queue : {s_queue}')
    logger.info(f'a_queue : {a_queue}')
    logger.info(f'n_s_queue : {n_s_queue}')
    logger.info(f'r_queue : {r_queue}')

    lock.release()

def preprocess2(s_queue, a_queue, n_s_queue, r_queue):
    lock.acquire()

    states = []
    while not s_queue.empty():
        states.append(s_queue.get())
    '''
    while not s_queue.empty():
        item = s_queue.get()
        if item is not None:  # None 타입 체크
            states.append(item)
    '''
    actions = []
    while not a_queue.empty():
        actions.append(a_queue.get())

    next_states = []
    while not n_s_queue.empty():
        next_states.append(n_s_queue.get())

    dis_rewards = []
    while not r_queue.empty():
        dis_rewards.append(r_queue.get())

    lock.release()

    logger.info(f'@ preprocess2 - states : {states}\n'
                    f'actions : {states}\n'
                    f'next_states : {states}\n'
                    f'dis_rewards : {dis_rewards}')
    '''
    if states:
        state_batch = np.concatenate(states, axis=0)
    else:
        # 원하는 초기 shape로 생성
        states = np.zeros((4, 9), dtype=np.float32)
        state_batch = np.concatenate(states, axis=0)
    '''

    if states:
        state_batch = np.concatenate(states, axis=0)
    else:
        logger.warning("🚨 preprocess2: states가 비어 있습니다. 초기화된 배열을 반환합니다.")
        states = np.zeros((4, 9), dtype=np.float32)
        state_batch = np.concatenate(*(states,), axis=0)

    # 3) actions 분기
    if actions:
        action_batch = np.concatenate(actions, axis=None)
    else:
        # 예: 배치 크기 4, action은 int이므로 zeros(4,)
        action_batch = np.zeros((4,), dtype=np.int32)

    # 4) next_states 분기
    if next_states:
        next_state_batch = np.concatenate(next_states, axis=0)
    else:
        next_state_batch = np.zeros((4 * 9,), dtype=np.float32)

    # 5) rewards 분기
    if dis_rewards:
        reward_batch = np.concatenate(dis_rewards, axis=None)
    else:
        reward_batch = np.zeros((4,), dtype=np.float32)

    #state_batch = np.concatenate(*(states,), axis=0)
    logger.info(f'@ preprocess2 - state_batch : {state_batch}')
    #action_batch = np.concatenate(*(actions,), axis=None)
    #next_state_batch = np.concatenate(*(next_states,), axis=0)
    #reward_batch = np.concatenate(*(dis_rewards,), axis=None)
    #exp = np.transpose(exp)

    if state_batch.size == 72 and state_batch.shape == (8, 9):
        flat = state_batch.flatten()
        state_batch = flat[:36].reshape(9, 4)

    if next_state_batch.size == 72 and next_state_batch.shape == (8, 9):
        flat = next_state_batch.flatten()
        next_state_batch = flat[:36].reshape(9, 4)

    if reward_batch.size == 8 and reward_batch.shape == (8,):
        reward_batch = reward_batch[:4]

    return state_batch, action_batch, next_state_batch, reward_batch

actor_losses = []
critic_losses = []
temp_step_loss_check_count = 0
# max_reward = 0
# for episode in range(EPISODES):
for time_step in range(TIME_SLOTS):
    logger.info(f'##### main simulation loop START - time_step : {time_step} #####')
    # changing beta at every 50 time-slots
    if time_step % 50 == 0:
        if time_step < 5000:
            logger.info(f"***** every 50 time slots : beta decreasing *****")
            beta -= 0.001
    #curent exploration probability
    explore_p = explore_stop + (explore_start - explore_stop) * np.exp(-decay_rate*time_step)
    logger.info(f'***** explore_p:{explore_p}')

    # Exploration
    if explore_p > np.random.rand():
        #random action sampling
        action = [0] * action_size
        action = env.sample()
        logger.info(f"+++++ Explored +++++")
        logger.info(f"+++++ random sampled action : {action}")
        
    # Exploitation
    else:
        logger.info(f"----- Exploited -----")
        #initializing action vector

        # Taking optimal action (Exploitation)
        if args.type == "A2C" or args.type == "A2C_ver2" or args.type == "DDPG" or args.type == "PPO":
            #action = np.zeros([NUM_USERS], dtype=np.int32)
            if args.env == "network":
                action = np.zeros([NUM_USERS], dtype=object)
            elif args.env == "mbr":
                action = np.zeros([NUM_HDMI], dtype=object)

        else:
            #action = np.zeros((NUM_USERS,6),)
            #action = np.zeros([NUM_USERS], dtype=object)
            action = np.zeros([NUM_USERS], dtype=np.int32)

        logger.info(f'@@@ check state: {state}\n')
        #converting input history into numpy array
        state_vector = np.array(history_input, dtype=object)

        logger.info(f"@@@ Converted @@@\n history_input :\n{history_input}\n --> state_vector :\n{state_vector}\n")
        logger.info(f"/////////////// each_user iter starts ///////////////")

        probs = []
        for each_user in range(NUM_USERS):
            logger.info(f"/////////////// User No:{each_user} ///////////////")

            if args.type == "A2C" or args.type == "DDPG":
                logger.info(f'*** state_vector[uid : {each_user}] :\n {state_vector[:,each_user]}')
                #logger.info(f'*** state[uid{}] :\n {}'.format(each_user, state[each_user]))

                #state[each_user] = np.resize(state[each_user], [1, action_size, state_size])
                #state[each_user] = np.resize(state[each_user], [1, state_size])

                #tempP = np.zeros((21,4), dtype=np.float32)
                #tempP = np.zeros((3, 6), dtype=np.float32)
                tempP = np.array(state_vector[:,each_user])
                tempP = np.array(tempP).astype(np.float32)
                #tempP = tempP[np.newaxis, :]
                logger.info(f'*** tempP :\n{tempP}')
                logger.info(f'*** state[uid : {each_user}] :\n{state[each_user]}')

                #probs = a2c.actor.policy.predict(tempP)[:,each_user]

                #probs = a2c.actor.predict(tempP)[:,each_user]
                #action = a2c.actor.predict(np.expand_dims(state[each_user], axis=0))[0]

                logger.info(f'*** tempP[uid : {each_user}] :\n{tempP[each_user]}')
                prob = a2c.actor.predict(np.expand_dims(tempP[each_user][:4], axis=0))
                #prob = a2c.actor.predict(tempP[each_user])
                logger.info(f'*** predicted prob[uid : {each_user}] : {prob}')

                #prob1 = (1-alpha)*np.exp(beta*probs)
                #prob = prob1/np.sum(np.exp(beta*probs)) + alpha/(NUM_CHANNELS+1)

            elif args.type == "PPO":
                # 1. Eager 모드 강제 활성화
                tf.config.run_functions_eagerly(True)

                #tempP = np.array(state_vector[:, each_user])
                #tempP = np.array(tempP).astype(np.float32)
                logger.info(f'--> iter each_user : {each_user}')
                tempP = np.array(state_vector[:, each_user])
                tempP = np.array(tempP).astype(np.float32)
                logger.critical(f'ppo tempP :\n{tempP}\nshape of tempP: {np.shape(tempP)}') # (None, 4, 9)
                logger.critical(f'ppo tempP[{each_user}] : {tempP[each_user]}')
                logger.info(f'ppo state[{each_user}] : {state[each_user]}')

                #extracted = tempP[:, :6]
                #extracted = np.expand_dims(extracted, axis=0)
                #logger.critical(f'ppo extracted :\n{extracted}') # (6, 4)

                #tempP = tempP.reshape((1, 4, 9))
                #action[each_user] = ppo.choose_action_(tempP)

                prob = ppo.model_actor.predict(np.expand_dims(tempP[:9], axis=0))
                logger.critical(f'*** no sess.run, predict\nprob[{each_user}] :\n{prob}')

                #state_batch = np.expand_dims(tempP, axis=0)  # → (1, 4, 9)
                #logger.critical(f'ppo actor real input :\n{state_batch}')
                #prob = ppo.model_actor.predict(state_batch)

                #prob = ppo.model_actor.predict_on_batch(state_batch)
                #prob = ppo.model_actor.predict(np.expand_dims(tempP[each_user][:9], axis=0))
                #logger.critical(f'*** no sess.run, predict_on_batch\nprob[{each_user}] :\n{prob}')
                '''
                # (0) 초기 1차원 증가
                tempP = np.expand_dims(tempP, axis=0)
                # (1) 모델을 레이어처럼 호출해서 Tensor 얻기
                prob_tensor = ppo.model_actor(tempP)  # tf.Tensor, shape=(1, A)
                # (2) sess.run → ndarray로 반환
                probs = ppo.sess.run(prob_tensor)  # ndarray, shape=(1, A)               
                logger.critical(f'*** with sess.run, predicted probs[{each_user}] :\n{probs}')
                '''
            elif args.type == "A2C_ver2":
                logger.info(f'state[{each_user}] :\n{state[each_user]}')

                policy = a2c.actor_model(tf.convert_to_tensor(state[each_user][None, :], dtype=tf.float32))
                '''choose_action_
                tempP = np.zeros((3, 6), dtype=np.float32)
                tempP = np.array(state_vector[:,each_user])
                tempP = np.array(tempP)
                logger.info(f'tempP[{tempP}]\n shape of tempP: {np.shape(tempP)}')

                policy = a2c.actor_model(tf.convert_to_tensor(tempP, dtype=tf.float32))
                '''
                logger.info(f'policy[{policy}]\n shape of policy: {np.shape(policy)}')


            elif args.type == "DDQN":
                # 1. Eager 모드 강제 활성화
                #tf.config.run_functions_eagerly(True)

                #state[each_user] = np.resize(state[each_user], [1, state_size])
                state[each_user] = np.resize(state[each_user], [1, 9])
                #state[each_user] = state_vector[:,each_user].reshape(step_size,NUM_HDMI)

                prob = mainQN.q_eval_model.predict_on_batch([state[each_user], np.ones((1, 1))])

                # 1) EagerTensor로 예측
                #qs_tensor = mainQN.q_eval_model([state[each_user], np.ones((1, 1))], training=False)

                #prob = K.get_value(qs_tensor)
                # Eager 모드 아님, sess.run 필요
                #Qs = mainQN.q_eval_model([state[each_user], np.ones((1, 1))])

                # 3) 현재 사용 중인 세션 가져오기
                #sess = tf.compat.v1.keras.backend.get_session()
                # 세션에서 직접 실행
                #prob = sess.run(Qs)

                #Qs = mainQN.q_eval_model.predict_on_batch([state[each_user], np.ones((1, 1))])
                #logger.critical(f'DDQN - Qs[{Qs}]\n shape of Qs: {np.shape(Qs)}')

                #prob1 = (1-alpha)*np.exp(beta*Qs)
                #prob = prob1/np.sum(np.exp(beta*Qs)) + alpha/(NUM_CHANNELS+1)
                logger.critical(f'DDQN - prob[{prob}]\n shape of prob: {np.shape(prob)}')

                '''
            elif args.type == "DDPG":                
                state[each_user] = np.resize(state[each_user], [1, state_size])
                Qs = mainQN.actor.predict(state[each_user])
                prob1 = (1-alpha)*np.exp(beta*Qs)
                prob = prob1/np.sum(np.exp(beta*Qs)) + alpha/(NUM_CHANNELS+1)
                logger.info(f'prob1:{prob1} prob:{prob}\n')
                '''
            else:
                #feeding the input-history-sequence of (t-1) slot for each user separately
                if args.env == "network":
                    feed = {mainQN.inputs_:state_vector[:,each_user].reshape(1,step_size,state_size)}
                else:

                    # 1) 원본 상태 벡터 배열 생성
                    state = state_vector[:, each_user].reshape(1, step_size, -1)

                    # 2) 배열 차원 확인 후 패딩 (batch=1, time=step_size, feature=F)
                    #    feed_dict의 마지막 차원을 3개 채울 경우
                    padded_state = np.pad(
                        state,
                        pad_width=((0, 0), (0, 0), (0, 3)),  # (batch, time, feature) 순서
                        mode='constant'
                    )
                    # 3) 패딩된 배열을 placeholder에 매핑
                    feed = {mainQN.inputs_: padded_state}


                #predicting Q-values of state respectively
                Qs = sess.run(mainQN.output, feed_dict=feed)
                #print Qs

                #   Monte-carlo sampling from Q-values  (Boltzmann distribution)
                ##################################################################################
                prob1 = (1-alpha)*np.exp(beta*Qs)

                # Normalizing probabilities of each action  with temperature (beta)
                prob = prob1/np.sum(np.exp(beta*Qs)) + alpha/(NUM_CHANNELS+1)
                #print prob
                logger.critical(f'##### calculated prob[{each_user}] : {prob}\n')

            #   This equation is as given in the paper :
            #   Deep Multi-User Reinforcement Learning for  
            #   Distributed Dynamic Spectrum Access :
            #   @Oshri Naparstek and Kobi Cohen (equation 12)
            ########################################################################################

            #  choosing action with max probability
            if args.type == "A2C":
                #action[each_user] = a2c.act(state[each_user])[0]
                #action[each_user] = a2c.actor.predict(np.expand_dims(state, axis=0))[0]
                '''
                action[each_user] = np.argmax(action, axis=0)
                logger.info(f'##### argmax action[{each_user}]: {action[each_user]}\n')
                '''
                action[each_user] = np.argmax(prob)
                logger.info(f'##### argmax action[{each_user}]: {action[each_user]}\n')

            elif args.type == "PPO":
                #action[each_user] = ppo.choose_action_(state[each_user])
                #from tensorflow.python.keras import backend as K
                #prob_np = K.eval(prob)  # K.eval() == get_session().run()
                #actions[each_user] = np.argmax(prob_np, axis=-1)

                prob = prob[0, each_user, :]
                action[each_user] = np.argmax(prob)

                '''
                probs = probs[0, each_user, :]
                action[each_user] = np.argmax(probs)
                '''
                logger.critical(f'##### ppo max prob - action[{each_user}]: {action[each_user]}\n')

            elif args.type == "A2C_ver2":
                categorized_policy = tf.random.categorical(policy, 1)
                logger.info(f'after categorical : categorized_policy[{categorized_policy}]\n shape of categorized_policy: {np.shape(categorized_policy)}')
                action[each_user] = categorized_policy

                #action[each_user] = tf.squeeze(tf.random.categorical(policy, 1), axis=-1)
                #action_ = tf.random.categorical(policy, 1)

            elif args.type == "DDQN":
                #action[each_user] = mainQN.actor(obs[each_user])
                action[each_user] = np.argmax(prob, axis=1)

                #prob = prob[0, each_user, :]
                #action[each_user] = np.argmax(prob)
                logger.critical(f'##### DDQN max prob - action[{each_user}]: {action[each_user]}\n')


            elif args.type == "DDPG":
                #a = mainQN.policy_action(obs[each_user])
                #a = mainQN.policy_action(state[each_user])
                #logger.info(f'@@ ddpg after policy_action a:{}'.format(a))
                logger.info(f'state[{each_user}]: {state[each_user]}\n')
                action[each_user] = mainQN.get_action(state[each_user], True)[0]
                #action[each_user] = np.argmax(prob, axis=1)[0]

            else:
                action[each_user] = np.argmax(prob,axis=1)
                logger.info(f'##### argmax action[{each_user}]: {action[each_user]}\n')


            #action[each_user] = np.random.choice(action_size, 1, p=policy)[0]
            #action[each_user] = np.argmax(Qs,axis=1)

            if time_step % interval == 0:
                logger.info(f'EachUser:{each_user} Debugging state_vector:\n{state_vector[:,each_user]}')
                if args.type != "A2C" and args.type != "A2C_ver2" and args.type != "DDPG" and args.type != "PPO"\
                        and args.type != "DDQN":
                    logger.info(f'Qs:{Qs}')
                    logger.info(f'prob:{prob}, sum of beta*Qs:{np.sum(np.exp(beta*Qs))}')
                    logger.info(f'End')

    logger.critical(f'@@ all user action : {action}\nshape of action: {np.shape(action)}')

    # taking action as predicted from the q values and receiving the observation from the environment
    #if args.type != "A2C" and args.type != "A2C_ver2" and args.type != "DDPG":
        #state = state_generator(action, obs)

    state = state_generator(action, obs)
    logger.info(f"@@ after generator - state : {state}\n")

    logger.info(f"@@ action :\n{action}")

    #logger.info(f'@@ argmax(action) :\n{np.argmax(action)}')

    obs = env.step(action)           # obs is a list of tuple with [(ACK,REW) for each user ,(CHANNEL_RESIDUAL_CAPACITY_VECTOR)]

    #logger.info(f"@@ obs{obs}\nshape of obs :choose_action_{np.shape(obs)}")
    logger.info(f"@@ obs :\n{obs}")

    logger.info(f"@@ len(obs) :\n{len(obs)}")

    # Generate next state from action and observation 
    next_state = state_generator(action, obs)
    logger.info(f"@@ next_state : {next_state}\n")

    # reward for all users given by environment
    reward = [i[1] for i in obs[:NUM_USERS]]
    logger.info(f'★☆$$ reward : {reward} $$☆★')
    # calculating sum of rewards
    # user들의 reward 합
    sum_r = np.sum(reward)
    logger.info(f'★☆$$ sum_r : {sum_r} $$☆★')

    episode_reward += sum_r
    logger.info(f'★☆$$ episode_reward : {episode_reward} $$☆★')

    if args.env == "network":
        # If NUM_CHANNELS = 2 , total possible reward = 2 , therefore collision = (2 - sum_r) or (NUM_CHANNELS - sum_r)
        collision = NUM_CHANNELS - sum_r
    elif args.env =="mbr":
        hdmi_obs = [i[0] for i in obs[:NUM_USERS]]
        logger.info(f'!! hdmi_obs : {hdmi_obs}')
        #collision = (hdmi_obs == 0)
        collision = int(0 in hdmi_obs)

    #collision = NUM_CHANNELS - sum_r

    logger.info(f'!! collision : {collision}')

    # calculating cumulative collision
    cum_collision.append(cum_collision[-1] + collision)
   
    #############################
    #  for co-operative policy we will give reward-sum to each user who have contributed
    #  to play co-operatively and rest 0
    '''
    for i in range(len(reward)):
        if reward[i] > 0:
            reward[i] = sum_r
    '''
    #############################

    total_rewards.append(sum_r)
    logger.info(f'$$ total_rewards :\n{total_rewards}')

    #if time_step % 100 == 99:
        #if time_step < 5000:
            #logger.info(f"***** every 100 time slots *****")
            # TODO:: measures and stores the mean reward score over that period

    # calculating cumulative reward
    logger.info(f'$$$ Before appending to cum_r : length = {len(cum_r)}, cum_r[-1] = {cum_r[-1]}')
    cum_r.append(cum_r[-1] + sum_r)
    logger.info(f'$$$ After appending to cum_r : length = {len(cum_r)}, cum_r[-1] = {cum_r[-1]}')
    means.append(means[-1] + (cum_r[-1] / (time_step + 1)))
    logger.info(f'$$$$ until time:{time_step} means:{cum_r[-1] / (time_step + 1)}')

    all_means.append(cum_r[-1] / (time_step + 1))
    logger.info(f'$$$$ until time:{time_step} all_means:\n{all_means}')

    if is_target(cum_r[-1]):
    #if is_target(time_step):
        logger.info(f"pykim - Target met (value={cum_r[-1]}) at time_step:{time_step} → breaking loop")
        break


    # add new experiences into the memory buffer as (state, action , reward , next_state) for training
    '''if args.with_per:
        td_error = 1
    else:
        td_error = 0'''

    if args.with_per:
        #memory.add((state, action, reward, next_state), td_error)
        #memory.store((state, action, reward, next_state))
        logger.info(f'## Before store_transition - state:{state} action:{action}')
        if args.type == "A2C":
            a2c.actor.store_transition(state, action, reward, next_state)
            #replay_memory.append((state, action, reward, next_state))
        elif args.type == "A2C_ver2":
            #transition = np.hstack(
                #[list(state[0]), list(state[1]), list(state[2]), list(np.r_[action, reward]), list(next_state[0]), list(next_state[1]), list(next_state[2])])

            transition = np.hstack(
                [(state[0]), (state[1]), (state[2]), (np.r_[action, reward]), (next_state[0]), (next_state[1]), (next_state[2])])

            #memory.store(transition)
            memory.add(transition, 1.0)

        elif args.type != "DRQN" and args.type != "PPO":
            mainQN.store_transition(state, action, reward, next_state)
        else:
            logger.critical(f'pykim - memory.add2 case')
            transition = np.hstack(
                [(state[0]), (state[1]), (state[2]), (np.r_[action, reward]), (next_state[0]), (next_state[1]), (next_state[2])])
            memory.add2(transition)
    else:   # NOT PER
        if args.type == "PPO":
            memory.add((state, action, reward, next_state))
            memory.store_each(state, action, reward, next_state, False)
            #ppo.store_transition(state, action, reward, next_state)
            if not memory.GAE_CALCULATED_Q:
                ppo.make_gae()
        elif args.type == "A2C":
            memory.add((state, action, reward, next_state))
            memory.store_each(state, action, reward, next_state, False)
            if not memory.GAE_CALCULATED_Q:
                a2c.make_gae()
        else:
            memory.add((state, action, reward, next_state))

    state = next_state
    #add new experience to generate input-history sequence for next state
    history_input.append(state)

    logger.info(f'///// BEFORE Training at time_step : {time_step} /////')
    #  Training block starts
    ###################################################################################
    logger.info(f'////////////////////////////////')
    logger.info(f'///// Training block START /////')

    #  sampling a batch from memory buffer for training
    if args.with_per:
        if args.with_ere:
            idx, is_weights, batch = memory.sample_ere(3)
        else:
            idx, is_weights, batch = memory.sample(batch_size)
        batch = np.array(batch)
    else:
        if args.type == "PPO" or args.type == "A2C":
            #batch, gae_r = memory.sample(batch_size, step_size)
            batch = memory.get_batch(batch_size, step_size)
        else:
            batch = memory.sample(batch_size, step_size)

    logger.info(f'///// batch:\n{batch}\n')

    if args.with_per:
        #next_states = np.vstack(batch[3])
        #next_states = get_next_states_user(batch)
        #idx, w, batch = memory.sample(batch_size)

        '''
        for index, sample in enumerate(batch):
            states, actions, rewards, next_states = sample
            logger.info(f'@ after sampling memory.update index:\n{} \n states:\n{}\n'.format(index, states))
        '''
        #idx, is_weights, tmpBatch = memory.sample(5)
        idx, is_weights, tmpBatch = memory.sample(batch_size)
        tmpBatch = np.array(tmpBatch)
        logger.info(f'@@ PER - tmpBatch :\n{np.shape(tmpBatch)}\n')
        logger.info(f'@@ PER - tmpBatch :\n{tmpBatch}\n')

        states = tmpBatch[:, :(NUM_CHANNELS+1)*2]
        states = states[np.newaxis, :]
        #states = tmpBatch[:, (NUM_USERS*2)*2:(NUM_USERS*2)*3]
        #states = states[np.newaxis, :]

        actions = tmpBatch[:, (NUM_USERS*2)*3:(NUM_USERS*2)*3+3]
        rewards = tmpBatch[:, (NUM_USERS*2)*3+3:(NUM_USERS*2)*3+6]
        logger.info(f'@@ PER - after sampling memory.update with states :\n{states}\n')
        logger.info(f'@@ PER - after sampling memory.update with actions :\n{actions}\n')
        logger.critical(f'@@ PER - after sampling  memory.update with rewards :\n{rewards}\n')
        #next_states = tmpBatch[:, (NUM_USERS*2)*4:(NUM_USERS * 2)*5]
        #next_states = tmpBatch[:, (NUM_USERS*2)*6:(NUM_USERS * 2)*7]
        next_states = tmpBatch[:, :(NUM_CHANNELS*2)+2]
        next_states = next_states[np.newaxis, :]
        logger.info(f'@@ PER - after sampling  memory.update with next_states :\n{next_states}\n')

        # (1,5,6)

    else:
        #   matrix of rank 4
        #   shape [NUM_USERS,batch_size,step_size,state_size]
        # (3,2,5)
        logger.info(f"@@ after sampling - batch :\n{batch}\n")
        states = get_states_user(batch)

        #   matrix of rank 3
        #   shape [NUM_USERS,batch_size,step_size]
        actions = get_actions_user(batch)

        #   matrix of rank 3
        #   shape [NUM_USERS,batch_size,step_size]
        rewards = get_rewards_user(batch)

        #   matrix of rank 4
        #   shape [NUM_USERS,batch_size,step_size,state_size]
        next_states = get_next_states_user(batch)

        #   Converting [NUM_USERS,batch_size]  ->   [NUM_USERS * batch_size]
        #   first two axis are converted into first axis
        logger.info(f"Before reshape")
        logger.info(f"## shape of states : {np.shape(states)}")
        logger.info(f"## shape of states.shape[0] : {states.shape[0]}")
        logger.info(f"## shape of states.shape[1] : {states.shape[1]}")
        logger.info(f"## shape of states.shape[2] : {states.shape[2]}")
        logger.info(f"## shape of states.shape[3] : {states.shape[3]}")
        if args.type != "A2C" and args.type != "A2C_ver2"\
                and args.type != "DDQN" and args.type != "PPO":
            states = np.reshape(states,[-1,states.shape[2],states.shape[3]])
            actions = np.reshape(actions,[-1,actions.shape[2]])
            rewards = np.reshape(rewards,[-1,rewards.shape[2]])
            next_states = np.reshape(next_states,[-1,next_states.shape[2],next_states.shape[3]])
        '''
        else:
            states = np.reshape(states,[-1,states.shape[1],states.shape[2]])
            actions = np.reshape(actions,[-1,actions.shape[2]])
            rewards = np.reshape(rewards,[-1,rewards.shape[2]])
            next_states = np.reshape(next_states,[-1,next_states.shape[1],next_states.shape[2]])
        '''
        logger.info(f"After reshape")
        logger.info(f"### shape of states : {np.shape(states)}", )
        logger.info(f"### shape of states.shape[0] : {states.shape[0]}")
        logger.info(f"### shape of states.shape[1] : {states.shape[1]}")
        logger.info(f"### shape of states.shape[2] : {states.shape[2]}")

        logger.info(f"### shape of next_states : {np.shape(next_states)}")

    if args.type != "A2C" and args.type != "A2C_ver2" and args.type != "DDQN"\
            and args.type != "DDPG" and args.type != "PPO":
        #  creating target vector (possible best action)
        logger.info(f'DQN/DRQN - output:{mainQN.output}')
        logger.info(f'DQN/DRQN - inputs_:{mainQN.inputs_}')
        logger.info(f'DQN/DRQN -  next_states:{next_states}')
        target_Qs = sess.run(mainQN.output, feed_dict={mainQN.inputs_:next_states})

        logger.info(f'DQN/DRQN - target_Qs :\n{target_Qs}')
        rewards = np.mean(rewards, axis=1)
        logger.info(f'DQN/DRQN - rewards :\n{rewards}')


        # Q_target =  reward + gamma * Q_next
        logger.info(f'DQN/DRQN - np.max(target_Qs, axis=1) :\n{np.max(target_Qs, axis=1)}')
        targets = rewards + args.gamma * np.max(target_Qs, axis=1)
        logger.info(f'DQN/DRQN - targets :\n{targets}')
        #  calculating loss and train using Adam optimizer
        loss, _ = sess.run([mainQN.loss,mainQN.opt],
                                feed_dict={mainQN.inputs_: states,
                                mainQN.targetQs_: targets,
                                mainQN.actions_: actions[:,-1]})
        logger.critical(f'DQN/DRQN - Training loss :\n{loss}')

        if args.with_per:
            old_val = np.max(target_Qs, axis=1)
            error = abs(old_val - targets)
            logger.info(f'DQN/DRQN - error :\n{error}')
            mainQN.learn(error)

        loss_list.append(loss)

    elif args.type == "A2C":
        logger.info(f'compare replay_memory len:{len(replay_memory)} with MINIMUM_REPLAY_MEMORY:{MINIMUM_REPLAY_MEMORY}')
        if len(replay_memory) < MINIMUM_REPLAY_MEMORY:
            continue

        #values = train_advantage_actor_critic(replay_memory, a2c.actor, a2c.critic)
        #train_a2c(replay_memory, actor, critic)

        # rewards (5,3)
        logger.info(f'A2C - shape(rewards):{np.shape(rewards)}\nrewards.ndim:{rewards.ndim}\nrewards:\n{rewards}')
        #logger.info(f'@@@ values.ndim:{values.ndim} values:{values}')

        rewards = np.sum(rewards, axis=1)
        #rewards = tf.reshape(rewards, (15,))

        minibatch = random.sample(replay_memory, batch_size)
        for index, sample in enumerate(minibatch):
            cur_state, action, reward, next_state = sample

        # actions (5,3)
        logger.info(f'A2C - shape(actions):{np.shape(actions)}\nactions.ndim:{actions.ndim}\nactions:\n{actions}')
        actions = np.sum(actions, axis=1)

        logger.info(f'A2C - minibatch -\nshape(cur_state):{np.shape(cur_state)}\ncur_state:\n{cur_state}')
        logger.info(f'\nshape(action):{np.shape(action)}\naction:\n{action}')
        logger.info(f'\nshape(next_state):{np.shape(next_state)}\nnext_state:\n{next_state}')
        logger.info(f'\nshape(reward):{np.shape(reward)}\nreward:\n{reward}')

        #preprocess1(states, actions, next_states, rewards, 1, s_queue, a_queue, n_s_queue, r_queue)
        preprocess1(cur_state, action, next_state, reward, 1, s_queue, a_queue, n_s_queue, r_queue)
        logger.info('##### after preprocess1 #####\n')
        logger.info(f'@@ s_queue :\n{s_queue}')
        logger.info(f'@@ a_queue :\n{a_queue}')
        logger.info(f'@@ r_queue :\n{r_queue}')

        state_batch, action_batch, next_state_batch, reward_batch = preprocess2(s_queue, a_queue, n_s_queue, r_queue)

        logger.info('##### after preprocess2 #####\n')
        logger.info(f'A2C - state_batch :\n{state_batch}\n@@@ action_batch :\n{action_batch}\n@@@ reward_batch :\n{reward_batch}\n@@@ next_state_batch :\n{next_state_batch}')
        logger.info(f'A2C - shape(action_batch):{np.shape(action_batch)}\n')
        logger.info(f'A2C - shape(state_batch):{np.shape(state_batch)}\n')
        logger.info(f'A2C - shape(reward_batch):{np.shape(reward_batch)}\n')

        #a2c.make_gae()

        #al, cl = a2c.learn(state_batch, action_batch, reward_batch)
        #al, cl, TD_errors = a2c.learn_(states, actions, next_states, rewards)
        #al, cl, TD_errors = a2c.learn_(np.reshape(state_batch, [24,4]), action_batch, np.reshape(next_state_batch, [24,4]), reward_batch)
        #al, cl, TD_errors = a2c.learn_(state_batch, action_batch, next_state_batch, reward_batch)
        #al, cl, TD_errors = a2c.learn_(np.reshape(state_batch, [9,4]), action_batch, np.reshape(next_state_batch, [9,4]), reward_batch, is_weights)

        state_batch = np.reshape(state_batch, [9, 4])
        state_batch = state_batch[:4, :]

        next_state_batch = np.reshape(next_state_batch, [9, 4])
        next_state_batch = next_state_batch[:4, :]

        if args.with_per:
            is_weights = is_weights[:4]

        temp_step_loss_check_count += 1

        if args.with_per:
            #al, cl, TD_errors = a2c.learn_(state_batch, action_batch, next_state_batch, reward_batch, is_weights)
            al, cl, TD_errors = a2c.train_step(state_batch, action_batch, next_state_batch, reward_batch, is_weights)
        else:
            action_batch = action_batch[:4]
            al, cl = a2c.learn(state_batch, action_batch, reward_batch)
            #total_loss = a2c.learn(states, actions, rewards, next_states)

        al = tf.reduce_mean(al)
        cl = tf.reduce_mean(cl)

        '''
        if args.with_per:
            idx, weights, transition = memory.sample(batch_size)
            idx = np.unique(idx)
            logger.info(f'@@@ idx: {idx}, transition:\n{transition}')
            for i in range(len(idx)):
                al, cl = a2c.learn_experience(idx[i], weights[i], transition[i])
        '''

        priorities = []
        if args.with_per:
            prios = ((al + cl) / 2.0 + 1e-5)
            #prios = tf.squeeze(((al + cl) / 2.0 + 1e-5))
            #prios = prios.astype(int)
            #prios = abs(((al + cl) / 2.0 + 1e-5).squeeze())

            td_errors = TD_errors
            new_priorities = np.abs(td_errors) + explore_p
            #new_priorities = tf.squeeze(td_errors) + explore_p
            logger.info(f'new_priorities.shape():{np.shape(new_priorities)}\nnew_priorities : {new_priorities}')
            #new_priorities = tf.squeeze(new_priorities)

            tf_max_value = tf.math.reduce_max(new_priorities)
            #priorities.append(np.max(new_priorities, axis=1))
            priorities.append(tf_max_value)

            logger.info(f"@@@ idx : {idx}\npriorities : {priorities}")

            memory.update_priorities(idx, priorities)
            #memory.update_priorities(idx, prios.data.cpu().numpy())


        logger.info(f"---------- actor loss : {al}\n")
        logger.info(f"---------- critic loss : {cl}\n")

        if args.with_per:
            actor_losses.append(al)
            critic_losses.append(cl)

        else:
            al = np.mean(al)
            actor_losses.append(al)
            critic_losses.append(cl)

        total_loss = al + cl
        logger.info(f'---------- total_loss : {total_loss}')
        loss_list.append(total_loss)

        '''
        if train_advantage_actor_critic(replay_memory, actor, critic) == False:
            logger.info(f"##### train_advantage_actor_critic FALSE !! #####")
            continue
        else:
            logger.info(f"##### train_advantage_actor_critic TRUE !! #####")
            #actor.learn(batch, batch_size, feature_size=NUM_USERS*2)
        '''

    elif args.type == "PPO":
        #ppo.train_network(step_size)
        #logger.info(f'---------- total_loss : {total_loss}')
        #loss_list.append(total_loss)

        minibatch = random.sample(replay_memory, batch_size)
        for index, sample in enumerate(minibatch):
            cur_state, action, reward, next_state = sample

        # actions (5,3)
        logger.info(f'@@@ shape(actions):{np.shape(actions)}\nactions.ndim:{actions.ndim}\nactions:\n{actions}')
        actions = np.sum(actions, axis=1)

        logger.critical(f'@ minibatch -\nshape(cur_state):{np.shape(cur_state)}\ncur_state:\n{cur_state}')
        logger.info(f'shape(action):{np.shape(action)}\naction:\n{action}')
        logger.info(f'shape(next_state):{np.shape(next_state)}\nnext_state:\n{next_state}')
        logger.info(f'shape(reward):{np.shape(reward)}\nreward:\n{reward}')

        if args.with_per:
            preprocess1(cur_state, action, next_state, reward, 1, s_queue, a_queue, n_s_queue, r_queue)
            logger.info('##### after preprocess1 #####\n')
            logger.info(f'@@ s_queue :\n{s_queue}')

            state_batch, action_batch, next_state_batch, reward_batch = preprocess2(s_queue, a_queue, n_s_queue, r_queue)

            logger.info('##### after preprocess2 #####\n')
            logger.critical(f'@@ state_batch :\n{state_batch}\n@@@ action_batch :\n{action_batch}\n@@@ reward_batch :\n{reward_batch}\n@@@ next_state_batch :\n{next_state_batch}')
            logger.info(f'@@ shape(action_batch):{np.shape(action_batch)}\n')
            logger.critical(f'@@ shape(state_batch):{np.shape(state_batch)}\n')

            state_batch = np.reshape(state_batch, [4, 9])
            #state_batch = state_batch[:4, :]

            next_state_batch = np.reshape(next_state_batch, [4, 9])
            #next_state_batch = next_state_batch[:4, :]

            #np.reshape(state_batch, [24, 4]), action_batch, np.reshape(next_state_batch, [24, 4]), reward_batch
            #total_loss = ppo.learn_(np.reshape(state_batch, [9, 4]), action_batch, reward_batch, np.reshape(next_state_batch, [9, 4]))
            #total_loss = ppo.learn_(state_batch, action_batch, reward_batch, next_state_batch)

            #memory.update_priorities(idx, priorities)

        if args.with_per:
            #total_loss = ppo.learn_(state_batch, action_batch, reward_batch, next_state_batch)
            total_loss = ppo.train_ppo_with_GT(state_batch, action_batch, reward_batch, next_state_batch)
        else:
            total_loss = ppo.train_ppo_with_GT(state_batch, action_batch, reward_batch, next_state_batch)
            #total_loss = ppo.train_ppo_with_GT(cur_state, action, reward, next_state)

        logger.critical(f'@ total_loss : {total_loss}')
        loss_list.append(total_loss)
    elif args.type == "A2C_ver2":
        total_loss = a2c.learn(states, actions, rewards, next_states, False)
        logger.info(f'---------- total_loss : {total_loss}')
        loss_list.append(total_loss)
    elif args.type == "DDQN":
        #train_ddqn(replay_memory, batch_size)
        total_loss = mainQN.learn(memory, replay_memory, batch_size)
        loss_list.append(total_loss)

    elif args.type == "DDPG":
        #train_ddpg(replay_memory, batch_size)

        mainQN.learn(states, actions, next_states, rewards, False)
        mainQN.soft_target_update()
    else:
        logger.info(f"### No need to do sess.run in other model ###")
        logger.info(f)

    logger.info(f'$$$$ until time:{time_step} loss_list:{loss_list}')

    if args.type == "A2C":
        # some book keeping
        logger.info(f'@ Before book keeping episode_reward:{episode_reward} max_reward:{max_reward}\n')
        if (episode_reward > max_reward):
            #2c.actor.model.layer[1].save_weights(str(episode_reward) + "_actor"+ ".h5")
            a2c.actor.save_weights("./save_weights/" + str(episode_reward) + "_actor.h5")
            a2c.critic.save_weights("./save_weights/" + str(episode_reward) + "_critic.h5")

        if time_step % 500 == 0:
            #a2c.actor.model.save_weights(str(episode_reward) + ".h5")
            a2c.actor.model.save("./save_weights/periodic_mbr_a2c_actor.h5")
            a2c.critic.model.save("./save_weights/periodic_mbr_a2c_critic.h5")

        max_reward = max(max_reward, episode_reward)
        logger.info(f'@ After book keeping episode_reward:{episode_reward} max_reward:{max_reward}\n')
    elif args.type == "PPO":
        ppo.model_actor.save_weights("./save_weights/mbr_ppo_actor.h5")
        ppo.model_critic.save_weights("./save_weights/mbr_ppo_critic.h5")
        max_reward = max(max_reward, episode_reward)
        logger.info(f'@ After book keeping episode_reward:{episode_reward} max_reward:{max_reward}\n')

    logger.info(f"///// Training block END /////")

    #reward = -100

    #   Training block ends
    ########################################################################################

    logger.info(f'##### main simulation loop END #####\n')

    '''
    if args.type == "A2C":
        # some book keeping
        logger.info(f'@ Before book keeping episode_reward:{episode_reward} cum_r[-1]:{cum_r[-1]}\n')
        if (episode_reward >= 100): #and (episode_reward > cum_r[-1]):
            actor.model.save_weights("book_keeping" + str(episode_reward) + ".h5")
            episode_reward = episode_reward - args.reward_discount * episode_reward
            logger.info(f'@ After book keeping episode_reward:{episode_reward}\n')
    '''

    logger.info(f'##### t: {time_step}\n##### episode_reward: {episode_reward}\n##### max_reward: {max_reward}\n##### epsilon: {explore_p}')
    '''
    #if time_step % 5000 == 4999:
    #if time_step % 1000 == 999:
    
    if time_step % TIME_SLOTS == (TIME_SLOTS-1):
        logger.info(f"##### PLOT start ! #####")
        plt.figure(1, figsize=(15, 10), dpi= 80, facecolor='w', edgecolor='k')
        plt.subplot(411)
        plt.plot(np.arange(TIME_SLOTS), total_rewards ,"r+")
        plt.xlabel('Time Slots')
        plt.ylabel('total rewards')
        plt.title('total rewards given per time_step')
        #plt.show()
        plt.subplot(412)
        plt.plot(np.arange(TIME_SLOTS+1), cum_collision, "r-")
        plt.xlabel('Time Slot')
        plt.ylabel('cumulative collision')
        #plt.show()
        plt.subplot(413)
        plt.plot(np.arange(TIME_SLOTS+1), cum_r, "b-")
        plt.xlabel('Time Slot')
        plt.ylabel('Cumulative reward of all users')
        #plt.title('Cumulative reward of all users')
        plt.subplot(414)
        plt.plot([2.00 for _ in range(time_step)], linestyle="--")
    
        a2c_scores = []
        drqn_scores = []
        ddqn_scores = []
        ddpg_scores = []
    
        plt.plot(np.mean(all_means, axis=0))
    
        if args.type == "A2C":
            plt.legend(["Best Possible", "ActorCritic"])
            a2c_scores = np.mean(all_means, axis=0)
            np.save("a2c_scores", a2c_scores)
            np.savetxt("a2c_scores.txt", a2c_scores, fmt=['%1.16f'], header='all_means', delimiter= ',')
    
        elif args.type == "DRQN":
            plt.legend(["Best Possible", "DRQN"])
            drqn_scores = np.mean(all_means, axis=0)
            np.save("drqn_scores", drqn_scores)
            np.savetxt("drqn_scores.txt", drqn_scores, fmt=['%1.16f'], header='all_means', delimiter= ',')
    
        elif args.type == "DDQN":
            plt.legend(["Best Possible", "DDQN"])
            ddqn_scores = np.mean(all_means, axis=0)
            np.save("ddqn_scores", ddqn_scores)
            np.savetxt("ddqn_scores.txt", ddqn_scores, fmt=['%1.16f'], header='all_means', delimiter= ',')
    
        elif args.type == "DDPG":
            plt.legend(["Best Possible", "DDPG"])
            ddpg_scores = np.mean(all_means, axis=0)
            np.save("ddpg_scores", ddpg_scores)
            np.savetxt("ddpg_scores.txt", ddpg_scores, fmt=['%1.16f'], header='all_means', delimiter= ',')
    
        #np.load("a2c_scores.npy")
        #plt.plot(a2c_scores)
    
        plt.xlabel('Time Slot')
        plt.ylabel('Mean reward of all users')
        plt.show()
        '''


logger.info(f'%%%%%%%%%% npy saving & plot START %%%%%%%%%%\n')

if args.type == "DQN":
    dqn_scores = []
    dqn_scores = all_means
    np.save("dqn_scores", dqn_scores)
    np.save("dqn_means", means)
    np.save("dqn_losses", loss_list)
    draw_res2(time_step, cum_collision, cum_r, loss_list, means, dqn_scores, TIME_SLOTS)
    saver.save(sess, "checkpoints/dqn-user.ckpt")
elif args.type == "DDQN":
    ddqn_scores = []
    ddqn_scores = all_means

    ddqn_loss = []
    ddqn_loss = loss_list
    np.save("ddqn_scores", ddqn_scores)
    np.save("ddqn_means", means)
    np.save("ddqn_loss", ddqn_loss)
    draw_res2(time_step, cum_collision, cum_r, loss_list, means, ddqn_scores, TIME_SLOTS)
    #saver.save(sess,'checkpoints/ddqn-user.ckpt')
elif args.type == "DRQN":
    drqn_scores = []
    drqn_scores = all_means
    #drqn_scores = cum_r
    np.save("drqn_scores", drqn_scores)
    np.save("drqn_means", means)
    np.save("drqn_losses", loss_list)

    draw_res2(time_step, cum_collision, cum_r, loss_list, means, drqn_scores, TIME_SLOTS)
    saver.save(sess,'checkpoints/drqn-user.ckpt')
elif args.type == "DDPG":
    ddpg_scores = []
    ddpg_scores = np.mean(all_means, axis=0)
    np.save("ddpg_scores", ddpg_scores)
    draw_res2(time_step, cum_collision, cum_r, loss_list, means, ddpg_scores, TIME_SLOTS)
    saver.save(sess, 'checkpoints/ddpg-user.ckpt')
elif args.type == "A2C":
    a2c_scores = []
    #a2c_scores = cum_r

    #a2c_scores = np.mean(all_means, axis=0)
    a2c_scores = all_means

    np.save("a2c_scores", a2c_scores)
    np.save("a2c_means", means)
    np.save("a2c_losses", loss_list)

    np.save("actor_losses", actor_losses)
    np.save("critic_losses", critic_losses)

    draw_res2(time_step, cum_collision, cum_r, loss_list, means, a2c_scores, TIME_SLOTS)
    #draw_res2(time_step, cum_collision, total_rewards, loss_list, means, a2c_scores, TIME_SLOTS)

    draw_losses(time_step+1, actor_losses, critic_losses)
    #saver.save(sess,'checkpoints/actor-critic-user.ckpt')
    a2c.actor.save('saved_model/actor-critic-model.keras')
elif args.type == "A2C_ver2":
    a2c_v2_scores = []
    a2c_v2_scores = np.mean(all_means, axis=0)
    np.save("a2c_v2_scores", a2c_v2_scores)
    draw_res2(time_step, cum_collision, cum_r, loss_list, means, a2c_v2_scores, TIME_SLOTS)
    saver.save(sess,'checkpoints/actor-critic-ver2-user.ckpt')
elif args.type == "PPO":
    ppo_scores = []
    ppo_scores = all_means
    np.save("ppo_scores", ppo_scores)
    np.save("ppo_means", means)
    np.save("ppo_losses", loss_list)
    draw_res2(time_step, cum_collision, cum_r, loss_list, means, ppo_scores, TIME_SLOTS)
    #saver.save(sess,'checkpoints/ppo-user.ckpt')

logger.info(f'%%%%%%%%%% npy saving & plot END %%%%%%%%%%\n')

#print time_step,loss , sum(reward) , Qs

logger.info(f'********** All process is finished **********\n')
total_rewards = []
means = [0]
all_means = []
cum_r = [0]
cum_collision = [0]
loss_list = [0]

logger.info(f'********** All variables are initialized **********\n')







