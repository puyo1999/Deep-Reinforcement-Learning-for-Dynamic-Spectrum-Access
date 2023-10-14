import keras.src.losses
import tensorflow as tf
tf.executing_eagerly()

from env.multi_user_network_env import env_network
from keras.losses import huber
from util.memory_buffer import Memory
from util.prioritized_memory import PerMemory
from util.parser import Parser
from util.utils import get_states_user, get_actions_user, get_rewards_user, get_next_states_user
from util.utils import draw_res2, draw_multi_algorithm
from model.A2C.a2c import A2C
from model.DDPG.ddpg import DDPG
from model.drqn import QNetwork
from model.dqn import DQNetwork
from model.ddqn import DDQN
import numpy as np
import random
import sys
from collections import deque
from multiprocessing import Queue
import tensorflow.compat.v1 as tf
#tf.disable_v2_behavior()

from tensorflow.python.framework.ops import enable_eager_execution
enable_eager_execution()

import yaml

import logging
logger = logging.getLogger(__name__)

with open('./config/config.yaml') as f:
    config = yaml.safe_load(f)

args = None
args = Parser.parse_args(args)

#TIME_SLOTS = 100000                            # number of time-slots to run simulation
#TIME_SLOTS = 1000
#TIME_SLOTS = args.time_slot
TIME_SLOTS = config['time_slots']
NUM_CHANNELS = config['num_channels']                               # Total number of channels
NUM_USERS = config['num_users']                                 # Total number of users
ATTEMPT_PROB = 1                               # attempt probability of ALOHA based  models 

MINIMUM_REPLAY_MEMORY = 32

#memory_size = 1000                      #size of experience replay deque
args.memory_size = config['memory_size']

batch_size = 32                         # Num of batches to train at each time_slot
pretrain_length = batch_size            # this is done to fill the deque up to batch size before training
hidden_size = 128                       # Number of hidden neurons
learning_rate = 1e-4                    # learning rate
actor_lr = 1e-4
critic_lr = 5e-4             # critic learning rate
explore_start = .02                     # initial exploration rate
explore_stop = .01                      # final exploration rate
decay_rate = .0001                      # rate of exponential decay of exploration
#gamma = .99                             # discount  factor


args.log_level_info = config['log_level_info']

if args.log_level_info:
    logger.setLevel(logging.DEBUG) # 모든 레벨의 로그를 Handler들에게 전달해야 합니다.
else:
    logger.setLevel(logging.WARNING) # WARNING 레벨 이상의 로그를 Handler들에게 전달해야 합니다.

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

logger.info('########## config.yaml ##########')
logger.info(config)
logger.info('########## config.yaml ##########')
#logger.info(fconfig)

args.gamma = config['gamma']
args.reward_discount = config['reward_discount']
args.type = config['type']              # DL algorithm
args.with_per = config['with_per']
args.graph_drawing = config['graph_drawing']

if args.graph_drawing:
    data1 = np.genfromtxt("a2c_scores.txt", delimiter=",")

    data2 = np.load("a2c_scores.npy")
    logger.info(f"@ shape of data1 : {np.shape(data1)}")
    logger.info(f"@ shape of data2 : {np.shape(data2)}")

    data3 = np.load("drqn_scores.npy")
    data4 = np.load("ddqn_scores.npy")
    logger.info(f"@ shape of data3 : {np.shape(data3)}")
    logger.info(f"@ shape of data4 : {np.shape(data4)}")
    draw_multi_algorithm(data2, data3, data4)

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

step_size = 1 + 2 + 2                   #length of history sequence for each datapoint  in batch
state_size = 2 * (NUM_CHANNELS + 1)     #length of input (2 * k + 2)   :k = NUM_CHANNELS
action_size = NUM_CHANNELS+1            #length of output  (k+1)
alpha = 0                               #co-operative fairness constant
beta = 1                                #Annealing constant for Monte - Carlo

#It creates a one hot vector of a number as num with size as len
def one_hot(num, len):
    #assert num >= 0 and num < len, "!! one_hot error"
    assert (num >= 0) & (num < len), "!! one_hot error"
    vec = np.zeros([len],np.int32)
    vec[num] = 1
    return vec

#generates next-state from action and observation
def state_generator(action, obs):
    input_vector = []
    if action is None:
        logger.info(f'no action, hence, no next_state !')
        sys.exit()
    logger.info(f'action.size:{action.size}')
    for user_i in range(action.size):
        logger.info(f'user_i:{user_i} action:{action[user_i]}')
        input_vector_i = one_hot(action[user_i],NUM_CHANNELS+1)
        channel_alloc = obs[-1] # obs 뒤에서 첫번째
        input_vector_i = np.append(input_vector_i,channel_alloc)
        input_vector_i = np.append(input_vector_i,int(obs[user_i][0]))    #ACK
        input_vector.append(input_vector_i)
    return input_vector


# reseting default tensorflow computational graph
tf.reset_default_graph()
sess = tf.Session()

# initializing the environment
env = env_network(NUM_USERS,NUM_CHANNELS,ATTEMPT_PROB)

#to sample random actions for each user
action = env.sample()

#
obs = env.step(action)
state = state_generator(action,obs)
reward = [i[1] for i in obs[:NUM_USERS]]
logger.info(f'##### Before init Deep Q Network #####\n')
logger.info(f'##### action :\n{action}\n')
logger.info(f'##### obs :\n{obs}\n')
logger.info(f'##### state :\n{state}\n')
logger.info(f'##### reward :\n{reward}\n')

#this is experience replay buffer(deque) from which each batch will be sampled and fed to the neural network for training
if args.with_per:
    memory = PerMemory(mem_size=args.memory_size, feature_size=NUM_USERS*2, prior=True)
else:
    memory = Memory(max_size=args.memory_size)

#initializing deep Q network model
if args.type == "DQN":
    logger.info(f"##### init DQN #####")
    mainQN = DQNetwork(name='main',hidden_size=hidden_size,learning_rate=learning_rate,step_size=step_size,state_size=state_size,action_size=action_size)
elif args.type == "DRQN":
    logger.info(f"##### init DRQN #####")
    mainQN = QNetwork(name='QNetwork',hidden_size=hidden_size,learning_rate=learning_rate,step_size=step_size,state_size=state_size,action_size=action_size, memory=memory)
elif args.type == "A2C":
    logger.info(f"##### A2C #####")
    a2c = A2C(sess=sess, act_dim=action_size, obs_dim=NUM_USERS*2, actor_lr=actor_lr, critic_lr=critic_lr, memory=memory)
elif args.type == "DDQN":
    logger.info(f"#### DDQN #####")
    mainQN = DDQN(name='main', feature_size=NUM_USERS*2, learning_rate=learning_rate, state_size=state_size, actions=range(action_size), action_size=action_size, step_size=step_size, prior=args.with_per, memory=memory, gamma=args.gamma)
elif args.type == "DDPG":
    #mainQN = DDPG(name= 'main', env=env, obs_dim=env.action_space.shape, act_dim=env.action_space.shape, memory=memory, steps=10000)
    mainQN = DDPG(name='main', env=env, obs_dim=NUM_USERS*2, act_dim=action_size, memory=memory, steps=10000)

replay_memory = deque(maxlen=100)

#this is our input buffer which will be used for  predicting next Q-values
if args.type == "A2C":
    history_input = deque(maxlen=action_size)
else:
    history_input = deque(maxlen=step_size)

step = 0
start_train = False
##############################################
for ii in range(pretrain_length*step_size*5):
    done = False
    action = env.sample()
    obs = env.step(action)      # obs is a list of tuple with [[(ACK,REW) for each user] ,CHANNEL_RESIDUAL_CAPACITY_VECTOR]
    next_state = state_generator(action,obs)
    reward = [i[1] for i in obs[:NUM_USERS]]

    if args.with_per:
        #memory.add(td_error, (state, action, reward, next_state))
        if args.type == "A2C":
            a2c.actor.store_transition(state, action, reward, next_state)
        else:
            mainQN.store_transition(state, action, reward, next_state)
        #memory.store((state, action, reward, next_state))
    else:
        memory.add((state, action, reward, next_state))

    #replay_memory.append((state, np.argmax(action),reward, next_state))
    replay_memory.append((state, action, reward, next_state))

    state = next_state
    history_input.append(state)
    logger.info(f'@@ Pretrain step:{step} after store_transition')

    if step >= args.memory_size:
        if not start_train:
            start_train = True
            logger.info(f'@@ Now start_train:{start_train}')
            break
        #if args.type == 'A2C':
            #actor.learn(batch, batch_size, feature_size=NUM_USERS*2)
    step += 1

interval = 1       # debug interval

# saver object to save the checkpoints of the DQN to disk
#saver = tf.train.Saver()
saver = tf.train.Checkpoint()

#initializing the session
sess = tf.Session()

#initialing all the tensorflow variables
#sess.run(tf.global_variables_initializer())

def train_ddpg(replay_memory, batch_size):
    global_steps = 0
    epochs = 0
    rewards_list = []
    minibatch = random.sample(replay_memory, batch_size)
    for index, sample in enumerate(minibatch):
        cur_state, action, reward, next_state = sample

    if args.with_per:
        idx, weights, transition = memory.sample(batch_size)
    else:
        #cur_state, action, reward, next_state = sample
        transition = memory.sample(batch_size, step_size)

    mainQN.learn(*map(lambda x: np.stack(x).astype('float32'), np.transpose(transition)))
    mainQN.soft_target_update()

    mainQN.actor.save_weights("./ddpg_actor/actor", overwrite=True)

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

def train_a2c(replay_memory, actor, critic):
    minibatch = random.sample(replay_memory, batch_size)
    X = []
    y = []
    delta = []
    advantages = np.zeros([batch_size], dtype=object)
    for index, sample in enumerate(minibatch):
        cur_state, action, reward, next_state = sample
        next_reward = critic.model.predict(np.expand_dims(next_state, axis=0))[0][0]
        advantages[index] = reward + args.gamma * next_reward - critic.model.predict(np.expand_dims(cur_state, axis=0))
        # Updating reward to trian state value fuction V(s_t)
        reward = reward + args.gamma * next_reward
        X.append(cur_state)
        y.append(reward)
        delta.append(advantages[index])


    X = np.array(X)
    y = np.array(y)
    #delta = np.array(delta)
    y = np.expand_dims(y, axis=1)
    # Training Actor and Critic - state_input [3, 6],
    #actor.train([X, delta], advantages)
    logger.info(f'shape of X:{np.shape(X)}\n shape of delta:{np.shape(delta)}\n action:{np.shape(action)}\n shape of y:{np.shape(y)}\n')
    actor.model.fit([X, delta], action, batch_size=batch_size, verbose=0)
    critic.model.fit(X, y, batch_size=batch_size, verbose=0)

def train_advantage_actor_critic(replay_memory, actor, critic):
    logger.info(f'############### A2C training 시작 ####################')
    minibatch = random.sample(replay_memory, batch_size)
    X = []
    y = []
    delta = []
    A = []

    advantages = np.zeros(shape=(batch_size, action_size))
    value = np.zeros(shape=(batch_size, action_size))
    next_value = np.zeros(shape=(batch_size, action_size))

    logger.info(f'minibatch : {minibatch}')
    for index, sample in enumerate(minibatch):
        cur_state, action, reward, next_state = sample
        logger.info(f'$$$$$ index: {index} sampled reward : {reward}')

        if np.shape(cur_state) != np.shape(next_state):
            logger.info(f'{cur_state}\n DIFF \n{next_state}')
            next_state = np.reshape(next_state, [3, 6])

        tempX = np.zeros((3, 6), dtype=float)
        tempX = np.array(cur_state)
        tempX = np.array(tempX)
        tempX = tempX[np.newaxis, :]

        tempX_ = np.zeros((3, 6), dtype=float)
        tempX_ = np.array(next_state)
        tempX_ = np.array(tempX_)
        tempX_ = tempX_[np.newaxis, :]

        for user_i in range(action_size):
            logger.info(f'* action: {action}\n')
            value[index][action[user_i]] = critic.model.predict(tempX)[0][0]
            next_value[index][action[user_i]] = critic.model.predict(tempX_)[0][0]
            logger.info(f'* value: {value[index][action[user_i]]}\n next_value: {next_value[index][action[user_i]]}\n')

        # if done: 과 동일
        logger.info(f' time_step : {time_step}\n')
        if time_step == TIME_SLOTS:
            logger.info(f'@@@ DONE @@@\n')
            for user_i in range(action_size):
                advantages[index][action[user_i]] = reward[user_i] - value[index][action[user_i]]
                #logger.info(f'* user_i: {} advantages: {}\n'.format(user_i, advantages[index][action[user_i]]))
                reward[user_i] = -100
                logger.info(f'@@@ reward punishment by (-100) @@@\n')

        else:

            for user_i in range(action_size):
                # if not last state
                # the advantage A(s_t, a_t) = reward_t + DISCOUNT * V(s_(t+1)) - V(s_t)
                next_reward = next_value[index][action[user_i]]
                logger.info(f'@@ next_reward : {next_reward}')
                #logger.info(f'* user_i: {} next_reward: {}\n'.format(user_i, next_reward))
                # Critic calculates the TD error
                advantages[index][action[user_i]] = reward[user_i] + args.gamma * next_reward - value[index][action[user_i]]
                # Updating reward to train state value function V(s_t)
                reward[user_i] = reward[user_i] + args.gamma * next_reward
                logger.info(f'@@ updated reward[{user_i}] : {reward[user_i]}\n')

                # 나름의 정규화를 위해 reward 감소분 적용해보자
                if (sum_r > 30000 and reward[user_i] > 10000):
                    #actor.model.save_weights(str(sum_r) + ".h5")
                    reward[user_i] = reward[user_i] - args.reward_discount * next_reward

        state_ = np.array(cur_state)
        X.append(state_)

        y.append(reward)
        #logger.info(f"@@ iter index:{} -\n X:\n{}\n y:\n{}\n advantages[][]:\n{} ".format(index, X, y, advantages[index][action]))
        logger.info(f'$$$$$ index: {index} y : {y}')

        advantages_ = np.array(advantages)
        #for user_i in range(action_size):
            #delta.append(advantages[index][action[user_i]])
        delta.append(advantages[index])

        #logger.info(f'$$$$$ calculated delta : {}'.format(delta))

        action_ = np.array(action)
        A.append(action_)
        #action = action[np.newaxis, :]
    #END for index, sample in enumerate(minibatch):

    state_ = state_[np.newaxis, :]
    action_ = action_[np.newaxis, :]
    advantages_ = advantages_[0][np.newaxis, :]
    #logger.info(f'@@@ pyk @@@ before fit\n state_:\n{}\n advantages_: \n{}\n action_: \n{}\n'.format(state_, advantages_, action_))

    #if args.with_per:
        #actor.model.fit([state_, advantages_], action_, verbose=0)

    #logger.info(f'@@@ pyk @@@ before memory.update with next_value:\n{}\n'.format(next_value))
    max_q = np.zeros(shape=(batch_size, 3))
    q_predict = np.zeros(shape=(batch_size, 3))
    if args.with_per:
        idx, w, transition = memory.sample(batch_size)
        for bidx in range(batch_size):
            max_q[bidx][0] = next_value[bidx][0]
            q_predict[bidx][0] = value[bidx][0]

        transition = np.array(transition)

    #logger.info(f'### pyk ### shape of transition : {}\n'.format(np.shape(transition)))
    #logger.info(f'@@@ pyk @@@ transition:\n{}\n'.format(transition))

    #tempState = transition[:, :(NUM_USERS*2)]
    #tempReward = transition[:, (NUM_USERS*2)+1]
    #tempReward = transition[1:(NUM_USERS*2)+1]
    tempReward = transition[:, (NUM_USERS * 2) * 3 + 3: (NUM_USERS * 2) * 3 + 6]

    q_target = np.copy(q_predict)
    q_target = tempReward + args.gamma * max_q

    p = []
    if args.with_per:
        for bidx in range(batch_size):
            p.append(np.sum(np.abs(q_predict[bidx][0] - q_target[bidx][0]), axis=0))
        #logger.info(f'@@@ pyk @@@ update with idx:\n{}\n p:\n{}\n'.format(idx, p))
        memory.update(idx=idx, tderr=p)

    # temporal code for checking update result about min_p
    if args.with_per:
        idx, w, transition = memory.sample(batch_size)

    X = np.array(X)
    y = np.array(y)
    delta = np.array(delta)
    A = np.array(A)

    y = np.expand_dims(y, axis=2)

    # Actor와 Critic 훈련 fit
    #if args.with_per:
        #actor.model.fit([X, delta], A, verbose=0)
        #actor.train(X, advantages)
        #critic.model.fit(X, y, batch_size=batch_size, verbose=0)
    #else:
        #actor.model.fit([X, delta], action, batch_size=batch_size, verbose=0)
        #critic.model.fit(X, y, batch_size=batch_size, verbose=0)

    logger.info(f'################ A2C training 끝 ####################')
    return value


# list of total rewards
total_rewards = []

# list mean reward
all_means = []
means = []

# cumulative reward
cum_r = [0]

# cumulative collision
cum_collision = [0]

# loss list
loss_list = []

#mse_loss = mse()
actor_losses = critic_losses = []

max_reward = 10000
episode_reward = 0

##########################################################################
####                      main simulation loop                    ########

s_queue = Queue()
a_queue = Queue()
r_queue = Queue()

def preprocess1(states, actions, rewards, gamma, s_queue, a_queue, r_queue):
    discnt_rewards = []
    sum_reward = 0
    #rewards.reverse()
    for r in rewards:
      sum_reward = r + gamma*sum_reward
      discnt_rewards.append(sum_reward)
    #discnt_rewards.reverse()
    states = np.array(states, dtype=np.float32)
    actions = np.array(actions, dtype=np.int32)
    discnt_rewards = np.array(discnt_rewards, dtype=np.float32)
    #exp = np.array([states, actions,discnt_rewards])
    #lock.acquire()
    s_queue.put(states)
    a_queue.put(actions)
    r_queue.put(discnt_rewards)
    logger.error(f's_queue : {s_queue}')
    logger.error(f'a_queue : {a_queue}')
    logger.error(f'r_queue : {r_queue}')
    #lock.release()

def preprocess2(s_queue, a_queue, r_queue):
    states = []
    while not s_queue.empty():
        states.append(s_queue.get())

    actions = []
    while not a_queue.empty():
        actions.append(a_queue.get())
    dis_rewards = []
    while not r_queue.empty():
        dis_rewards.append(r_queue.get())

    state_batch = np.concatenate(*(states,), axis=0)
    action_batch = np.concatenate(*(actions,), axis=None)
    reward_batch = np.concatenate(*(dis_rewards,), axis=None)
    # exp = np.transpose(exp)

    return state_batch, action_batch, reward_batch


for time_step in range(TIME_SLOTS):
    logger.info(f'##### main simulation loop START - time_step{time_step} #####')
    # changing beta at every 50 time-slots
    if time_step % 50 == 0:
        if time_step < 5000:
            logger.info(f"***** every 50 time slots : beta decreasing *****")
            beta -= 0.001
    #curent exploration probability
    explore_p = explore_stop + (explore_start - explore_stop) * np.exp(-decay_rate*time_step)
    logger.info(f'explore_p:{explore_p}')

    # Exploration
    if explore_p > np.random.rand():
        #random action sampling
        action = env.sample()
        logger.info(f"+++++ Explored +++++")
        
    # Exploitation
    else:
        logger.info(f"----- Exploited -----")
        #initializing action vector
        action = np.zeros([NUM_USERS], dtype=np.int32)
        #action = np.zeros([NUM_USERS], dtype=np.object)

        #converting input history into numpy array
        state_vector = np.array(history_input)

        logger.info(f"@@@ Converted @@@\n history_input :\n{history_input}\n --> state_vector :\n{state_vector}\n")
        logger.info(f"/////////////// each_user iter starts ///////////////")

        for each_user in range(NUM_USERS):

            if args.type == "A2C":
                #logger.info(f'*** state_vector[uid{}] :\n {}'.format(each_user, state_vector[:,each_user]))
                #logger.info(f'*** state[uid{}] :\n {}'.format(each_user, state[each_user]))

                #state[each_user] = np.resize(state[each_user], [1, action_size, state_size])
                #state[each_user] = np.resize(state[each_user], [1, state_size])

                tempP = np.zeros((3, 6), dtype=np.float32)
                tempP = np.array(state_vector[:,each_user])
                tempP = np.array(tempP)
                #tempP = tempP[np.newaxis, :]
                logger.error(f'tempP :\n{tempP}')
                logger.error(f'state[{each_user}] :\n{state[each_user]}')

                #probs = a2c.actor.policy.predict(tempP)[:,each_user]
                #probs = a2c.actor.predict(tempP)[:,each_user]

                #prob1 = (1-alpha)*np.exp(beta*probs)
                #prob = prob1/np.sum(np.exp(beta*probs)) + alpha/(NUM_CHANNELS+1)

            elif args.type == "DDQN":
                state[each_user] = np.resize(state[each_user], [1, state_size])
                #state[each_user] = state_vector[:,each_user].reshape(step_size,state_size)
                Qs = mainQN.model.predict([np.array(state[each_user]), np.ones((1, 1))])
                prob1 = (1-alpha)*np.exp(beta*Qs)
                prob = prob1/np.sum(np.exp(beta*Qs)) + alpha/(NUM_CHANNELS+1)
            elif args.type == "DDPG":
                state[each_user] = np.resize(state[each_user], [1, state_size])
                Qs = mainQN.actor.predict(state[each_user])
                prob1 = (1-alpha)*np.exp(beta*Qs)
                prob = prob1/np.sum(np.exp(beta*Qs)) + alpha/(NUM_CHANNELS+1)
            else:
                #feeding the input-history-sequence of (t-1) slot for each user separately
                feed = {mainQN.inputs_:state_vector[:,each_user].reshape(1,step_size,state_size)}

                #predicting Q-values of state respectively
                Qs = sess.run(mainQN.output,feed_dict=feed)
                #print Qs

                #   Monte-carlo sampling from Q-values  (Boltzmann distribution)
                ##################################################################################
                prob1 = (1-alpha)*np.exp(beta*Qs)

                # Normalizing probabilities of each action  with temperature (beta)
                prob = prob1/np.sum(np.exp(beta*Qs)) + alpha/(NUM_CHANNELS+1)
                #print prob

            #   This equation is as given in the paper :
            #   Deep Multi-User Reinforcement Learning for  
            #   Distributed Dynamic Spectrum Access :
            #   @Oshri Naparstek and Kobi Cohen (equation 12)
            ########################################################################################

            #  choosing action with max probability
            if args.type == "A2C":
                #action[each_user] = np.random.choice(action_size, 1, p=probs[each_user])
                #action[each_user] = np.argmax(prob, axis=1)
                action[each_user] = a2c.act(state[each_user])[0]

            elif args.type == "DDQN":
                action[each_user] = mainQN.actor(obs[each_user])
                #action[each_user] = np.argmax(prob, axis=1)
            elif args.type == "DDPG":
                #a = mainQN.policy_action(obs[each_user])
                #a = mainQN.policy_action(state[each_user])
                #logger.info(f'@@ ddpg after policy_action a:{}'.format(a))
                action[each_user] = mainQN.get_action(state[each_user], True)
            else:
                action[each_user] = np.argmax(prob,axis=1)

            #action[each_user] = np.random.choice(action_size, 1, p=policy)[0]
            #action[each_user] = np.argmax(Qs,axis=1)

            if time_step % interval == 0:
                logger.info(f'EachUser:{each_user} Debugging state_vector:\n{state_vector[:,each_user]}')
                if args.type != "A2C": #and args.type != "DDQN":
                    logger.info(f'Qs:{Qs}')
                    logger.info(f'prob:{prob}, sum of beta*Qs:{np.sum(np.exp(beta*Qs))}')
                    logger.info(f'End')

    # taking action as predicted from the q values and receiving the observation from the environment
    if args.type != "A2C":
        state = state_generator(action, obs)

    #logger.info(f"@@ action :\n{}".format(action))
    logger.info(f'@@ action :\n{action}')
    logger.info(f'@@ argmax(action) :\n{np.argmax(action)}')

    obs = env.step(action)           # obs is a list of tuple with [(ACK,REW) for each user ,(CHANNEL_RESIDUAL_CAPACITY_VECTOR)]

    logger.info(f"@@ obs :\n{obs}")
    logger.info(f"@@ len(obs) :\n{len(obs)}")

    # Generate next state from action and observation 
    next_state = state_generator(action,obs)
    #logger.info(f"@@ next_state :\n{}".format(next_state))

    # reward for all users given by environment
    reward = [i[1] for i in obs[:NUM_USERS]]
    logger.info(f'$$ reward :\n{reward}')
    # calculating sum of rewards
    # user들의 reward 합
    sum_r = np.sum(reward)
    logger.info(f'$$ sum_r :\n{sum_r}')

    episode_reward += sum_r

    # If NUM_CHANNELS = 2 , total possible reward = 2 , therefore collision = (2 - sum_r) or (NUM_CHANNELS - sum_r)
    collision = NUM_CHANNELS - sum_r
    logger.info(f'!! collision :\n{collision}')

    # calculating cumulative collision
    cum_collision.append(cum_collision[-1] + collision)
   
    #############################
    #  for co-operative policy we will give reward-sum to each user who have contributed
    #  to play co-operatively and rest 0
    for i in range(len(reward)):
        if reward[i] > 0:
            reward[i] = sum_r
    #############################

    total_rewards.append(sum_r)
    logger.info(f'$$ total_rewards :\n{total_rewards}')

    #if time_step % 100 == 99:
        #if time_step < 5000:
            #logger.info(f"***** every 100 time slots *****")
            # TODO:: measures and stores the mean reward score over that period

    # calculating cumulative reward
    logger.info(f'$$$ Before calculating : cum_r + sum_r = {cum_r[-1] + sum_r}')
    cum_r.append(cum_r[-1] + sum_r)
    means.append(cum_r[-1] / (time_step + 1))

    logger.info(f'$$$$ until time:{time_step} means:{cum_r[-1] / (time_step + 1)}')
    all_means.append(means)


    # add new experiences into the memory buffer as (state, action , reward , next_state) for training
    '''if args.with_per:
        td_error = 1
    else:
        td_error = 0'''

    if args.with_per:
        #memory.add(td_error, (state, action, reward, next_state))

        #memory.store((state, action, reward, next_state))
        logger.info(f'Before store_transition - state:{state}')
        if args.type == "A2C":
            a2c.actor.store_transition(state, action, reward, next_state)
            #replay_memory.append((state, action, reward, next_state))
        else:
            mainQN.store_transition(state, action, reward, next_state)
    else:
        memory.add((state, action, reward, next_state))

    state = next_state
    #add new experience to generate input-history sequence for next state
    history_input.append(state)

    logger.info(f'///// BEFORE Training at time_step:{time_step} /////')
    #  Training block starts
    ###################################################################################
    logger.info(f'////////////////////////////////')
    logger.info(f'///// Training block START /////')

    #  sampling a batch from memory buffer for training
    if args.with_per:
        idx, is_weights, batch = memory.sample(batch_size)
        batch = np.array(batch)
    else:
        batch = memory.sample(batch_size, step_size)

    logger.info(f'shape of batch:\n{np.shape(batch)}\n')
    logger.info(f'batch:\n{batch}\n')

    if args.with_per:
        #next_states = np.vstack(batch[3])
        #next_states = get_next_states_user(batch)
        #idx, w, batch = memory.sample(batch_size)

        '''
        for index, sample in enumerate(batch):
            states, actions, rewards, next_states = sample
            logger.info(f'@ after sampling memory.update index:\n{} \n states:\n{}\n'.format(index, states))
        '''
        idx, is_weights, tmpBatch = memory.sample(5)
        tmpBatch = np.array(tmpBatch)
        states = tmpBatch[:, :(NUM_USERS*2)]
        states = states[np.newaxis, :]

        actions = tmpBatch[:, (NUM_USERS*2)*3 : (NUM_USERS*2)*3+3]
        rewards = tmpBatch[:, (NUM_USERS*2)*3+3 : (NUM_USERS*2)*3+6]
        logger.info(f'@ after sampling memory.update with states :\n{states}\n')
        logger.info(f'@ after sampling  memory.update with rewards :\n{rewards}\n')
        next_states = tmpBatch[:, (NUM_USERS*2)*4 : (NUM_USERS * 2)*5]
        next_states = next_states[np.newaxis, :]
        logger.info(f'@ after sampling  memory.update with next_states :\n{next_states}\n')

    else:
        #   matrix of rank 4
        #   shape [NUM_USERS,batch_size,step_size,state_size]
        logger.info(f"@@ after sampling - batch\n : {batch}")
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
        if args.type != "A2C" and args.type != "DDQN":
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

    if args.type != "A2C" and args.type != "DDQN" and args.type != "DDPG":
        #  creating target vector (possible best action)
        target_Qs = sess.run(mainQN.output,feed_dict={mainQN.inputs_:next_states})

        #  Q_target =  reward + gamma * Q_next
        targets = rewards[:,-1] + args.gamma * np.max(target_Qs,axis=1)

        #  calculating loss and train using Adam optimizer
        loss, _ = sess.run([mainQN.loss,mainQN.opt],
                                feed_dict={mainQN.inputs_:states,
                                mainQN.targetQs_:targets,
                                mainQN.actions_:actions[:,-1]})
        loss_list.append(loss)

        if args.with_per:
            old_val = np.max(target_Qs, axis=1)
            error = abs(old_val - targets)
            mainQN.learn(error)

    elif args.type == "A2C":
        logger.info(f'compare replay_memory len:{len(replay_memory)} with MINIMUM_REPLAY_MEMORY:{MINIMUM_REPLAY_MEMORY}')
        if len(replay_memory) < MINIMUM_REPLAY_MEMORY:
            continue

        #values = train_advantage_actor_critic(replay_memory, a2c.actor, a2c.critic)
        #train_a2c(replay_memory, actor, critic)

        # rewards (5,3)
        logger.error(f'@@@ shape(rewards):{np.shape(rewards)}\nrewards.ndim:{rewards.ndim}\nrewards:{rewards}')
        #logger.error(f'@@@ values.ndim:{values.ndim} values:{values}')

        rewards = np.sum(rewards, axis=1)
        #rewards = tf.reshape(rewards, (15,))


        minibatch = random.sample(replay_memory, batch_size)
        for index, sample in enumerate(minibatch):
            cur_state, action, reward, next_state = sample

        # actions (5,3)
        logger.error(f'@@@ shape(actions):{np.shape(actions)}\nactions.ndim:{actions.ndim}\nactions:{actions}')
        actions = np.sum(actions, axis=1)

        preprocess1(states, actions, rewards, 1, s_queue, a_queue, r_queue)

        #state_batch = np.concatenate(*(states,), axis=0)
        #action_batch = np.concatenate(*(actions,), axis=None)
        #reward_batch = np.concatenate(*(rewards,), axis=None)
        #reward_batch = rewards.sum(axis=1)

        state_batch, action_batch, reward_batch = preprocess2(s_queue, a_queue, r_queue)

        logger.error(f'@@@ state_batch :\n{state_batch}\n@@@ action_batch :\n{action_batch}\n@@@ reward_batch :\n{reward_batch}')
        logger.error(f'@@@ shape(action_batch):{np.shape(action_batch)}\n')

        al, cl = a2c.learn(state_batch, action_batch, reward_batch)

        actor_losses.append(al)
        critic_losses.append(cl)
        logger.error(f"al{al}")
        logger.error(f"cl{cl}")

        total_loss = al + cl
        logger.error(f'---------- total_loss : {total_loss}')
        loss_list.append(total_loss)

        '''
        if train_advantage_actor_critic(replay_memory, actor, critic) == False:
            logger.info(f"##### train_advantage_actor_critic FALSE !! #####")
            continue
        else:
            logger.info(f"##### train_advantage_actor_critic TRUE !! #####")
            #actor.learn(batch, batch_size, feature_size=NUM_USERS*2)
        '''
    elif args.type == "DDQN":
        #train_ddqn(replay_memory, batch_size)
        mainQN.learn(memory, replay_memory, batch_size)
    elif args.type == "DDPG":
        train_ddpg(replay_memory, batch_size)
        #mainQN.learn(states, actions, next_states, rewards, False)
        #mainQN.soft_target_update()
    else:
        logger.info(f"### No need to do sess.run in other model ###")
        logger.info(f)

    '''
    if args.type == "A2C":
        # some book keeping
        logger.info(f'@ Before book keeping episode_reward:{episode_reward} max_reward:{max_reward}\n')
        if (episode_reward > 400 and episode_reward > max_reward):
            actor.model.save_weights(str(episode_reward) + ".h5")
        max_reward = max(max_reward, episode_reward)
        logger.info(f'@ After book keeping episode_reward:{episode_reward} max_reward:{max_reward}\n')
    '''
    logger.info(f"///// Training block END /////")

    #reward = -100

    #   Training block ends
    ########################################################################################

    logger.error(f'##### main simulation loop END #####\n')

    '''
    if args.type == "A2C":
        # some book keeping
        logger.error(f'@ Before book keeping episode_reward:{episode_reward} cum_r[-1]:{cum_r[-1]}\n')
        if (episode_reward >= 100): #and (episode_reward > cum_r[-1]):
            actor.model.save_weights("book_keeping" + str(episode_reward) + ".h5")
            episode_reward = episode_reward - args.reward_discount * episode_reward
            logger.error(f'@ After book keeping episode_reward:{episode_reward}\n')
    '''

    logger.error(f'##### t: {time_step} episode_reward: {episode_reward} max_reward: {max_reward} epsilon: {explore_p}')
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


logger.error(f'%%%%%%%%%% npy saving & plot START %%%%%%%%%%\n')

if args.type == "DQN":
    dqn_scores = []
    dqn_scores = np.mean(all_means, axis=0)
    np.save("dqn_scores", dqn_scores)
    draw_res2(time_step, cum_collision, cum_r, loss_list, means, dqn_scores, TIME_SLOTS)
    saver.save(sess, "checkpoints/dqn-user.ckpt")
elif args.type == "DDQN":
    ddqn_scores = []
    ddqn_scores = np.mean(all_means, axis=0)
    np.save("ddqn_scores", ddqn_scores)
    draw_res2(time_step, cum_collision, cum_r, loss_list, means, ddqn_scores, TIME_SLOTS)
    saver.save(sess,'checkpoints/ddqn-user.ckpt')
elif args.type == "DRQN":
    drqn_scores = []
    drqn_scores = np.mean(all_means, axis=0)
    np.save("drqn_scores", drqn_scores)
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
    a2c_scores = np.mean(all_means, axis=0)
    np.save("a2c_scores", a2c_scores)
    draw_res2(time_step, cum_collision, cum_r, loss_list, means, a2c_scores, TIME_SLOTS)
    saver.save(sess,'checkpoints/actor-critic-user.ckpt')
logger.error(f'%%%%%%%%%% npy saving & plot END %%%%%%%%%%\n')

#print time_step,loss , sum(reward) , Qs

logger.error(f'********** All process is finished **********\n')
total_rewards = []
means = []
all_means = []
cum_r = [0]
cum_collision = [0]
los_list = []

logger.error(f'********** All variables are initialized **********\n')

   






