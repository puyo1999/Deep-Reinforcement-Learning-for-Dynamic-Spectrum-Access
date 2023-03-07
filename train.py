from env.multi_user_network_env import env_network
from util.memory_buffer import Memory
from util.prioritized_memory import PerMemory
from util.parser import Parser
from model.ActorNetwork import ActorNetwork
from model.CriticNetwork import CriticNetwork
from model.drqn import QNetwork
from model.dqn import DQNetwork
from model.ddqn import DDQN
import numpy as np
import random
import sys
import matplotlib.pyplot as plt
from collections import deque
import os
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import time
import yaml

with open('./config/config.yaml') as f:
    config = yaml.safe_load(f)
print(config)

args = None
args = Parser.parse_args(args)

#TIME_SLOTS = 100000                            # number of time-slots to run simulation
#TIME_SLOTS = 1000
#TIME_SLOTS = args.time_slot
TIME_SLOTS = config['time_slot']
NUM_CHANNELS = 2                               # Total number of channels
NUM_USERS = 3                                 # Total number of users
ATTEMPT_PROB = 1                               # attempt probability of ALOHA based  models 

MINIMUM_REPLAY_MEMORY = 32

#It creates a one hot vector of a number as num with size as len
def one_hot(num, len):
    assert num >= 0 and num < len, "!! one_hot error"
    vec = np.zeros([len],np.int32)
    vec[num] = 1
    return vec

#generates next-state from action and observation
def state_generator(action, obs):
    input_vector = []
    if action is None:
        print('no action, hence, no next_state !')
        sys.exit()
    for user_i in range(action.size):
        input_vector_i = one_hot(action[user_i],NUM_CHANNELS+1)
        channel_alloc = obs[-1]
        input_vector_i = np.append(input_vector_i,channel_alloc)
        input_vector_i = np.append(input_vector_i,int(obs[user_i][0]))    #ACK
        input_vector.append(input_vector_i)
    return input_vector

memory_size = 1000                      #size of experience replay deque

batch_size = 32                          # Num of batches to train at each time_slot
pretrain_length = batch_size            #this is done to fill the deque up to batch size before training
hidden_size = 128                       #Number of hidden neurons
learning_rate = 1e-4                    #learning rate
explore_start = .02                     #initial exploration rate
explore_stop = .01                      #final exploration rate
decay_rate = .0001                     #rate of exponential decay of exploration
gamma = .99                            #discount  factor

'''
batch_size = args.batch_size
pretrain_length = batch_size  # this is done to fill the deque up to batch size before training
hidden_size = args.hidden  # Number of hidden neurons
learning_rate = args.lr  # learning rate
explore_start = args.explore_start  # initial exploration rate
explore_stop = args.explore_stop  # final exploration rate
decay_rate = args.decay_rate  # rate of exponential decay of exploration
gamma = args.gamma  # discount  factor
print("+++ batch_size : ", batch_size)
print("+++ pretrain_length : ", pretrain_length)
print("+++ hidden_size : ", hidden_size)
print("+++ learning_rate : ", learning_rate)
print("+++ explore_start : ", explore_start)
print("+++ explore_stop : ", explore_stop)
print("+++ decay_rate : ", decay_rate)
print("+++ gamma : ", gamma)
'''

step_size = 1 + 2 + 2                   #length of history sequence for each datapoint  in batch
state_size = 2 * (NUM_CHANNELS + 1)     #length of input (2 * k + 2)   :k = NUM_CHANNELS
action_size = NUM_CHANNELS+1            #length of output  (k+1)
alpha = 0                               #co-operative fairness constant
beta = 1                                #Annealing constant for Monte - Carlo


# reseting default tensorflow computational graph
tf.reset_default_graph()
sess = tf.Session()

# initializing the environment
env = env_network(NUM_USERS,NUM_CHANNELS,ATTEMPT_PROB)

args.type = config['type']

#initializing deep Q network
if args.type == "DQN":
    print("##### DQN #####")
    mainQN = DQNetwork(name='main',hidden_size=hidden_size,learning_rate=learning_rate,step_size=step_size,state_size=state_size,action_size=action_size)
elif args.type == "DRQN":
    print("##### DRQN #####")
    mainQN = QNetwork(name='main',hidden_size=hidden_size,learning_rate=learning_rate,step_size=step_size,state_size=state_size,action_size=action_size)
elif args.type == "A2C":
    print("##### A2C #####")
    actor = ActorNetwork(sess, state_size, action_size)
    critic = CriticNetwork(sess, state_size, action_size)
elif args.type == "DDQN":
    print("#### DDQN #####")
    ddqn = DDQN(name='main', learning_rate=learning_rate, state_size=state_size, action_size=action_size)

#this is experience replay buffer(deque) from which each batch will be sampled and fed to the neural network for training
if args.with_per:
    memory = PerMemory(capacity=memory_size, prior=True)
else:
    memory = Memory(max_size=memory_size)

replay_memory = deque(maxlen = 100)

#this is our input buffer which will be used for  predicting next Q-values   
history_input = deque(maxlen=step_size)

#to sample random actions for each user
action = env.sample()

#
obs = env.step(action)
state = state_generator(action,obs)
reward = [i[1] for i in obs[:NUM_USERS]]


##############################################
for ii in range(pretrain_length*step_size*5):
    
    action = env.sample()
    obs = env.step(action)      # obs is a list of tuple with [[(ACK,REW) for each user] ,CHANNEL_RESIDUAL_CAPACITY_VECTOR]
    next_state = state_generator(action,obs)
    reward = [i[1] for i in obs[:NUM_USERS]]
    # ToDo :: td_error 실제 계산 과정 필요
    if args.with_per:
        td_error = 1
    else:
        td_error = 0
    if args.with_per:
        memory.add(td_error, (state, action, reward, next_state))
    else:
        memory.add((state, action, reward, next_state))

    replay_memory.append((state, np.argmax(action),reward, next_state))

    state = next_state
    history_input.append(state)

##############################################
def get_states(batch): 
    states = []
    for i in batch:
        states_per_batch = []
        for step_i in i:
            states_per_step = []
            for user_i in step_i[0]:
                states_per_step.append(user_i)
            states_per_batch.append(states_per_step)
        states.append(states_per_batch)
    return states

def get_actions(batch):
    actions = []
    for each in batch:
        actions_per_batch = []
        for step_i in each:
            actions_per_step = []
            for user_i in step_i[1]:
                actions_per_step.append(user_i)
            actions_per_batch.append(actions_per_step)
        actions.append(actions_per_batch)

    return actions

def get_rewards(batch):
    rewards = []
    for each in batch:
        rewards_per_batch = []
        for step_i in each:
            rewards_per_step = []
            for user_i in step_i[2]:
                rewards_per_step.append(user_i)
            rewards_per_batch.append(rewards_per_step)
        rewards.append(rewards_per_batch)
    return rewards

def get_next_states(batch):
    next_states = []
    for each in batch:
        next_states_per_batch = []
        for step_i in each:
            next_states_per_step = []
            for user_i in step_i[3]:
                next_states_per_step.append(user_i)
            next_states_per_batch.append(next_states_per_step)
        next_states.append(next_states_per_batch)
    return next_states        

def get_states_user(batch):
    states = []
    for user in range(NUM_USERS):
        states_per_user = []
        for each in batch:
            states_per_batch = []
            for step_i in each:
                try:
                    states_per_step = step_i[0][user]
                    
                except IndexError:
                    print(step_i)
                    print("-----------")
                    print("get_states_user eror")
                    '''
                    for i in batch:
                        print i
                        print "**********"
                    '''
                    sys.exit()
                states_per_batch.append(states_per_step)
            states_per_user.append(states_per_batch)
        states.append(states_per_user)
    #print len(states)
    return np.array(states)

def get_actions_user(batch):
    actions = []
    for user in range(NUM_USERS):
        actions_per_user = []
        for each in batch:
            actions_per_batch = []
            for step_i in each:
                actions_per_step = step_i[1][user]
                actions_per_batch.append(actions_per_step)
            actions_per_user.append(actions_per_batch)
        actions.append(actions_per_user)
    return np.array(actions)

def get_rewards_user(batch):
    rewards = []
    for user in range(NUM_USERS):
        rewards_per_user = []
        for each in batch:
            rewards_per_batch = []
            for step_i in each:
                rewards_per_step = step_i[2][user] 
                rewards_per_batch.append(rewards_per_step)
            rewards_per_user.append(rewards_per_batch)
        rewards.append(rewards_per_user)
    return np.array(rewards)
# 
def get_next_states_user(batch):
    next_states = []
    for user in range(NUM_USERS):
        next_states_per_user = []
        for each in batch:
            next_states_per_batch = []
            for step_i in each:
                next_states_per_step = step_i[3][user] 
                next_states_per_batch.append(next_states_per_step)
            next_states_per_user.append(next_states_per_batch)
        next_states.append(next_states_per_user)
    return np.array(next_states)



interval = 1       # debug interval

# saver object to save the checkpoints of the DQN to disk
saver = tf.train.Saver()

#initializing the session
sess = tf.Session()

#initialing all the tensorflow variables
sess.run(tf.global_variables_initializer())

def train_ddqn(replay_memory, batch_size):
    minibatch = random.sample(replay_memory, batch_size)
    for index, sample in enumerate(minibatch):
        if args.with_per:
            batches, idx, weights = memory.sample(batch_size, step_size)
        else:
            cur_state, action, reward, next_state = sample

    #for state, action, reward, next_state, done in minibatch:
        target = ddqn.model.predict(state)
        if time_step == TIME_SLOTS:
            target[0][action] = reward
        else:
            # a = self.model.predict(next_state)[0]
            t = ddqn.target_model.predict(next_state)[0]
            target[0][action] = reward + ddqn.gamma * np.amax(t)
            # target[0][action] = reward + self.gamma * t[np.argmax(a)]
        ddqn.model.fit(state, target, epochs=1, verbose=0)
    if ddqn.epsilon > ddqn.epsilon_min:
        ddqn.epsilon *= ddqn.epsilon_decay

    if args.with_per:
        p = np.sum(np.abs(t - target), axis=1)
        memory.update(idx, p)


def train_advantage_actor_critic(replay_memory, actor, critic):
    minibatch = random.sample(replay_memory, batch_size)
    X = []
    y = []
    # print("@ action_size : ", action_size)
    # print("@ action_dim : ", action_dim)
    # print("@ shape of X : ", np.shape(X))

    advantages = np.zeros(shape=(batch_size, action_size))
    value = np.zeros(shape=(batch_size, action_size))
    next_value = np.zeros(shape=(batch_size, action_size))
    # advantages = np.zeros(shape=(action_size))
    # print("@ shape of advantages : ", np.shape(advantages))
    for index, sample in enumerate(minibatch):
        cur_state, action, reward, next_state = sample
        print("##### time_step : ", time_step)
        # print("# action : ", action)

        #print("@@@ np.ndim(cur_state) : ", np.ndim(cur_state))

        # Critic 네트워크에서 예측한 가치
        if np.shape(cur_state) != np.shape(next_state):
            return False

        #print("@@@@@@@@@@ Critic Model Summary @@@@@@@@@@")

        #critic.model.summary()
        # Critic 네트워크에서 예측한 가치

        #print("@@@ shape of cur_state : ", np.shape(cur_state))
        #print("@@@ shape of cur_state[0] : ", np.shape(cur_state[0]))
        #print("@@@ shape of reward : ", np.shape(reward))
        if np.ndim(cur_state) == 2 and np.size(cur_state) == 18:
            cur_state[0] = np.array([cur_state[0]])
        #print("KKK shape of cur_state : ", np.shape(cur_state))
        #print("KKK shape of cur_state[0] : ", np.shape(cur_state[0]))

        value[index][action] = critic.model.predict(cur_state[0])[0][0]

        if np.ndim(next_state) == 2 and np.size(next_state) == 18:
            next_state[0] = np.array([next_state[0]])
        #print("KKK shape of next_state : ", np.shape(next_state))
        #print("KKK shape of next_state[0] : ", np.shape(next_state[0]))
        next_value[index][action] = critic.model.predict(next_state[0])[0][0]
        # print("@@@ shape of value : ", np.shape(value))
        # print("@@@ shape of next_value : ", np.shape(next_value))
        # print("@@@ shape of reward : ", np.shape(reward))
        if time_step == TIME_SLOTS:
            # advantages[index][action] = reward - critic.model.predict(np.expand_dims(cur_state, axis=0))[0][0]
            advantages[index][action] = reward[action] - value[index][action]
        else:
            next_reward = critic.model.predict(next_state[0])[0]
            # Critic calculates the TD error
            # advantages[index][action] = reward + gamma * next_reward - critic.model.predict(np.array(cur_state, dtype=object))[0][0]
            advantages[index][action] = reward[action] + gamma * (next_value[index][action]) - value[index][action]

            # Updating reward to train state value fuction V(s_t)
            reward = reward + gamma * next_value

        X.append(cur_state)
        y.append(reward)
    # print("222 shape of X : ", np.shape(X))

    X = np.array(X)
    y = np.array(y)

    # X = np.expand_dims(X, axis=1)
    y = np.expand_dims(y, axis=2)
    # dvantages = np.expand_dims(advantages, axis=0)
    # Actor와 Critic 훈련
    # print("@@ shape of X : ", np.shape(X))
    # print("@@ shape of y : ", np.shape(y))
    # print("@@ shape of advantages : ", np.shape(advantages))

    # print("@@@@@@@@@@ Actor Model Summary @@@@@@@@@@")
    actor.model.summary()

    if args.with_per:
        batches, idx, weights = memory.sample(batch_size, step_size)
    p = np.sum(np.abs(y[0][0] - X[0][0]), axis=1)
    memory.update(idx, p)

    actor.train(X[0][0], advantages)
    # actor.model.fit(X, advantages, batch_size=batch_size, verbose=0)
    critic.model.fit(X[0][0], y[0][0], batch_size=batch_size, verbose=0)



# list of total rewards
total_rewards = []

# list mean reward
all_means = []
means = []

# cumulative reward
cum_r = [0]

# cumulative collision
cum_collision = [0]

##########################################################################
####                      main simulation loop                    ########


for time_step in range(TIME_SLOTS):
    print("$$$$$")
    print()
    print()
    # changing beta at every 50 time-slots
    if time_step % 50 == 0:
        if time_step < 5000:
            print("***** every 50 time slots : beta decreasing *****")
            beta -= 0.001

    #curent exploration probability
    explore_p = explore_stop + (explore_start - explore_stop) * np.exp(-decay_rate*time_step)
   

    # Exploration
    if explore_p > np.random.rand():
        #random action sampling
        action = env.sample()
        print("explored")
        
    # Exploitation
    else:
        print("exploited")
        #initializing action vector
        action = np.zeros([NUM_USERS], dtype=np.int32)

        #converting input history into numpy array
        state_vector = np.array(history_input)

        #print np.array(history_input)
        print ("///////////////")

        for each_user in range(NUM_USERS):

            if args.type == "A2C":
                state[each_user] = np.resize(state[each_user], [1,state_size])
                policy = actor.model.predict(state[each_user], batch_size=1).flatten()

            else:
                #feeding the input-history-sequence of (t-1) slot for each user seperately
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
                action[each_user] = np.random.choice(action_size, 1, p=policy)[0]
            else:
                action[each_user] = np.argmax(prob,axis=1)

            #action[each_user] = np.argmax(Qs,axis=1)
            if time_step % interval == 0:
                print("@@ state_vector")
                print(state_vector[:,each_user])
                if args.type != "A2C":
                    print("Qs")
                    print(Qs)
                    print("prob, sum of beta*Qs")
                    print(prob, np.sum(np.exp(beta*Qs)))

    # taking action as predicted from the q values and receiving the observation from thr envionment
    obs = env.step(action)           # obs is a list of tuple with [(ACK,REW) for each user ,(CHANNEL_RESIDUAL_CAPACITY_VECTOR)] 
    print("@ action :")
    print(action)
    print("@@@@@@@@@")
    print("@@ obs :")
    print(obs)
    print("@@@@@@@@@")

    # Generate next state from action and observation 
    next_state = state_generator(action,obs)
    print("@@@ next_state :")
    print(next_state)
    print("@@@@@@@@@")

    # reward for all users given by environment
    reward = [i[1] for i in obs[:NUM_USERS]]
    
    # calculating sum of rewards
    sum_r = np.sum(reward)
    print("$$$ sum_r :")
    print(sum_r)
    print("$$$$$$$$$$")

    # calculating cumulative reward
    cum_r.append(cum_r[-1] + sum_r)

    # If NUM_CHANNELS = 2 , total possible reward = 2 , therefore collision = (2 - sum_r) or (NUM_CHANNELS - sum_r)
    collision = NUM_CHANNELS - sum_r
    print("!!!!! collision :")
    print(collision)
    print("!!!!!!!!!!!!!!!!!!!!!!!!")

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
    print("$$$$ reward : ")
    print(reward)
    print("$$$$$$$$$$")
    '''
    print("$$$$$ total_rewards : ")
    print(total_rewards)
    print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
    '''
    #if time_step % 100 == 99:
        #if time_step < 5000:
            #print("***** every 100 time slots *****")
            # TODO:: measures and stores the mean reward score over that period

    means.append(cum_r[-1] / (time_step + 1))
    print("&&& means :")
    print(means)
    print("&&&&&&&&&&")
    all_means.append(means)
    '''
    print("&&&&& all_means :")
    print(all_means)
    print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
    '''
    # add new experiences into the memory buffer as (state, action , reward , next_state) for training
    if args.with_per:
        td_error = 1
    else:
        td_error = 0

    if args.with_per:
        memory.add(td_error, (state, action, reward, next_state))
    else:
        memory.add((state, action, reward, next_state))
    
    state = next_state
    #add new experience to generate input-history sequence for next state
    history_input.append(state)


    #  Training block starts
    ###################################################################################

    if args.type == "A2C":
        print("replay_memory length:", len(replay_memory))
        if len(replay_memory) < MINIMUM_REPLAY_MEMORY:
            continue
        if train_advantage_actor_critic(replay_memory, actor, critic) == False:
            print("##### train_advantage_actor_critic FALSE !! #####")
            continue
    elif args.type == "DDQN":
        train_ddqn(replay_memory)

    #  sampling a batch from memory buffer for training
    if args.with_per:
        batch, idxs, is_weights = memory.sample(batch_size, step_size)
    else:
        batch = memory.sample(batch_size, step_size)

    #   matrix of rank 4
    #   shape [NUM_USERS,batch_size,step_size,state_size]
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
    print("Before reshape")
    print("## shape of states : ", np.shape(states))
    print("## shape of states.shape[0] : ", states.shape[0])
    print("## shape of states.shape[1] : ", states.shape[1])
    print("## shape of states.shape[2] : ", states.shape[2])
    print("## shape of states.shape[3] : ", states.shape[3])
    if args.type != "A2C":
        states = np.reshape(states,[-1,states.shape[2],states.shape[3]])
        actions = np.reshape(actions,[-1,actions.shape[2]])
        rewards = np.reshape(rewards,[-1,rewards.shape[2]])
        next_states = np.reshape(next_states,[-1,next_states.shape[2],next_states.shape[3]])
    else:
        states = np.reshape(states,[-1,states.shape[1],states.shape[2]])
        actions = np.reshape(actions,[-1,actions.shape[2]])
        rewards = np.reshape(rewards,[-1,rewards.shape[2]])
        next_states = np.reshape(next_states,[-1,next_states.shape[1],next_states.shape[2]])
    print("After reshape")
    print("### shape of states : ", np.shape(states))
    print("### shape of states.shape[0] : ", states.shape[0])
    print("### shape of states.shape[1] : ", states.shape[1])
    print("### shape of states.shape[2] : ", states.shape[2])
    if args.type != "A2C":
        #  creating target vector (possible best action)
        target_Qs = sess.run(mainQN.output,feed_dict={mainQN.inputs_:next_states})

        #  Q_target =  reward + gamma * Q_next
        targets = rewards[:,-1] + gamma * np.max(target_Qs,axis=1)

        #  calculating loss and train using Adam  optimizer
        loss, _ = sess.run([mainQN.loss,mainQN.opt],
                                feed_dict={mainQN.inputs_:states,
                                mainQN.targetQs_:targets,
                                mainQN.actions_:actions[:,-1]})
    else:
        print("### No need to do sess.run in A2C model ###")
        print()
        print()
        print()
        print()

    #   Training block ends
    ########################################################################################
    
    #if time_step % 5000 == 4999:
    #if time_step % 1000 == 999:
if time_step % TIME_SLOTS == (TIME_SLOTS-1):
    print("##### PLOT start ! #####")
    plt.figure(1)
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
    plt.plot(np.mean(all_means, axis=0))
    if args.type == "A2C":
        plt.legend(["Best Possible", "ActorCritic"])
    plt.xlabel('Time Slot')
    plt.ylabel('Mean reward of all users')
    plt.show()

    greedy_scores = np.mean(all_means, axis=0)
    np.save("+++++ greedy_scores", greedy_scores)

    total_rewards = []
    means = []
    cum_r = [0]
    cum_collision = [0]

    if args.type == "DQN":
        saver.save(sess,'checkpoints/dqn-user.ckpt')
    elif args.type == "DRQN":
        saver.save(sess,'checkpoints/drqn-user.ckpt')
    elif args.type == "A2C":
        saver.save(sess,'checkpoints/actor-critic-user.ckpt')
    #print time_step,loss , sum(reward) , Qs

print("*************************************************")

   






