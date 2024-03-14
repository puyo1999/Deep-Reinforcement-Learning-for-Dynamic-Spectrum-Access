import numpy as np
import matplotlib.pyplot as plt
import yaml
import logging
logger = logging.getLogger(__name__)

with open('./config/config.yaml') as f:
    config = yaml.safe_load(f)

NUM_CHANNELS = config['num_channels']                               # Total number of channels
NUM_USERS = config['num_users']                                 # Total number of users

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
        print("@@@ get_states_user : ", user)
        states_per_user = []
        for each in batch:
            states_per_batch = []
            #step_cnt = 0
            print(f"@ get_states_user - each\n : {each}")
            for step_i in each:
                print(f"@ get_states_user - step_i\n : {step_i}")
                print(f"@ get_states_user - step_i[0][{user}]\n : {step_i[0][user]}")
                '''
                if step_cnt >= 1:
                    continue
                step_cnt += 1
                try:
                    states_per_step = step_i[0][user]
                except IndexError:
                    print(step_i)
                    print("-----------")
                    print("get_states_user error")
                    for i in batch:
                        print("i : ",i)
                        print("**********")
                    sys.exit()
                '''
                states_per_step = step_i[0][user]
                states_per_batch.append(states_per_step)
            states_per_user.append(states_per_batch)
        states.append(states_per_user)
    #print len(states)
    print("@ get_states_user - states\n : {}".format(states))
    return np.array(states)

def get_actions_user(batch):
    actions = []
    for user in range(NUM_USERS):
        actions_per_user = []
        for each in batch:
            print(f"@ get_actions_user - each\n : {each}")
            actions_per_batch = []
            for step_i in each:
                print(f"@ get_actions_user - step_i\n : {step_i}")
                print(f"@ get_actions_user - step_i[1][{user}]\n : {step_i[1][user]}")
                actions_per_step = step_i[1][user]
                actions_per_batch.append(actions_per_step)
            actions_per_user.append(actions_per_batch)
        actions.append(actions_per_user)
    print(f"@ get_actions_user - actions\n : {actions}")
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
    print("@ get_next_states_user - states : ", next_states)
    return np.array(next_states)


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

    logger.error(f'@ state_generator - input_vector:{input_vector}')
    return input_vector

def draw_res2(time_step, cum_collision, cum_r, loss_list, means, mean_scores, time_slots):
    if time_step % time_slots == time_slots-1:
        plt.figure(1)
        plt.subplot(411)
        # plt.plot(np.arange(1000),total_rewards,"r+")
        # plt.xlabel('Time Slots')
        # plt.ylabel('total rewards')
        # plt.title('total rewards given per time_step')
        # plt.show()
        plt.plot(np.arange(time_slots+1), cum_collision, "r-")
        plt.xlabel('Time Slot')
        plt.ylabel('cumulative collision')

        plt.subplot(412)
        plt.plot(np.arange(time_slots+1), cum_r, "b-")
        plt.xlabel('Time Slot')
        plt.ylabel('Cumulative reward of all users')

        plt.subplot(413)
        plt.plot(np.arange(len(means)), means, "c-")
        plt.xlabel('Time Slot')
        plt.ylabel('cumulative reward Means')

        plt.subplot(414)
        plt.plot(np.arange(len(loss_list)), loss_list, "g-")
        plt.xlabel('Time Slot')
        plt.ylabel('Loss')

        plt.show()

        total_rewards = []
        cum_r = [0]
        cum_collision = [0]
        #saver.save(sess, 'checkpoints/dqn_multi-user.ckpt')
        #print(time_step, loss, sum(reward), Qs)

    # print ("*************************************************")

def draw_multi_algorithm(data1, data2, data3):
    plt.xlabel('Time Slot')
    plt.ylabel('all means')

    t1 = np.arange(0, 1000, 1)
    t2 = np.arange(0, 5000, 1)
    t3 = np.arange(0, 100, 1)

    plt.plot(t1, data1, t2, data2, t3, data3, 'r-')
    plt.show()

