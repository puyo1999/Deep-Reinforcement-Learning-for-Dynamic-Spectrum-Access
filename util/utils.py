import numpy as np
import matplotlib.pyplot as plt
import yaml
import logging
from config.setup import *
logger = logging.getLogger(__name__)

with open('D:/Research Files/DRLforDSA/Deep-Reinforcement-Learning-for-Dynamic-Spectrum-Access/config/config.yaml') as f:
    config = yaml.safe_load(f)

env = config["env"]
if env == "network":
    NUM_CHANNELS = dic_env_conf["NUM_CHANNELS"]
elif env == "mbr":
    NUM_CHANNELS = dic_env_conf["NUM_CS"]
else:
    NUM_CHANNELS = dic_env_conf['NUM_CHANNELS']                                  # Total number of channels

if env == "network":
    NUM_USERS = dic_env_conf["NUM_USERS"]
elif env == "mbr":
    NUM_USERS = dic_env_conf["NUM_HDMI"]
else:
    NUM_USERS = dic_env_conf["NUM_USERS"]
#NUM_USERS = config['num_users']                                 # Total number of users

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
        #print(f"@@@ get_states_user - user:{user}")
        states_per_user = []
        for each in batch:
            states_per_batch = []
            #step_cnt = 0
            #print(f"@ get_states_user - each :\n{each}")
            for step_i in each:
                #print(f"@ get_states_user - step_i :\n{step_i}")
                #print(f"@ get_states_user - step_i[0][{user}]\n : {step_i[0][user]}")
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
                states_per_step = []
                states_per_step = step_i[0][user]
                states_per_batch.append(states_per_step)
            states_per_user.append(states_per_batch)
        states.append(states_per_user)
    #print len(states)
    #print("@ get_states_user - states\n : {}".format(states))
    return np.array(states)

def get_actions_user(batch):
    actions = []
    for user in range(NUM_USERS):
        actions_per_user = []
        for each in batch:
            #print(f"@ get_actions_user - each\n : {each}")
            actions_per_batch = []
            for step_i in each:
                #print(f"@ get_actions_user - step_i\n : {step_i}")
                #print(f"@ get_actions_user - step_i[1][{user}]\n : {step_i[1][user]}")
                actions_per_step = step_i[1][user]
                actions_per_batch.append(actions_per_step)
            actions_per_user.append(actions_per_batch)
        actions.append(actions_per_user)
    #print(f"@ get_actions_user - actions\n : {actions}")
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
    print(f"@ get_next_states_user - next_states :\n{next_states}")
    return np.array(next_states, dtype=object)


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
    logger.critical(f'action.size:{action.size}')
    for user_i in range(action.size):
        logger.critical(f'user_i:[{user_i}] action : {action[user_i]}')
        input_vector_i = one_hot(action[user_i], NUM_CHANNELS+1) # mbr - 6 개
        logger.critical(f'1st input_vector_i: {input_vector_i}')
        #channel_alloc = obs[-1] # obs 뒤에서 첫번째
        channel_alloc = obs[user_i] # 해당 사용자의 obs
        input_vector_i = np.append(input_vector_i,channel_alloc)
        input_vector_i = np.append(input_vector_i,int(obs[user_i][0]))    #ACK

        '''
        if env == "mbr":
            input_vector_i = np.append(input_vector_i,channel_alloc)    #
            input_vector_i = np.append(input_vector_i,int(obs[user_i][0]))    #
        '''

        input_vector.append(input_vector_i)

    logger.critical(f'@ state_generator - input_vector:{input_vector}\n'
                 f'input_vector shape:{np.shape(input_vector)}')
    return input_vector
def draw_res(time_step, cum_collision, cum_r, loss_list, time_slots):
    logger.info(f'@ time_step - time_step:{time_step}, time_slot:{time_slots}')
    if time_step % time_slots == time_slots-1:
        plt.figure(1)
        plt.subplot(311)
        plt.plot(np.arange(time_slots+1), cum_collision, "r-")
        plt.xlabel('Time Slot')
        plt.ylabel('cumulative collision')

        plt.subplot(312)
        plt.plot(np.arange(time_slots+1), cum_r, "b-")
        plt.xlabel('Time Slot')
        plt.ylabel('Cumulative reward of all users')


        plt.subplot(313)
        plt.plot(np.arange(len(loss_list)), loss_list, "g-")
        plt.xlabel('Time Slot')
        plt.ylabel('Loss')

        plt.show()

'''
x = np.arange(0, 2, 0.2)
'''
fontdict = {
    'fontsize': 16,
    'fontweight': 'normal',
    'color': 'black',
    'family': 'Arial',
    'style': 'normal'
}
def draw_res2(time_step, cum_collision, cum_r, loss_list, means, mean_scores, time_slots):
    logger.info(f'@ draw_res2 - time_step:{time_step}, time_slot:{time_slots}')
    #if time_step % time_slots == time_slots-1:
    plt.figure(1)
    plt.subplot(411)
    # plt.plot(np.arange(1000),total_rewards,"r+")
    # plt.xlabel('Time Slots')
    # plt.ylabel('total rewards')
    # plt.title('total rewards given per time_step')
    # plt.show()
    plt.plot(np.arange(time_step+2), cum_collision, color='r', marker='v', markersize=5, markevery=10)
    #plt.xlabel('Time Slot')
    plt.ylabel('Cumulative Collision')

    plt.subplot(412)
    plt.plot(np.arange(len(cum_r)), cum_r, color='b', marker='D', markersize=5, markevery=10)
    #plt.xlabel('Time Slot')
    plt.ylabel('Cumulative Reward')

    plt.subplot(413)
    #plt.plot(np.arange(len(means)), means, "c.-", markevery=5)
    plt.plot(np.arange(len(mean_scores)), mean_scores, color='m', marker='o', markersize=5, markevery=10)
    #plt.xlabel('Time Slot')
    plt.ylabel('Mean Reward')

    plt.subplot(414)
    plt.grid(True, linestyle='--')
    plt.plot(np.arange(len(loss_list)), loss_list, color='g', marker='h', markersize=5, markevery=10)
    plt.xlabel('Time Slot', fontdict=fontdict)
    plt.ylabel('Loss')

    #plt.tight_layout()
    plt.grid(True, linestyle='--')
    plt.subplots_adjust(hspace=0.1, wspace=0.5)
    plt.show()

    #total_rewards = []
    #cum_r = [0]
    #cum_collision = [0]
    #saver.save(sess, 'checkpoints/dqn_multi-user.ckpt')
    #print(time_step, loss, sum(reward), Qs)

    print ("*************************************************")

def draw_res3(time_step, cum_collision, cum_r, loss_list, time_slots):
    logger.info(f'@ draw_res3 - time_step:{time_step}, time_slot:{time_slots}')
    if time_step % time_slots == time_slots-1:
        plt.figure(1)
        plt.subplot(311)
        plt.plot(np.arange(time_slots+1), cum_collision, "r-")
        plt.xlabel('Time Slot')
        plt.ylabel('cumulative collision')

        plt.subplot(312)
        plt.plot(np.arange(time_slots+1), cum_r, "b-")
        plt.xlabel('Time Slot')
        plt.ylabel('Cumulative reward of all users')

        plt.subplot(313)
        plt.plot(np.arange(len(loss_list)), loss_list, "g-")
        plt.xlabel('Time Slot')
        plt.ylabel('Loss')

        plt.show()
def draw_losses(time, loss1, loss2):
    t1 = np.arange(0., time, 1)

    fig = plt.figure(figsize=(15, 10))  ## 캔버스 생성
    fig.set_facecolor('white')  ## 캔버스 색상 설정

    al = fig.add_subplot(211)  ## 그림 뼈대(프레임) 생성, 2x1 matrix 의 1행
    plt.ylabel('Actor loss')

    cl = fig.add_subplot(212)  # 2x1 matrix 의 2행
    plt.xlabel('Time Slot')
    plt.ylabel('Critic loss')

    plt.title('Actor vs Critic Loss Over Time', fontsize=16, fontweight='bold')
    # actor loss
    al.plot(
        t1, loss1,
        'rv-',  # 빨간색 삼각형 마커
        markersize=3,
        markevery=10,
        linewidth=1,
        label='actor loss'
    )

    # critic loss
    cl.plot(
        t1, loss2,
        'bD-',  # 파란색 다이아몬드 마커
        markersize=3,
        markevery=10,
        linewidth=1,
        label='critic loss'
    )

    plt.legend(loc="upper left")
    plt.grid(alpha=0.3)

    plt.show()

def draw_multi_algorithm(data1, data2, data3, data4):
    plt.xlabel('Time Slot')
    plt.ylabel('all means')

    #t1 = np.arange(0, 501, 10)
    t1 = np.arange(0., 501., 1)
    t2 = np.arange(0, 1001, 1)
    t3 = np.arange(0, 100, 1)

    # evenly sampled time at 200ms intervals
    t4 = np.arange(0., 5., 0.2)

    # red dashes, blue squares and green triangles
    plt.plot(t1, data1, 'r--', t2, data2, 'bs', t3, data3, 'g^')

    #plt.plot(t1, data1, t2, data2, t3, data3, 'r-')

    #plt.plot(t1, data1, marker="o", color="#0000FF", label="$a2c$")
    #plt.plot(t2, data2, 'c^', color="#00FF00", label="$drqn$")
    #plt.plot(t3, data3, "d--", color="#FF0000", label="$ddqn$")  # TODO: Fix the notation here
    plt.legend(loc="upper left")

    plt.title("Average Reward", fontsize=16, fontweight='bold')
    plt.show()


def plot_rewards(rewards, time):
    # evenly sampled time at 1s intervals
    t1 = np.arange(0, time+1, 1)

    # red dashes, blue squares and green triangles
    #plt.plot(t1, rewards, 'bs')
    plt.plot(t1, rewards, "r.-", markevery=5)
    plt.xlabel('Time Slot')
    plt.ylabel('Cumulative reward of trained model')

    plt.show()

