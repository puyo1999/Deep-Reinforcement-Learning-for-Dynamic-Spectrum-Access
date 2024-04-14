## 학습된 신경망 파라미터를 가져와서 에이전트를 실행

import gym
import tensorflow as tf
tf.executing_eagerly()
import tensorflow.compat.v1 as tf

import numpy as np
from model.A2C.a2c import A2C
from config.setup import NUM_USERS, NUM_CHANNELS, ATTEMPT_PROB
from config.setup import MEMORY_SIZE, ACTOR_LR, CRITIC_LR, WITH_PER
from env.multi_user_network_env import env_network
from util.prioritized_memory import PerMemory
from util.memory_buffer import Memory
from util.utils import state_generator
from util.utils import plot_rewards

state_size = 2 * (NUM_CHANNELS + 1)     #length of input (2 * k + 2)   :k = NUM_CHANNELS
action_size = NUM_CHANNELS+1            #length of output  (k+1)

def main():
    #env_name = "Pendulum-v0"
    #env = gym.make(env_name)

    env = env_network(NUM_USERS, NUM_CHANNELS, ATTEMPT_PROB)
    if WITH_PER:
        # memory = PerMemory(mem_size=args.memory_size, feature_size=NUM_USERS*2, prior=True)
        memory = PerMemory(MEMORY_SIZE, 2 * (NUM_CHANNELS + 1), True)
    else:
        memory = Memory(MEMORY_SIZE)

    sess = tf.Session()
    agent = A2C(env, sess, action_size, state_size, memory, ACTOR_LR, CRITIC_LR, WITH_PER)

    agent.load_weights('./save_weights/')

    time = 0
    state = env.reset()

    # cumulative reward
    cum_r = [0]
    rewards = []
    while True:
        #env.render()

        #action = agent.actor(tf.convert_to_tensor([state], dtype=tf.float32))[0][0]
        action = env.sample()
        print(f'@ load_play - action:\n{action}\n')

        #state, reward, done, _ = env.step(action)
        obs = env.step(action)
        state = state_generator(action, obs)
        reward = [i[1] for i in obs[:NUM_USERS]]

        time += 1

        sum_r = np.sum(reward)
        print(f'$$$ Before appending cur_r : cum_r[-1] = {cum_r[-1]}')
        cum_r.append(cum_r[-1] + sum_r)
        print(f'$$$ After appending cur_r : cum_r[-1] = {cum_r[-1]}')

        print(f"Time: {time}, cum_r: {cum_r}\n")

        #if done:
        if time == 200:
            break
    rewards = cum_r

    #env.close()

    print(f"All Rewards : {rewards}\n")
    #agent.plot_rewards(rewards)
    plot_rewards(rewards, time)


if __name__ == "__main__":
    main()