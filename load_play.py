## 학습된 신경망 파라미터를 가져와서 에이전트를 실행

import gym
import tensorflow as tf
tf.executing_eagerly()
import tensorflow.compat.v1 as tf

from py_lab.lib.logger import logger

import numpy as np
from model.A2C.a2c import A2C
from config.setup import NUM_USERS, NUM_CHANNELS, ATTEMPT_PROB
from config.setup import NUM_DT, NUM_HDMI, NUM_MF, NUM_CS
from config.setup import MEMORY_SIZE, ACTOR_LR, CRITIC_LR, WITH_PER, GAMMA
from env.multi_user_network_env import env_network
from env.mbr_detection_env import mbr_env
from util.prioritized_memory import PerMemory
from util.memory_buffer import Memory
from util.utils import state_generator
from util.utils import plot_rewards

import torch

state_size = 2 * (NUM_CHANNELS + 1)     #length of input (2 * k + 2)   :k = NUM_CHANNELS
#action_size = 2*(NUM_CHANNELS+1)            #length of output  (k+1)
action_size = NUM_CS+1

# 3️⃣ Reward 계산 함수 정의
def calculate_reward(state, action):
    """
    주어진 state와 action에 대해 보상을 계산하는 함수
    - 여기서는 간단한 예제 환경에서 reward를 임의로 정의
    """
    reward = np.sum(state) / (action + 1)  # 예제 보상 계산 방식 (상태 값에 따라 다르게 보상)

    return reward

def main():
    #env_name = "Pendulum-v0"
    #env = gym.make(env_name)

    #env = env_network(NUM_USERS, NUM_CHANNELS, ATTEMPT_PROB)
    env = mbr_env(NUM_HDMI, NUM_DT, NUM_MF, NUM_CS, ATTEMPT_PROB)
    if WITH_PER:
        # memory = PerMemory(mem_size=args.memory_size, feature_size=NUM_USERS*2, prior=True)
        memory = PerMemory(MEMORY_SIZE, 2 * (NUM_CHANNELS + 1), True)
    else:
        memory = Memory(MEMORY_SIZE)

    sess = tf.Session()
    #agent = A2C(env, sess, action_size, NUM_HDMI, memory, ACTOR_LR, CRITIC_LR, WITH_PER)
    agent = A2C(env, sess, 4, 9, memory, ACTOR_LR, CRITIC_LR, WITH_PER)

    #agent.load_weights('./save_weights/')


    # 저장된 모델 불러오기
    actor_model = agent.load_models("./save_weights/periodic_mbr_a2c_actor.h5")

    # 모델 구조 확인
    actor_model.summary()

    time = 0

    init_action = env.sample()
    init_obs = env.step(init_action)
    logger.error(f'@ load_play - init_obs :\n{init_obs}\n')

    init_state = state_generator(init_action, init_obs)
    logger.error(f'@ load_play - init_state:\n{init_state}\n')

    # 모델 재학습 수행
    actor_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                        loss="mse")  # 적절한 손실 함수 선택

    # cumulative reward
    cum_r = [0]
    rewards = []

    criterion = torch.nn.MSELoss()

    while True:
        #env.render()

        #action = agent.actor(tf.convert_to_tensor([init_state], dtype=tf.float32))[0][0]
        #action = env.sample()
        #action = env.sample_()

        if np.random.rand() < 0.1:
            action = env.sample_()
            logger.error(f'## random action:\n{action}')
        else:
            temp_state = np.array(init_state)
            temp_state = temp_state[:, 3:9]
            temp_state = temp_state.reshape(1, 6, 4)
            action_probs = actor_model.predict(temp_state)
            logger.error(f'## action_probs:\n{action_probs}')
            logger.error(f'## shape of action_probs:\n{np.shape(action_probs)}')

            #action = action_probs[0][0][0:4]
            #action = np.argmax(action_probs)
            # 가운데 4개의 행만 추출 (맨 위, 맨 아래 제외)
            middle_action_probs = action_probs[:, 1:-1, :]

            # 각 행에서 확률이 높은 4개의 action 인덱스 선택
            action = np.argmax(middle_action_probs, axis=-1).flatten()
            logger.error(f'## selected best action:\n{action}\nfrom action_probs:\n')

        logger.error(f'@ load_play - action: {action}\n')
        logger.error(f'@ load_play - shape of action: {np.shape(action)}\n')

        #state, reward, done, _ = env.step(action)
        obs = env.step(action)
        logger.error(f'@ load_play - obs: {obs}\n')

        state = state_generator(action, obs)

        new_state = state_generator(action, obs)

        reward = [i[1] for i in obs[:NUM_USERS]]
        logger.error(f'@ load_play - \nreward: {reward}\n')

        state = np.array(state)
        action = np.array(action)
        logger.error(f'@ load_play - before reshape\nstate: {state}\naction: {action}\n')

        # 필요한 크기만 사용하여 크기 조정
        state = state[:, 3:9]  # 열 개수를 줄여 (6, 4) 형태로 변환

        # 9 → 12로 맞춤
        # 48로 만드는 상황이 된다
        #state = np.pad(state, ((0, 0), (0, 3)), mode='constant')

        # 차원 변환 (모델과 일치하도록 reshape)
        state = state.reshape(1, 6, 4)  # (batch_size, action_dim, feature_dim)
        #action = action.reshape(-1, 1)  # (batch_size, output_dim)
        action = action[:state.shape[0]]

        logger.error(f'@ load_play - after reshape\nstate: {state}\naction: {action}\n')

        actor_model.fit(state, action, epochs=10, batch_size=4)

        # 4️⃣ Reward 계산 실행
        new_state = np.array(new_state)
        new_state = new_state[:, 3:9]
        new_state = new_state.reshape(1, 6, 4)

        new_action_probs = actor_model.predict(new_state)
        logger.error(f"????? new_action_probs:\n{new_action_probs}")

        predicted_action = np.argmax(actor_model.predict(new_state), axis=-1 ).flatten()  # 모델이 예측한 행동
        logger.error(f"????? predicted_action:\n{predicted_action}")

        # 선택된 행동의 확률 추출
        new_action_prob = new_action_probs[0, predicted_action]
        logger.error(f"????? new_action_prob:\n{new_action_prob}")
        # 로그 확률 계산
        log_prob = tf.math.log(new_action_prob + 1e-10)  # 로그 연산의 안정성 확보
        logger.error(f"????? log_prob:\n{log_prob}")

        actor_reward = calculate_reward(new_state[0], predicted_action)
        logger.error(f'@ load_play - \nactor_reward: {actor_reward}\n')

        # 보상 기반의 policy gradient loss
        loss = -log_prob * actor_reward
        logger.error(f"????? Policy Loss:\n{loss.numpy()}")

        sum_r = np.sum(reward)
        logger.error(f'$$$ Before appending cur_r : cum_r[-1] = {cum_r[-1]}')
        cum_r.append(cum_r[-1] + sum_r)
        logger.error(f'$$$ After appending cur_r : cum_r[-1] = {cum_r[-1]}')

        logger.error(f"Time: {time}, cum_r: {cum_r}\n")

        # 오차 계산을 여기선 할 필요가 없는게 이건 이미 학습한 기반으로 play 하는 코드라서,

        time += 1

        #if done:
        if time == 200:
            break
    rewards = cum_r

    # 학습된 모델 저장 (필요할 경우)
    actor_model.save("./save_weights/updated_actor_model.h5")

    #env.close()

    print(f"All Rewards : {rewards}\n")
    #agent.plot_rewards(rewards)
    plot_rewards(rewards, time)



if __name__ == "__main__":
    main()