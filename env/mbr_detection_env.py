import numpy as np
import random
import itertools
from py_lab.lib import logger
logger = logger.get_logger(__name__)

cec_brand = ['samsung', 'apple', 'sony', 'microsoft', 'roku', 'kt', 'sk', 'lg', 'amazon', 'google']
codeset_map = []

# candidate_codeset 를 초기화하기 위한 함수
def initialize_candidate_codeset(brand_list, num_candidates=5):
    """
    각 브랜드마다 num_candidates 개수의 후보 코드를 생성합니다.
    후보 코드는 예시로 "BRAND_번호" 형식의 문자열로 구성되며,
    각 브랜드별로 무작위 순서로 섞여 저장됩니다.
    """
    codeset = {}
    for brand in brand_list:
        # 예를 들어, "SAMSUNG_1", "SAMSUNG_2", ..., "SAMSUNG_5" 생성 후 섞기
        code_list = [f"{brand.upper()}_{i}" for i in range(1, num_candidates + 1)]
        random.shuffle(code_list)
        codeset[brand] = code_list
    return codeset

# 특정 브랜드의 candidate_codeset 업데이트를 위한 함수
def update_candidate_codeset_for_brand(codeset, brand, new_codes):
    """
    주어진 브랜드의 candidate code 리스트를 new_codes 로 업데이트합니다.
    브랜드가 존재하지 않으면 ValueError 를 발생시킵니다.
    """
    if brand in codeset:
        codeset[brand] = new_codes
    else:
        raise ValueError(f"Brand {brand} is not in candidate_codeset.")


class mbr_env:
    def __init__(self, num_hdmi, num_dt, num_mf, num_cs, attempt_prob):
        self.ATTEMPT_TIME = 30
        self.ATTEMPT_PROB = attempt_prob
        self.NUM_HDMI = num_hdmi
        self.NUM_DT = num_dt    # device type
        self.NUM_MF = num_mf    # manufacturer
        self.NUM_CS = num_cs    # code set

        self.REWARD = 1
        self.DOUBLE_REWARD = 2
        logger.info(f'@@ Mbr Env init : {self.NUM_CS}')
        self.action_space = np.arange(self.NUM_CS+1)

        self.hdmi_action = np.zeros([self.NUM_HDMI], np.int32)
        self.hdmi_observation = np.zeros([self.NUM_HDMI], np.int32)
        self.hdmi_detection_time = np.zeros([self.NUM_CS+1], np.int32)

        self.cec_brand_name = [0] * 10 # 10개 대표 브랜드에 대해서 고려
        self.cec_brand_index = np.zeros([self.NUM_HDMI], np.int32)
        self.cec_brand_name = list(cec_brand)
        for i in range(self.NUM_HDMI):
            #self.cec_brand_index[i] = random.randint(1, self.NUM_MF)
            self.cec_brand_index[i] = random.randint(1, self.NUM_CS) # 4개의 HDMI 포트 연결 기기에 대해서, 정답 생성?

        # 초기 candidate_codeset 생성 (각 브랜드마다 5개의 후보 코드 생성)
        candidate_codeset = initialize_candidate_codeset(cec_brand, num_candidates=5)

        # 초기 candidate_codeset 출력
        logger.info("초기 Candidate Codeset:")
        for brand, codes in candidate_codeset.items():
            logger.info(f"{brand}: {codes}")

        logger.info(f'@@ Mbr Env init -\n'
              f'action_space : {self.action_space}\n'
              f'hdmi_action : {self.hdmi_action}\n'
              f'hdmi_observation:{self.hdmi_observation}\n'
              f'cec_brand_name:{self.cec_brand_name}\n'
              f'cec_brand_index:{self.cec_brand_index}\n') # cec_brand_index:[5 1 3 6]

    def reset(self):
        logger.info(f'@@ Mbr Env reset - Do nothing now !!')
        pass

    def sample_(self):
        x = np.random.choice(self.action_space, size=self.NUM_HDMI, replace=False)
        logger.info(f'##### Mbr Env @@@@@ sample x:{x} from action_space:{self.action_space}')
        return x

    def sample(self):
        x = np.random.choice(self.action_space, size=self.NUM_HDMI)
        logger.info(f'##### Mbr Env @@@@@ sample x:{x} from action_space:{self.action_space}')
        return x

    def step(self, action):
        logger.info(f'action.size: {action.size}, NUM_HDMI: {self.NUM_HDMI}')
        assert (action.size) == self.NUM_HDMI, "action and user should have the same dim action.size:\n{}".format(action.size)
        logger.info(f'@ step - action: {action}, NUM_CS: {self.NUM_CS}')

        # 이차원 배열로 생성해서 HDMI 별로 코드셋 최대 개수 주기 설정해야 함

        #codeset_alloc_frequency = np.zeros([self.NUM_CS + 1],np.int32)  # 0 for no codeset access
        codeset_alloc_frequency = [[0] * (self.NUM_CS + 1) for _ in range(self.NUM_HDMI)]
        #codeset_alloc_order = [list(range(1, self.NUM_CS+1)) for _ in range(self.NUM_HDMI)]
        codeset_alloc_order = list(range(1, self.NUM_CS+1))
        time_elapsed = np.ones([self.NUM_CS+1], np.int32)
        logger.info(f'@ step - shape of codeset_alloc_frequency: {np.shape(codeset_alloc_frequency)}')
        logger.info(f'@ step - initial codeset_alloc_frequency:\n{codeset_alloc_frequency}')

        obs = []
        reward = np.zeros([self.NUM_HDMI])
        j = 0
        for each in action:

            prob = random.uniform(0,1)
            logger.info(f'@@ In loop step[{each}] - prob: {prob}')

            ## TODO:: 코드셋 list 변경 시도 확률 수정해 가면서 시뮬레이션 필요
            if prob <= self.ATTEMPT_PROB:
                self.hdmi_action[j] = each  # action

                #codeset_alloc_frequency[each] += 1
                codeset_alloc_frequency[j][each] += 1
                #codeset_alloc_frequency[each][j] += 1

            time = random.randint(3, 60)
            logger.info(f'@@ time: {time}')
            time_elapsed[each] = time

            #map_cec_brand = list(map(int, cec_brand))
            #idx = cec_brand.index(self.cec_brand_int[each])
            #self.cec_brand_int[each] = map_cec_brand[each]
            #logger.info(f'@@ cec_brand_index[{each}] : {self.cec_brand_index[each]}')

            j += 1

        logger.info(f'@ step - codeset_alloc_order:\n{codeset_alloc_order}')

        logger.info(f'@ step - original codeset_alloc_frequency:\n{codeset_alloc_frequency}')
        logger.info(f'@ step - time_elapsed:\n{time_elapsed}')
        logger.info(f'@ step - cec_brand_index:\n{self.cec_brand_index}')
        '''
        @ step - original codeset_alloc_frequency:
        [3 0 0 1 0 0]
        @ step - time_elapsed:
        [43  1  1 34  1  1]
        @ step - cec_brand_index:
        [0 1 2 3]
        '''
        # combination on codeset list
        #codeset_comb = list(itertools.combinations(codeset_alloc_frequency, 3))
        #logger.info(f'@ step - codeset_comb: {codeset_comb}')

        '''
        for i in range(1, len(codeset_alloc_frequency)):
            if codeset_alloc_frequency[i] > 5:
                codeset_alloc_frequency[i] = 5
        '''

        for i in range(len(action)):
            logger.info(f'@@ In loop action range[{i}]')
            self.hdmi_observation[i] = codeset_alloc_frequency[i][self.hdmi_action[i]]
            #self.hdmi_observation[i] = codeset_alloc_frequency[self.hdmi_action[i]][i]
            self.hdmi_detection_time[i] = time_elapsed[i]
            idx = cec_brand.index(self.cec_brand_name[i]) # predefined mapping is needed
            self.cec_brand_index[i] = idx

            logger.info(f'@@@ step - hdmi_action[{i}] : {self.hdmi_action[i]}')
            logger.info(f'@@@ step - hdmi_observation[{i}] : {self.hdmi_observation[i]}')
            logger.info(f'@@@ step - hdmi_detection_time[{i}] : {self.hdmi_detection_time[i]}')
            logger.info(f'@@@ step - cec_brand_index[{i}] : {self.cec_brand_index[i]}')
            # 기기 인식에 걸린 시간을 기준으로 보너스 보상 계산
            bonus_reward = (1 / (time_elapsed[i] + 1))
            #bonus_reward = 0.21
            logger.info(f'@@@ step - bonus_reward : {bonus_reward}')
            if self.hdmi_action[i] == 0:  # accessing no codeset
                self.hdmi_observation[i] = 0
                reward[i] = 0 + bonus_reward
                logger.info(f'$ step : accessing no codeset - only bonus reward[{i}] : {reward[i]}')
            if self.hdmi_observation[i] == 1:
                reward[i] = 0.1 + bonus_reward
                logger.info(f'$ step : obs 1 - reward[{i}] : {reward[i]}')
            elif self.hdmi_observation[i] == 2:
                reward[i] = 0.2 + bonus_reward
                logger.info(f'$ step : obs 2 - reward[{i}] : {reward[i]}')
            elif self.hdmi_observation[i] == 3:
                reward[i] = 0.3 + bonus_reward
                logger.info(f'$ step : obs 3 - reward[{i}] : {reward[i]}')
            elif self.hdmi_observation[i] == 4:
                reward[i] = 0.4 + bonus_reward
                logger.info(f'$ step : obs 4 - reward[{i}] : {reward[i]}')
            elif self.hdmi_observation[i] == 5:
                reward[i] = 0.5 + bonus_reward
                logger.info(f'$ step : obs 5 - reward[{i}] : {reward[i]}')

            # reward = (R_max - R_min) * e... + R_min

            # cec brand 잘 맞았다면, 높은 보상을 주자.

            logger.info(f'☆★☆★ step - cec_brand_index[{i}] : {self.cec_brand_index[i]}')
            logger.info(f'☆★☆★ step - hdmi_observation[{i}] : {self.hdmi_observation[i]}')
            if self.cec_brand_index[i] == self.hdmi_observation[i]:
                reward[i] = reward[i] * 1.2
                logger.info(f'☆★☆★ step - 20% more reward[{i}] : {reward[i]}')

            ## TODO:: cec_brand 와 안맞은 경우에 수동으로 codeset 순서를 변경하는 policy 필요
            else:
            # codeset_alloc_frequency
            # logger.info(f'@ step - reordered codeset_alloc_frequency:\n{codeset_alloc_frequency}')
                reward[i] = reward[i] * 0.5
                logger.info(f'♨♨♨♨ step - half reward[{i}] : {reward[i]}')

                self.hdmi_observation[i] = codeset_alloc_order[i]

                #row = codeset_alloc_order[i]
                #start = row.index(i)
                #for offset in range(self.NUM_CS - start):
                #    row[start + offset] = offset + 1

            # 추가적인 개선 제안으로 복합 리워드에 아래 식도 활용해보자.
            '''
            r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
            r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
            reward = r1 + r2
            '''

            obs.append((self.hdmi_observation[i], reward[i]))

        logger.info(f'☆★☆★ step - saved observation: {obs}')
        return obs

    def close(self):
        pass