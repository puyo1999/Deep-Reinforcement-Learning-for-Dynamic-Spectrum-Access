import numpy as np
import random
import itertools

class mbr_env:
    def __init__(self, num_hdmi, num_dt, num_mf, num_cs, attempt_prob):
        self.ATTEMPT_PROB = attempt_prob
        self.NUM_HDMI = num_hdmi
        self.NUM_DT = num_dt    # device type
        self.NUM_MF = num_mf    # manufacturer
        self.NUM_CS = num_cs    # code set

        self.REWARD = 1
        self.DOUBLE_REWARD = 2
        print(f'@@ Mbr Env init : {self.NUM_CS}')
        self.action_space = np.arange(self.NUM_CS+1)

        self.hdmi_action = np.zeros([self.NUM_HDMI], np.int32)
        self.hdmi_observation = np.zeros([self.NUM_HDMI], np.int32)
        print(f'@@ Mbr Env init -\n'
              f'action_space : {self.action_space}\n'
              f'hdmi_action : {self.hdmi_action}\n'
              f'hdmi_observation:{self.hdmi_observation}\n')

    def reset(self):
        pass

    def sample(self):
        #x = np.random.choice(self.action_space, size=self.NUM_HDMI)
        x = np.random.choice(self.action_space, size=self.NUM_HDMI)
        print(f'##### Mbr Env @@@@@ sample x:{x} from action_space:{self.action_space}')
        return x

    def step(self, action):
        print(f'action.size: {action.size}, NUM_HDMI: {self.NUM_HDMI}')
        assert (action.size) == self.NUM_HDMI, "action and user should have same dim action.size:\n{}".format(action.size)
        print(f'NUM_CS: {self.NUM_CS}')
        codeset_alloc_frequency = np.zeros([self.NUM_CS + 1],np.int32)  #0 for no codeset access
        obs = []
        reward = np.zeros([self.NUM_HDMI])
        j = 0
        for each in action:
            prob = random.uniform(0,1)
            print(f'@@ In loop step[{each}] - prob: {prob}')
            if prob <= self.ATTEMPT_PROB:
                self.hdmi_action[j] = each  # action
                codeset_alloc_frequency[each] += 1
            j += 1

        print(f'@ step - original codeset_alloc_frequency:\n{codeset_alloc_frequency}')

        # combination on codeset list
        #codeset_comb = list(itertools.combinations(codeset_alloc_frequency, 3))
        #print(f'@ step - codeset_comb: {codeset_comb}')

        for i in range(1, len(codeset_alloc_frequency)):
            if codeset_alloc_frequency[i] > 5:
                codeset_alloc_frequency[i] = 5

        # print codeset_alloc_frequency
        print(f'@ step - modified codeset_alloc_frequency:\n{codeset_alloc_frequency}')
        for i in range(len(action)):
            print(f'@@ In loop action range[{i}]')
            self.hdmi_observation[i] = codeset_alloc_frequency[self.hdmi_action[i]]

            print(f'@@@ step - hdmi_action[{i}] : {self.hdmi_action[i]}')
            print(f'@@@ step - hdmi_observation[{i}] : {self.hdmi_observation[i]}')

            if self.hdmi_action[i] == 0:  # accessing no codeset
                print(f'$ step - reward[{i}] : 0')
                self.hdmi_observation[i] = 0
                reward[i] = 0
            if self.hdmi_observation[i] == 1:
                print(f'$ step - reward[{i}] : 1')
                reward[i] = 1
            elif self.hdmi_observation[i] == 2:
                print(f'$ step - reward[{i}] : 2')
                reward[i] = 2
            elif self.hdmi_observation[i] == 3:
                print(f'$ step - reward[{i}] : 3')
                reward[i] = 3
            elif self.hdmi_observation[i] == 4:
                print(f'$ step - reward[{i}] : 4')
                reward[i] = 4
            elif self.hdmi_observation[i] == 5:
                print(f'$ step - reward[{i}] : 5')
                reward[i] = 5
            obs.append((self.hdmi_observation[i], reward[i]))

        print(f'% observation: {obs}')
        return obs

    def close(self):
        pass