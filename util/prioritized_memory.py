import random
import numpy as np
from py_lab.lib import logger
logger = logger.get_logger(__name__)

from collections import deque
from .sumtree import SumTree

import config.setup as cs

class PerMemory(object):
    """ Memory Buffer Helper class for Experience Replay
    using a double-ended queue or a Sum Tree (for PER)
    """
    p_upper = 1.
    e = .01
    alpha = .6
    alpha_decrement_per_sampling = .002
    beta = .4
    beta_increment_per_sampling = .001
    recent_sampling_weight = 1.2

    # feature_size = state_size 이므로 2 * (NUM_CHANNELS + 1)
    def __init__(self, mem_size, feature_size, prior=False):
        """ Initialization
        """
        logger.info("@ PerMemory : init - feature_size = ", feature_size)
        self.prior = prior
        if cs.env == "mbr":
            self.data_len = feature_size * 5 + 2
        else:
            self.data_len = 6 * feature_size + 6
        if cs.TYPE == "DDQN":
            self.data_len = feature_size * 6 + 8

        #self.data_len = feature_size
        logger.info("@ PerMemory : init - prior:{} data_len:{}".format(self.prior ,self.data_len))

        # Prioritized Experience Replay
        if prior:
            self.tree = SumTree(mem_size, self.data_len)

        else:
            self.mem_size = mem_size
            self.mem = np.zeros(mem_size, self.data_len, dtype=np.float32)
            self.mem_ptr = 0

        self.batch_gae_r = []
        self.GAE_CALCULATED_Q = False

    def store(self, transition):
        if self.prior:
            p = self.tree.max_p()
            logger.critical(f'PER store - with p(max_p): {p}\n')
            if p == 0:
                p = self.p_upper
            logger.critical(f"PER store - p:{p}")
            logger.info(f"PER store - transition:\n{transition}")
            self.tree.add(p, transition)
            #logger.info('PER @ store - check min_p:{}\n'.format(self.tree.min_p))
        else:
            self.mem[self.mem_ptr] = transition
            self.mem_ptr += 1
            if self.mem_ptr == self.mem_size:
                self.mem_ptr = 0

    def add2(self, transition):
        max_p = np.max(self.tree.tree[-self.tree.size])
        logger.info(f'PER @ add2 - max_p: {max_p}')
        if max_p == 0:
            max_p = self.p_upper
        self.tree.add(max_p,transition)

    def add(self, sample, error):
        """ Save an experience to memory, optionally with its TD-Error
        """
        if self.prior:
            p = self._get_priority(error)
            logger.info(f'PER @ add - p: {p}, error:{error}, type of p: {type(p)}')
            self.tree.add(p, sample)
        else:
            self.mem[self.mem_ptr] = sample
            self.mem_ptr += 1
            if self.mem_ptr == self.mem_size:
                self.mem_ptr = 0
        #logger.info('PER @ add - check min_p:{}\n'.format(self.tree.min_p))

    def _get_priority(self, error):
        """ Compute an experience priority, as per Schaul et al.
        """
        #logger.info('PER @ _get_priority : {}'.format(np.power(error + self.e, self.a).squeeze()))
        #return np.power(error + self.e, self.a).squeeze()
        self.a = np.max([1., self.alpha - self.alpha_decrement_per_sampling])
        logger.info(f'PER @ _get_priority : {(abs(error) + self.e) ** self.a}')
        return (abs(error) + self.e) ** self.a


    def sample(self, n):
        """ Sample a batch, optionally with (PER)
        """
        if self.prior:
            #min_p = self.tree.min_p
            #logger.info('PER @ sample : check current min_p : {}'.format(min_p))

            segment = self.tree.total() / n
            #batch = np.zeros((n, self.data_len), dtype=np.float32)
            batch = []
            #w = np.zeros((n,1), np.float32)
            #idx = np.zeros(n, np.int32)
            idxs = []
            a = 0
            priorities = []
            self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])

            for i in range(n):
                #b = a + segment
                a = segment * i
                b = segment * (i + 1)
                v = np.random.uniform(a, b)
                #idx[i], p, batch[i] = self.tree.get(v)
                (idx, p, data) = self.tree.get(v)
                #logger.info(f'@@ sample idx:{idx}')
                priorities.append(p)
                batch.append(data)
                idxs.append(idx)

                #w[i] = (p / min_p) ** (-self.beta)
                a += segment

            #self.beta = min(1., self.a + .01)
            sampling_probabilities = priorities / self.tree.total()
            is_weight = np.power(self.tree.write * sampling_probabilities, -self.beta)
            is_weight /= is_weight.max()
            return idxs, is_weight, batch
        else:
            mask = np.random.choice(range(self.mem_size), n)
            return self.mem[mask]
        '''
        batch = []
        idxs = []
        segment = self.tree.total() / n
        priorities = []
        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])

        # Sample using priorities

        for i in range(n):
            temp_buffer = []
            a, b = segment * i, segment * (i + 1)
            s = random.uniform(a, b)
            idx, p, data = self.tree.get(s)
            for j in range(step_size):
                temp_buffer.append(data)
            priorities.append(p)
            #batch.append(data)
            batch.append(temp_buffer)
            idxs.append(idx)

        sampling_probabilities = priorities / self.tree.total()
        is_weight = np.power(self.tree.n_entries * sampling_probabilities, -self.beta)
        is_weight /= is_weight.max()

        return batch, idxs, is_weight
        #return batch
        '''
    def sample_ere(self, c_k):
        segment = self.tree.total() / c_k
        batch = []
        idxs = []
        a = 0
        priorities = []
        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])

        for i in range(c_k):
            a = segment * i
            b = segment * (i + 1)
            v = np.random.uniform(a, b)
            (idx, p, data) = self.tree.get(v)
            #logger.info(f'@@ sample idx:{idx}')
            priorities.append(p)
            batch.append(data)
            idxs.append(idx)

            a += segment

        prios = np.array(list(priorities)[:c_k])
        probs = (prios* self.recent_sampling_weight) ** self.alpha

        sampling_probabilities = probs / self.tree.total()
        is_weight = np.power(self.tree.n_entries * sampling_probabilities, -self.beta)
        is_weight /= is_weight.max()
        return idxs, is_weight, batch

    def get_prioritized_indexes(self, batch_size):
        '''3. TD Error에 따른 확률로 index 추출'''

        # TD Error의 총 절댓값 합 계산
        sum_absolute_td_error = np.sum(np.absolute(self.memory))
        sum_absolute_td_error += TD_ERROR_EPSILON * len(self.memory)  # 각 transition마다 충분히 작은 epsiolon을 더함

        # [0, sum_absolute_td_error] 구간의 batch_size개 만큼 난수 생성
        rand_list = np.random.uniform(0, sum_absolute_td_error, batch_size)
        rand_list = np.sort(rand_list)  # batch_size개의 생성한 난수를 오름차순으로 정렬

        # 위에서 만든 난수로 index 결정
        indexes = []
        idx = 0
        tmp_sum_absolute_td_error = 0
        for rand_num in rand_list:  # 제일 작은 난수부터 꺼내기
            # 각 memory의 td-error 값을 더해가면서, 몇번째 index
            while tmp_sum_absolute_td_error < rand_num:
                tmp_sum_absolute_td_error += (
                        abs(self.memory[idx]) + TD_ERROR_EPSILON)
                idx += 1

            # TD_ERROR_EPSILON을 더한 영향으로 index가 실제 개수를 초과했을 경우를 보정
            if idx >= len(self.memory):
                idx = len(self.memory) - 1
            indexes.append(idx)

        return indexes


    def importance_sampling(self, x, pi):
        b = np.array([0.2, 0.2, 0.6]) # 확률 분포 변경, 두 확률 분포를 비슷하게 mbr codeset 기반의 새로운 확률 분포 생성

        n = self.batch_size
        samples = []

        for _ in range(n):
            idx = np.arange(len(b))
            i = np.random.choice(idx, p=b)
            s = x[i]
            rho = pi[i]/b[i]
            samples.append(s*rho)

        return samples

    def update(self, idx, tderr):
        """ Update priority for idx (PER)
        """
        if self.prior:
            for i in range(len(idx)):
                '''
                tderr[i] += self.e
                tderr[i] = np.minimum(tderr[i], self.p_upper)
                self.tree.update(idx[i], tderr[i] ** self.a)
                '''
                priorities = self._get_priority(tderr[i])
                self.tree.update(idx[i], priorities)

                '''
                p = self._get_priority(tderr[i])
                self.tree.update(idx, p)
                '''

    def update_priorities(self, batch_indices, batch_priorities):
        if self.prior:
            for idx, prio in zip(batch_indices, batch_priorities):
                priorities = self._get_priority(abs(prio))

                logger.info(f'PER @ update_priorities - idx: {idx} priorities : {priorities}')

                self.tree.update(idx, priorities)

    def update_priority(self, idx, prios):
        priorities = self._get_priority(abs(prios))
        self.tree.update(idx, priorities)

