import random
import numpy as np

from collections import deque
from .sumtree import SumTree

class PerMemory(object):
    """ Memory Buffer Helper class for Experience Replay
    using a double-ended queue or a Sum Tree (for PER)
    """
    p_upper = 1.
    e = .01
    a = .6
    a_decrement_per_sampling = .002
    beta = .4
    beta_increment_per_sampling = .001

    # feature_size = state_size 이므로 2 * (NUM_CHANNELS + 1)
    def __init__(self, mem_size, feature_size, prior=False):
        """ Initialization
        """
        print("@ PerMemory : init - feature_size = ", feature_size)
        self.prior = prior
        self.data_len = 6 * feature_size + 6
        #self.data_len = feature_size
        print("@ PerMemory : init - prior:{} data_len:{}".format(self.prior ,self.data_len))

        # Prioritized Experience Replay
        if prior:
            self.tree = SumTree(mem_size, self.data_len)

        else:
            self.mem_size = mem_size
            self.mem = np.zeros(mem_size, self.data_len, dtype=np.float32)
            self.mem_ptr = 0

    def store(self, transition):
        if self.prior:
            p = self.tree.max_p
            print('PER @ store with p(max_p): {}\n'.format(p))
            if not p:
                p = self.p_upper
            #print("PER @ store - p:{}".format(p))
            #print("PER @ store - transition:{}".format(transition))
            self.tree.add(p, transition)
            #print('PER @ store - check min_p:{}\n'.format(self.tree.min_p))
        else:
            self.mem[self.mem_ptr] = transition
            self.mem_ptr += 1
            if self.mem_ptr == self.mem_size:
                self.mem_ptr = 0

    def add(self, sample, error):
        """ Save an experience to memory, optionally with its TD-Error
        """
        if self.prior:
            p = self._get_priority(error)
            #print('PER @ add - p: {}'.format(p))
            self.tree.add(p, sample)
        else:
            self.mem[self.mem_ptr] = sample
            self.mem_ptr += 1
            if self.mem_ptr == self.mem_size:
                self.mem_ptr = 0
        #print('PER @ add - check min_p:{}\n'.format(self.tree.min_p))

    def _get_priority(self, error):
        """ Compute an experience priority, as per Schaul et al.
        """
        #print('PER @ _get_priority : {}'.format(np.power(error + self.e, self.a).squeeze()))
        #return np.power(error + self.e, self.a).squeeze()
        self.a = np.max([1., self.a - self.a_decrement_per_sampling])
        #print('PER @ _get_priority : {}'.format((np.abs(error) + self.e) ** self.a))
        return (np.abs(error) + self.e) ** self.a

    def sample(self, n):
        """ Sample a batch, optionally with (PER)
        """
        if self.prior:
            #min_p = self.tree.min_p
            #print('PER @ sample : check current min_p : {}'.format(min_p))

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
                self.tree.update(idx, priorities)


