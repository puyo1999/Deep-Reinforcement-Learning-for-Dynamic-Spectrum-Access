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
    a = .7
    beta = .5
    beta_increment_per_sampling = .001

    # feature_size = state_size 이므로 2 * (NUM_CHANNELS + 1)
    def __init__(self, mem_size, feature_size, prior=True):
        """ Initialization
        """
        print("@ PerMemory : init - feature_size = ", feature_size)
        self.prior = prior
        self.data_len = 6 * feature_size + 6
        #self.data_len = feature_size
        print("@ PerMemory : init - data_len : {}".format(self.data_len))

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
            print('@ store with p(max_p): {}\n'.format(p))
            if not p:
                p = self.p_upper
            print("@ PerMemory : store")
            print("@ PerMemory : p = ", p)
            print("@ PerMemory : transition = ", transition)
            self.tree.add(p, transition)
            print('@ after store, check min_p:{}\n'.format(self.tree.min_p))
        else:
            self.mem[self.mem_ptr] = transition
            self.mem_ptr += 1
            if self.mem_ptr == self.mem_size:
                self.mem_ptr = 0

    def add(self, sample, error=100000):
        """ Save an experience to memory, optionally with its TD-Error
        """
        if self.prior:
            p = self._get_priority(error)
            self.tree.add(p, sample)
        else:
            self.mem[self.mem_ptr] = sample
            self.mem_ptr += 1
            if self.mem_ptr == self.mem_size:
                self.mem_ptr = 0

    def _get_priority(self, error):
        """ Compute an experience priority, as per Schaul et al.
        """
        print('@ _get_priority : {}'.format(np.power(error + self.e, self.a).squeeze()))
        return np.power(error + self.e, self.a).squeeze()
        #print('@ _get_priority : {}'.format((np.abs(error) + self.e) ** self.a))
        #return (np.abs(error) + self.e) ** self.a

    def sample(self, n):
        """ Sample a batch, optionally with (PER)
        """
        if self.prior:
            min_p = self.tree.min_p
            print('@ sample : check current min_p : {}'.format(min_p))
            if min_p == 0:
                min_p = .873
            segment = self.tree.total_p / n
            batch = np.zeros((n, self.data_len), dtype=np.float32)
            w = np.zeros((n,1), np.float32)
            idx = np.zeros(n, np.int32)
            a = 0
            for i in range(n):
                b = a + segment
                v = np.random.uniform(a, b)
                idx[i], p, batch[i] = self.tree.get(v)
                #idx[i], p, batch[i] = self.tree.sample(v)
                if min_p == 0:
                    min_p = .873
                w[i] = (p / min_p) ** (-self.beta)
                a += segment
            self.beta = min(1., self.a + .01)
            return idx, w, batch
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
