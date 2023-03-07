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
    beta = .4
    beta_increment_per_sampling = .001

    def __init__(self, capacity, prior=True):
        """ Initialization
        """
        self.prior = prior
        self.capacity = capacity
        # Prioritized Experience Replay
        if prior:
            self.tree = SumTree(capacity)
        else:
            self.mem = np.zeros(capacity, dtype=object)
            self.mem_ptr = 0

    def add(self, error, sample):
        """ Save an experience to memory, optionally with its TD-Error
        """
        if self.prior:
            p = self.tree.max_p
            if not p:
                p = self.p_upper
                #p = self._get_priority(error)
            self.tree.add(p, sample)
        else:
            self.mem[self.mem_ptr] = sample
            self.mem_ptr += 1
            if self.mem_ptr == self.capacity:
                self.mem_ptr = 0

    def _get_priority(self, error):
        """ Compute an experience priority, as per Schaul et al.
        """
        return (np.abs(error) + self.e) ** self.a

    def sample(self, n, step_size):
        """ Sample a batch, optionally with (PER)
        """
        if self.prior:
            min_p = self.tree.min_p
            segment = self.tree.total_p / n
            batch = np.zeros((n, self.capacity), dtype=object)
            w = np.zeros((n,1), np.float32)
            idx = np.zeros(n, np.int32)
            a = 0
            for i in range(n):
                b = a + segment
                v = np.random.uniform(a, b)
                idx[i], p, batch[i] = self.tree.get(v)
                w[i] = (p/min_p) ** (-self.beta)
                a += segment
            self.beta = min(1., self.a + .01)
            return idx, w, batch
        else:
            mask = np.random.choice(range(self.capacity), n)
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
            tderr += self.e
            tderr = np.minimum(tderr, self.p_upper)
            for i in range(len(idx)):
                self.tree.update(idx[i], tderr[i] ** self.a)
        '''
        p = self._get_priority(error)
        self.tree.update(idx, p)
        '''