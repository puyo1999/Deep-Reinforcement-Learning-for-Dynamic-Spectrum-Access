import random
import numpy as np

from collections import deque
from .sumtree import SumTree

class PerMemory(object):
    """ Memory Buffer Helper class for Experience Replay
    using a double-ended queue or a Sum Tree (for PER)
    """
    e = 0.01
    a = 0.6
    beta = 0.4
    beta_increment_per_sampling = 0.001

    def __init__(self, capacity):
        """ Initialization
        """
        # Prioritized Experience Replay
        self.tree = SumTree(capacity)
        self.capacity = capacity

    def add(self, error, sample):
        """ Save an experience to memory, optionally with its TD-Error
        """
        p = self._get_priority(error)
        self.tree.add(p, sample)

    def _get_priority(self, error):
        """ Compute an experience priority, as per Schaul et al.
        """
        return (np.abs(error) + self.e) ** self.a

    def sample(self, n, step_size):
        """ Sample a batch, optionally with (PER)
        """
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


    def update(self, idx, error):
        """ Update priority for idx (PER)
        """
        p = self._get_priority(error)
        self.tree.update(idx, p)