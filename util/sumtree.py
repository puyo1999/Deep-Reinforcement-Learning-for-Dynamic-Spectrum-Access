import numpy as np

""" Original Code by @jaara: https://github.com/jaara/AI-blog/blob/master/SumTree.py
"""

class SumTree:
    #write = 0

    def __init__(self, mem_size, data_len):
        self.tree = np.zeros(2 * mem_size - 1)
        #self.data = np.zeros((mem_size, data_len), dtype=object)
        self.data = np.zeros((mem_size, data_len), dtype=np.float32)
        print("### SumTree init - shape of SumTree data : ", np.shape(self.data))

        #self.n_entries = 0
        self.size = mem_size
        self.write = 0

    # update to the root node
    def _propagate(self, idx, change):
        parent = (idx - 1) // 2

        self.tree[parent] += change

        if parent != 0:
            self._propagate(parent, change)

    # find sample on leaf node
    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s-self.tree[left])

    def total(self):
        return self.tree[0]

    # store priority and sample
    def add(self, p, data):
        print("@ SumTree : add - p:{}".format(p))
        self.data[self.write] = data
        self.update(self.write, p)

        self.write += 1
        if self.write == self.size:
            self.write = 0

    # update priority
    def update(self, idx, p):
        #tree_idx = idx + self.size - 1
        diff = p - self.tree[idx]
        self.tree[idx] += diff
        while idx:
            idx = (idx - 1) // 2
            self.tree[idx] += diff
    '''
    def update(self, idx, p):
        change = p - self.tree[idx]
        self.tree[idx] = p
        self._propagate(idx, change)
    '''
    def get(self, s):
        assert s <= self.total()
        idx = self._retrieve(0, s)
        dataIdx = idx - self.size + 1
        return (idx, self.tree[idx], self.data[dataIdx])

    def sample(self, value):
        ptr = 0
        while ptr < self.size - 1:
            left = 2 * ptr + 1
            if value < self.tree[left]:
                ptr = left
            else:
                value -= self.tree[left]
                ptr = left + 1

        return ptr - (self.size - 1), self.tree[ptr], self.data[ptr - (self.size - 1)]
    @property
    def total_p(self):
        return self.tree[0]
    @property
    def max_p(self):
        # 전체에서 max 값 추출
        print('@@@ pyk : size:{} max_p:{}'.format(self.size, np.max(self.tree[-self.size:])))
        return np.max(self.tree[-self.size:])
    @property
    def min_p(self):
        # 전체에서 min 값 추출
        print('@@@ pyk : size:{} min_p:{}'.format(self.size, np.min(self.tree[-self.size:])))
        return np.min(self.tree[-self.size:])