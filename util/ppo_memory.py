import numpy as np
from collections import deque

class PPOMemory(object):
    """class Memory:

    """
    def __init__(self):
        self.buffer = deque(maxlen=1000)
        self.batch_s = []
        self.batch_a = []
        self.batch_r = []
        self.batch_gae_r = []
        self.batch_s_ = []
        self.batch_done = []
        self.GAE_CALCULATED_Q = False

    def get_batch(self, batch_size, step_size):
        print(f'## get_batch ##\nbatch_size:{batch_size}, step_size:{step_size}')
        batches = []
        idx = np.random.choice(np.arange(len(self.buffer) - step_size),
                               size=batch_size, replace=False)

        for i in idx:
            tmp_batch = []
            for j in range(step_size):
                s,a,r,gae_r,s_,d = [],[],[],[],[],[]

                s.append(self.batch_s[j])
                a.append(self.batch_a[j])
                r.append(self.batch_r[j])
                gae_r.append(self.batch_gae_r[j])
                s_.append(self.batch_s_[j])
                d.append(self.batch_done[j])

                tmp_batch.append(self.buffer[i+j])

            batches.append(tmp_batch)
        print(f'@ get_batch - batches:{batches}')
        return batches

    def get_batch_each(self, batch_size):
        for _ in range(batch_size):
            s,a,r,gae_r,s_,d = [],[],[],[],[],[]
            pos = np.random.randint(len(self.batch_s))
            s.append(self.batch_s[pos])
            a.append(self.batch_a[pos])
            r.append(self.batch_r[pos])
            gae_r.append(self.batch_gae_r[pos])
            s_.append(self.batch_s_[pos])
            d.append(self.batch_done[pos])
        return s, a, r, gae_r, s_, d

    def store(self, experience):
        self.buffer.append(experience)

    def store_each(self, s, a, r, s_, done):
        """
        :param s:
        :param a:
        :param s_:
        :param r:
        :param done:
        :return:
        """
        self.batch_s.append(s)
        self.batch_a.append(a)
        self.batch_r.append(r)
        self.batch_s_.append(s_)
        self.batch_done.append(done)

    def clear(self):
        self.batch_s.clear()
        self.batch_a.clear()
        self.batch_r.clear()
        self.batch_s_.clear()
        self.batch_done.clear()
        self.GAE_CALCULATED_Q = False

    @property
    def cnt_samples(self):
        return len(self.buffer)