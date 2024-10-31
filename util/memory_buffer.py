import numpy as np
from collections import deque

class Memory():
    def __init__(self, max_size=1000):
        self.buffer = deque(maxlen=max_size)
        self.batch_s = []
        self.batch_a = []
        self.batch_r = []
        self.batch_gae_r = []
        self.batch_s_ = []
        self.batch_done = []
        self.GAE_CALCULATED_Q = False

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size, step_size):
        print(f'@ sample -\n batch_size:{batch_size}\n step_size:{step_size}, len(self.buffer): {len(self.buffer) }\n')

        idx = np.random.choice(np.arange(len(self.buffer) - step_size),
                               size=batch_size, replace=False)

        res = []
        gae_r = []
        print(f'@ sample - idx:{idx}')

        for i in idx:
            temp_buffer = []
            for j in range(step_size):
                temp_buffer.append(self.buffer[i + j])
            res.append(temp_buffer)
            print(f'@ sample loop - i:{i}')
            print(f'@@ batch_gae_r len - {len(self.batch_gae_r)}')
            gae_r.append(self.batch_gae_r[i])
        print(f'@ sample - res:{res}')
        return res, gae_r

    def get_batch_each(self, batch_size):
        for _ in range(batch_size):
            s,a,r,gae_r,s_,d = [],[],[],[],[],[]
            pos = np.random.randint(len(self.batch_s))
            s.append(self.batch_s[pos])
            a.append(self.batch_a[pos])
            gae_r.append(self.batch_gae_r[pos])
            s_.append(self.batch_s_[pos])
            d.append(self.batch_done[pos])
        return (s, a, r, gae_r, s_, d)
        #return s,a,r,gae_r,s_,d

    def get_batch(self, batch_size, step_size):
        print(f'## get_batch ##\nbatch_size:{batch_size}, step_size:{step_size}')
        batches = []
        idx = np.random.choice(np.arange(len(self.buffer) - step_size),
                               size=batch_size, replace=False)

        for i in idx:
            tmp_batch = []
            for j in range(step_size):
                s,a,r,gae_r,s_,d = [],[],[],[],[],[]

                if j >= len(self.batch_gae_r):
                    print(f"Index {j} is out of range for batch_gae_r with length {len(self.batch_gae_r)}")
                    continue

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

    def get_length(self):
        return len(self.buffer)

    def clear(self):
        self.buffer = deque()
        self.GAE_CALCULATED_Q = False

    def store_each(self, s, a, r, s_, done):

        self.batch_s.append(s)
        self.batch_a.append(a)
        self.batch_r.append(r)
        self.batch_s_.append(s_)
        self.batch_done.append(done)

        print(f'@ store - s_ : {s_}')

    @property
    def cnt_samples(self):
        return len(self.batch_s)



