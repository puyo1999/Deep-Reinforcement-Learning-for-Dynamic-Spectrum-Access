import torch
import numpy as np
from collections import deque

class PrioritizedReplay(object):

    def __init__(self, capa,batch, gamma, al=0.6, bs=0.4, bf = 1e5, ere=False):
        self.al = al
        self.bs = bs
        self.bf = bf
        self.frame = 1

        self.capa = capa
        self.buffer = []
        self.pos = 0
        self.priorities = np.zeros((capa,), dtype = np.float32)

        self.ere = ere
        self.sample = self.sample_
        if ere == False:
            self.sample = self.sample_
        else:
            self.sample = self.sample_ere

    def beta_by_frame(self, fidx):
        return min(1.0, self.bs + fidx*(1.0 - self.bs))

    def add(self, state, action, reward, next_state, done):
        max_prio = self.priorities.max() if self.buffer else 1.0

        if len(self.buffer) < self.capacity:
            self.buffer.append(())
        else:
            self.buffer[self.pos] = (state, action, reward, next_state, done)
        self.priorities[self.pos] = max_prio
        self.pos = (self.pos+1) % self.capa
        self.iter_ += 1
    def sample_(self):
        N = len(self.buffer)
        if N == self.capa:
            prios = self.priorities
        else:
            prios = self.priorities[:self.pos]
        # calc P = p^a/sum(p^a)
        probs = prios ** self.al
        P = probs / probs.sum()

        indices = np.random.choice(N, self.bs, p=P)
        samples = [self.buffer[idx] for idx in indices]

        beta = self.beta_by_frame(self.frame)
        self.frame += 1

        # Compute importance-sampling weight
        weights = (N * P[indices]) ** (-beta)
        # normalize weights
        weights /= weights.max()
        weights = np.array(weights, dtype=np.float32)

        states, actions, rewards, next_states, dones = zip(*samples)

        states = torch.FloatTensor(np.float32(np.concatenate(states))).to(self.device)
        next_states = torch.FloatTensor(np.float32(np.concatenate(next_states))).to(self.device)
        actions = torch.cat(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device).unsqueeze(1)
        dones = torch.FloatTensor(dones).to(self.device).unsqueeze(1)
        weights = torch.FloatTensor(weights).unsqueeze(1)

        return (states, actions, rewards, next_states, dones, indices, weights)

    def sample_ere(self, c_k):
        N = len(self.buffer)
        if c_k > N:
            c_k = N
        if N == self.capa:
            prios = np.array(self.priorities)
        else:
            prios = np.array(list(self.priorities)[:c_k])

        probs = prios ** self.al
        P = probs / probs.sum()

        # get the indices depending on the probability p and the c_k range of the buffer
        indices = np.random.choice(c_k, self.bs, p=P)
        samples = [self.buffer[idx] for idx in indices]


        beta = self.beta_by_frame(self.frame)
        self.frame += 1

        # Compute importance-sampling weight
        weights = (c_k * P[indices]) ** (-beta)
        # normalize weights
        weights /= weights.max()
        weights = np.array(weights, dtype=np.float32)

        states, actions, rewards, next_states, dones = zip(*samples)
        states = torch.FloatTensor(np.float32(np.concatenate(states))).to(self.device)
        next_states = torch.FloatTensor(np.float32(np.concatenate(next_states))).to(self.device)
        actions = torch.cat(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device).unsqueeze(1)
        dones = torch.FloatTensor(dones).to(self.device).unsqueeze(1)
        weights = torch.FloatTensor(weights).unsqueeze(1)

        return states, actions, rewards, next_states, dones, indices, weights

    def update_priorities(self, batch_indices, batch_priorities):
        for idx, prio in zip(batch_indices, batch_priorities):
            self.priorities[idx] = prio
    def __len__(self):
        return len(self.buffer)