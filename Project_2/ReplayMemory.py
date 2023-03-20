from collections import deque
import numpy as np


class ReplayMemory:
    def __init__(self, batch_size, maxlen):
        self.q = deque(maxlen=maxlen)
        self.batch_size = batch_size

    def append(self, s, a, r, s_new, done):
        self.q.append((s, a, r, s_new, done))

    def get_batch(self):
        if self.batch_size > len(self.q):
            return False
        indices = np.random.choice(len(self.q), self.batch_size, replace=False)
        states = [self.q[i][0] for i in indices]
        actions = [self.q[i][1] for i in indices]
        rewards = [self.q[i][2] for i in indices]
        new_states = [self.q[i][3] for i in indices]
        dones = [self.q[i][4] for i in indices]
        return states, actions, rewards, new_states, dones
