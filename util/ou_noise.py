import numpy as np
import numpy.random as nr
import matplotlib.pyplot as plt

class OUNoise:
    def __init__(self, action_dim, mu=0, theta=0.15, sigma=0.3):
        self.action_dim = action_dim
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_dim) * self.mu
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu

    def noise(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * nr.randn(len(x))
        print('dx : {}'.format(dx))
        self.state = x + dx
        return self.state

if __name__ == '__main__':
    ou = OUNoise(3)
    states = []
    for i in range(10):
        states.append(ou.noise())
    print('shape of states :\n{}\n'.format(np.shape(states)))
    plt.plot(states)
    plt.show()