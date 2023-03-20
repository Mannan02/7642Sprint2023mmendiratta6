import matplotlib
matplotlib.use('TKAgg')
import gym
import numpy as np
import torch
from torch.nn.functional import mse_loss
from gym.envs.box2d import LunarLander
import matplotlib.pyplot as plt

from DNN import DNN
from ReplayMemory import ReplayMemory


class QLearningAgent(object):
    def __init__(self, hidden_size, learning_rate):
        self.soft_update_factor = 0.001
        self.C = 2
        self.batch_size = 64
        self.hidden_size = hidden_size
        self.gamma = 0.99
        self.learning_rate = learning_rate
        self.epsilon = 1
        self.epsilon_decay = 0.99
        self.num_episodes = 1500
        self.episode_scores = []
        self.episode_score_averages = {}

        self.env: LunarLander = gym.make("LunarLander-v2")
        self.num_possible_actions = self.env.action_space.n
        self.num_observations = self.env.observation_space.shape[0]
        self.Q = DNN(self.num_observations, self.hidden_size, self.num_possible_actions)
        self.Q_prime = DNN(self.num_observations, self.hidden_size, self.num_possible_actions)
        self.D = ReplayMemory(self.batch_size, 100000)
        self.optimizer = torch.optim.Adam(self.Q.parameters(), self.learning_rate)

    def choose_action_idx(self, state):
        n = np.random.rand()
        if n < self.epsilon:
            a = np.random.randint(0, self.num_possible_actions)
        else:
            with torch.no_grad():
                a = torch.argmax(self.Q.forward(state)).item()
        return a

    def train(self):
        for episode in range(self.num_episodes):
            self.epsilon = max(0.01, self.epsilon * self.epsilon_decay)
            state = self.env.reset()
            state = torch.as_tensor(state)
            t = 0
            done = False
            score = 0
            while t < 1000 and not done:
                t += 1
                a = self.choose_action_idx(state)
                new_state, reward, done, prob = self.env.step(a)
                score += reward
                new_state = torch.as_tensor(new_state)
                self.D.append(state, a, reward, new_state, done)
                state = new_state
                batch = self.D.get_batch()
                if not batch:
                    continue
                states, actions, rewards, new_states, dones = batch
                states = torch.stack(states)
                new_states = torch.stack(new_states)
                dones = torch.Tensor(dones).type(torch.bool)
                rewards = torch.Tensor(rewards)
                with torch.no_grad():
                    next_rewards = torch.max(self.Q_prime(new_states), dim=1).values
                    y = torch.where(dones, rewards, rewards + self.gamma * next_rewards)
                self.optimizer.zero_grad()
                loss = mse_loss(y.float(), self.Q(states)[torch.arange(self.batch_size), actions])
                loss.backward()
                self.optimizer.step()
                if t % self.C == 0:
                    self.Q_prime.load_state_dict(self.Q.state_dict())
            print("\r" + str(episode) + ": " + str(score), end="")
            self.episode_scores.append(float(score))
            # if (episode+1) % 100 == 0:
            #     scores = self.conduct_tests()
            #     self.episode_scores[episode+1] = scores
            #     self.episode_score_averages[episode+1] = np.mean(scores)
            #     print(f"\rThe Average score at the end of {episode + 1} episodes is: {self.episode_score_averages[episode+1]}")

    def conduct_tests(self):
        scores = []
        epsilon = self.epsilon
        self.epsilon = 0
        for _ in range(100):
            state = self.env.reset()
            state = torch.as_tensor(state)
            score = 0
            done = False
            t = 0
            while not done and t < 1000:
                t += 1
                # self.env.render()
                action = self.choose_action_idx(state)
                state, reward, done, prob = self.env.step(action)
                state = torch.as_tensor(state)
                score += reward
                # self.env.render()
            scores.append(score)
        self.epsilon = epsilon
        print()
        print(np.mean(scores))
        with open('test2.txt', 'w') as file:
            file.write('\n'.join(str(score) for score in scores))
        fig, ax = plt.subplots()
        x = [j for j in range(0, 100)]
        ax.plot(x, scores, color='b')
        fig.savefig('test2.png')

    def plot_training_curves(self, color, label):
        rolling_averages = []
        n = 100
        curr_sum = sum(self.episode_scores[:n])
        rolling_averages.append(curr_sum/n)
        for j in range(n, self.num_episodes):
            curr_sum -= self.episode_scores[j-n]
            curr_sum += self.episode_scores[j]
            rolling_averages.append(curr_sum/n)
        with open(color+label+'scorestrain2.txt', 'w') as file:
            file.write('\n'.join(str(score) for score in self.episode_scores))
        with open(color+label+'train2.txt', 'w') as file:
            file.write('\n'.join(str(avg) for avg in rolling_averages))
        # lists = sorted(self.episode_scores.items())
        fig, ax = plt.subplots()
        x = [j for j in range(n-1, self.num_episodes)]
        ax.plot(x, rolling_averages, color=color, label=label)
        ax.legend()
        fig.savefig('final_train2.png')
        # lists = sorted(self.episode_score_averages.items())
        # x, y = zip(*lists)
        # plt.plot(x, y, color=color,label=label)

if __name__ == '__main__':
    colors = ['b', 'g', 'r', 'c', 'm', 'k']
    i = 0
    # for lr in [0.1, 0.01, 0.001, 0.005]:
    #     qa = QLearningAgent(128, lr)
    #     qa.train()
    #     qa.plot_training_curves(colors[i], str(lr))
    #     i += 1
    # plt.legend()
    # plt.savefig('lr_4.png')

    # for hidden_size in [16, 32, 64, 128]:
    #     qa = QLearningAgent(hidden_size, 0.001)
    #     qa.train()
    #     qa.plot_training_curves(colors[i], str(hidden_size))
    #     i += 1
    qa = QLearningAgent(128, 0.001)
    qa.train()
    qa.plot_training_curves('b', 'rolling average')
    qa.conduct_tests()
    # scores = []
    # for _ in range(100):
    #     scores.append(qa.test())
    # print(np.mean(scores))