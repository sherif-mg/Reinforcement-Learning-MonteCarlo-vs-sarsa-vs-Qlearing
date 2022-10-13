import gym
import numpy as np
import math


class Qlearing():
    def __init__(self, num_episodes=4000, min_lr=0.2, min_explore=0.2, discount=1, decay=25,size = (181, 141)):
        self.num_episodes = num_episodes
        self.min_lr = min_lr
        self.min_explore = min_explore
        self.discount = discount
        self.decay = decay
        self.env = gym.make('MountainCar-v0')
        self.Q_table = np.zeros(size + (self.env.action_space.n,))

    def get_explore_rate(self, t):
        return max(self.min_explore, min(1., 1. - math.log10((t + 1) / self.decay)))

    def get_lr(self, t):
        return max(self.min_lr, min(1., 1. - math.log10((t + 1) / self.decay)))

    def update_q(self, state, action, reward, new_state):
        self.Q_table[state][action] += self.lr * (reward + self.discount * np.max(self.Q_table[new_state]) - self.Q_table[state][action])


    def choose_action(self, state):
        if (np.random.random() < self.explore_rate):
            return self.env.action_space.sample()
        else:
            return np.argmax(self.Q_table[state])

    def discretize_state(self, state):
        state[0] = state[0].round(1)
        state[1] = state[1].round(2)
        state[0] *= 10
        state[1] *= 100
        state[0] += 12
        state[1] += 7
        integer = [0, 0]
        integer[0] = int(state[0])
        integer[1] = int(state[1])
        return tuple(integer)

    def discretize_state1(self, state):
        state[0] = state[0].round(2)
        state[1] = state[1].round(3)
        state[0] *= 100
        state[1] *= 1000
        state[0] += 120
        state[1] += 70
        integer = [0, 0]
        integer[0] = int(state[0])
        integer[1] = int(state[1])
        return tuple(integer)

    def train(self):
        for e in range(self.num_episodes):
            current_state = self.discretize_state1(self.env.reset())
            self.lr = self.get_lr(e)
            self.explore_rate = self.get_explore_rate(e)
            done = False
            while not done:
                action = self.choose_action(current_state)##############different from sarsa
                obs, reward, done, _ = self.env.step(action)
                new_state = self.discretize_state1(obs)
                self.update_q(current_state, action, reward, new_state)
                current_state = new_state
        print('Finished training!')

    def run(self):
        self.env = gym.make('MountainCar-v0')
        while True:
            current_state = self.discretize_state1(self.env.reset())
            done = False
            self.explore_rate = 0.0
            while not done:
                self.env.render()
                action = self.choose_action(current_state)
                obs, reward, done, _ = self.env.step(action)
                new_state = self.discretize_state1(obs)
                current_state = new_state

    def savePolicy(self):
      np.save("Qualityepisodes{episodes},discount{dic},min_lr{min_lr}_Qlearing".format(episodes=self.num_episodes, dic=self.discount,min_lr=self.min_lr), self.Q_table)

    def LoadPolicy(self,path):
      self.Policy_table = np.load(path)

    def LoadQuality(self,path):
      self.Q_table = np.load(path)

agent = Qlearing()
#agent.train()
agent.LoadQuality("Qualityepisodes30000,discount1,min_lr0.2_Qlearing.npy")
agent.run()
