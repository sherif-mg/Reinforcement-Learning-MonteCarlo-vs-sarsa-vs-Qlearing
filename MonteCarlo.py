import gym
import numpy as np
import math
class MonteCarlo:
  def __init__(self, num_episodes=1000, min_lr=0.25, min_explore=0.4,discount=1,size=(181,141)):
        self.num_episodes = num_episodes
        self.min_lr = min_lr
        self.min_explore = min_explore
        self.discount = discount
        self.env = gym.make('MountainCar-v0')
        self.Q_table = np.zeros(size + (self.env.action_space.n,))
        #self.Policy_table = np.random.randint(3, size=(19,15))
        self.Policy_table = np.random.choice(np.arange(0,3),size=size)#, p=[0.4, 0.2, 0.4]
  def discretize_state(self, state):
    state[0] = state[0].round(1)
    state[1] = state[1].round(2)
    state[0] *= 10
    state[1] *= 100
    state[0] += 12
    state[1] += 7
    integer = [0,0]
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
    integer = [0,0]
    integer[0] = int(state[0])
    integer[1] = int(state[1])
    return tuple(integer)
  def update_q(self, state, action, G):
        self.Q_table[state][action] = self.Q_table[state][action] + self.min_lr* (G-self.Q_table[state][action])

  def choose_action(self, state):
        if np.random.uniform() < self.min_explore:
            return self.env.action_space.sample()
        else:
            return np.argmax(self.Q_table[state])

  def UpdatePolicy(self,episode):
    for state, action, reward in episode:
        self.Policy_table[state] = np.argmax(self.Q_table[state])



  def generateEpisode(self):
    episode = []
    state = self.discretize_state1(self.env.reset())
    while True:
        action = self.choose_action(state)
        next_state, reward, done, info = self.env.step(action)
        next_state=self.discretize_state1(next_state)
        episode.append((state, action, reward))
        state = next_state
        if done:
            break
    return episode

  def mc_prediction(self):
    for i_episode in range(1, self.num_episodes + 1):
        episode=self.generateEpisode()
        for state,action,reward in episode:
            # Find the first occurance of the state in the episode
            first_occurence_idx = next(i for i,x in enumerate(episode) if x[0][0] == state[0] and x[0][1] == state[1])
            G = sum([x[2]*(self.discount**i) for i,x in enumerate(episode[first_occurence_idx:])])
            self.update_q(state, action, G)
        #self.UpdatePolicy(episode)

  def train(self):
      self.mc_prediction()
      print("Finish")
  def run(self):
    for e in range(self.num_episodes):
        current_state = self.discretize_state1(self.env.reset())
        done = False
        while not done:
            action = self.choose_action(current_state)
            #action = self.Policy_table[current_state]
            obs, reward, done, _ = self.env.step(action)
            self.env.render()
            current_state = self.discretize_state1(obs)
    print('Finished running!')
  def savePolicy(self):
     np.save("Qualityepisodes{episodes},discount{dic},min_lr{min_lr}pp".format(episodes=self.num_episodes, dic=self.discount,min_lr=self.min_lr), self.Q_table)

  def LoadPolicy(self,path):
     self.Policy_table = np.load(path)

  def LoadQuality(self,path):
     self.Q_table = np.load(path)

a=MonteCarlo()
#a.train()
#a.savePolicy()
a.LoadQuality("quality table monto carlo\Qualityepisodes80000,discount0.1,min_lr0.2gameed.npy")
#a.LoadPolicy("policy monto carlo argmax\Policyepisodes5000dic_1_newrewardfinishreward.npy")

a.run()
