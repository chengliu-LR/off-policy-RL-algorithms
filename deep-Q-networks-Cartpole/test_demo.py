import gym
from gym import wrappers
import random
import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

env = gym.make('CartPole-v0')
#env = wrappers.Monitor(env, 'logger', force=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#get the number of actons from gym action space
n_actions = env.action_space.n
#steps counter for epsilon annealing
steps_done = 0
num_episodes = 5
episode_rewards = np.zeros(num_episodes)

class FcDQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(FcDQN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.fc_net = nn.Sequential(
            nn.Linear(self.input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, self.output_dim)
        )

    def forward(self, observation):
        qvals = self.fc_net(observation.view(-1, self.input_dim))
        return qvals

def select_action(state):
    global steps_done
    sample = random.random()
    steps_done += 1
    with torch.no_grad():
        #pick action with the largest expected reward
        return policy_net(state).argmax(dim=1, keepdim=True)

#initialize the networks, optimizers and the memory
policy_net = FcDQN(4, n_actions).to(device)
policy_net.load_state_dict(torch.load('cartpole-trained-model.pt'))
policy_net.eval()

for epoch in range(num_episodes):
    state = torch.from_numpy(env.reset()).unsqueeze(0).float().to(device)
    done = False
    #until the game is end
    while not done:
        #select and perform the action
        action = select_action(state)
        obs, reward, done, _ = env.step(action.item())
        if done:
            next_state = None
        else:
            next_state = torch.from_numpy(obs).unsqueeze(0).float().to(device)
        episode_rewards[epoch] += reward
        #move to the next state
        state = next_state
        env.render()

    print('epoch:', epoch, 'accumulated reward:', episode_rewards[epoch], 'frames:', steps_done)

averaged_rewards = np.zeros(num_episodes)
averaged_rewards[:] = episode_rewards[:].sum() / num_episodes
print('Averaged reward among 10 tests:', episode_rewards[:].sum())

#plot
plt.figure(dpi = 500) #set the resolution
plt.plot(episode_rewards, label='accumulated rewards')
plt.plot(averaged_rewards, label='averaged rewards')
plt.xlabel('Episodes')
plt.ylabel('Rewards')
plt.legend()
plt.title('DQN network test control {} episodes'.format(num_episodes))
plt.savefig("./figures/test_cartpole_control.png")
plt.close()