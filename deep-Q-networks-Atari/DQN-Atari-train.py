import gym
import random
import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from torchsummary import summary
from collections import namedtuple
import skimage.transform as transforms
from skimage import color
import datetime
#from torchviz import make_dot

BATCH_SIZE = 64
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 5000
TARGET_UPDATE = 20
num_episodes = 10000
episode_rewards = np.zeros(num_episodes)

#frames between actions (input channel to ConvNet)
num_frames = 4
#steps counter for epsilon annealing
steps_done = 0

#make atari environment and set the device to GPU if supported
env = gym.make('SpaceInvadersNoFrameskip-v0')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#get the number of actons from gym action space
n_actions = env.action_space.n

#Replay Memory. It stores the transitions that the agent observes, allowing reusing this data later.
#By sampling from it randomly, the transitions that build up a batch are decorrelated.
#It greatly stabilizes and improves the DQN training procedure.
Transition = namedtuple('Transition', 
                        ('state', 'action', 'next_state', 'reward'))

class ReplayMemory():
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0
    
    def push(self, *args):
        """save as a transition"""
        if len(self.memory) < self.capacity:
            self.memory.append(None)

        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class ConvDQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ConvDQN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        #if feature_dim = 6272, input_batch / 2 = output_batch
        self.feature_dim = 3136

        self.conv_net = nn.Sequential(
            nn.Conv2d(self.input_dim, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
            )

        self.fc_net = nn.Sequential(
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(),
            nn.Linear(512, self.output_dim)
            )

    def forward(self, observation):
        features = self.conv_net(observation)
        qvals = self.fc_net(features.view(-1, self.feature_dim))
        return qvals

def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            #pick action with the largest expected reward
            return policy_net(state).argmax(dim=1, keepdim=True)
    else:
        return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)

def update_policy():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    #convert the batch-array of transitions to Transition of batch-arrays
    batch = Transition(*zip(*transitions))
    #compute a mask of non-terminal states and concatenate the batch elements
    non_termi_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)),
                                            device=device, 
                                            dtype=torch.bool)
    next_state_batch = torch.cat([s for s in batch.next_state if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)
    #compute Q(s_t, a) based on policy network and action selection for each batch state
    q_values = policy_net(state_batch).gather(1, action_batch)
    #compute expected values based on target network
    max_next_q = torch.zeros((BATCH_SIZE, 1), device=device)
    max_next_q[non_termi_mask] = target_net(next_state_batch).max(dim=1, keepdim=True)[0].detach()
    expected_q_values = (max_next_q * GAMMA) + reward_batch
    #compute loss and optimizer update
    loss = F.smooth_l1_loss(q_values, expected_q_values)
    optimizer.zero_grad()
    loss.backward()
    #error clipping further improve the stability of the algorithm
    for param in policy_net.parameters():
        param.grad.data.clamp_(-2, 2)
    optimizer.step()
    #print loss on screen
    if steps_done % 1000 == 0:
        print('loss:', loss.item())

#convert the np ndarray RGB image to flickering removed and stacked frame chains (states)
def observe_env(obs, action):
    total_reward = 0
    state = torch.zeros((num_frames, 84, 84))
    obs_buffer = np.zeros((2, ) + env.observation_space.shape, dtype=np.uint8)
    obs_buffer[0] = obs
    # note that if this inner loop finished before state is filled, state is an incorrect 
    #representation of the environemnt.
    #it doens't matter since the termial state is not used for network input.
    for i in range(num_frames):
        obs, reward, done, info = env.step(action)
        if done:
            state = None
            break
        obs_buffer[1] = obs
        total_reward += reward
        #remove flickering by max operation and warp the image to 84x84, rgb2gray will convert
        #the uint8 to float
        max_frame = transforms.resize(obs_buffer.max(axis=0), (84, 84))
        state[i] = torch.from_numpy(color.rgb2gray(max_frame)).float()
        obs_buffer[0] = obs
    if state is not None:
        state = state.unsqueeze(0).to(device)
    return state, total_reward, done, obs

#initialize the networks, optimizers and the memory
policy_net = ConvDQN(num_frames, n_actions).to(device)
#policy_net.load_state_dict(torch.load('policy_net_state_dict.pt'))
target_net = ConvDQN(num_frames, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.RMSprop(policy_net.parameters(), lr=1e-3)

memory = ReplayMemory(200000)

for epoch in range(num_episodes):
    obs = env.reset()
    action = env.action_space.sample()
    state, _, done, obs = observe_env(obs, action)
    #until the game is end
    while not done:
        #select and perform the action
        action = select_action(state)
        #obs, reward, done, _ = env.step(action.item())
        next_state, reward, done, obs = observe_env(obs, action.item())
        episode_rewards[epoch] += reward
        reward = torch.tensor([[reward]], device=device).float()    #tensor shape(1,1)
        #store the transition in memory
        memory.push(state, action, next_state, reward)
        #move to the next state
        state = next_state
        #perform one step of optimization
        update_policy()
        env.render()
        
    print('time:', datetime.datetime.now(), 'epoch:', epoch, 'accumulated reward:', episode_rewards[epoch], 'frames:', steps_done)
    if epoch % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())
    if epoch % 100 == 0:
        #save trained model
        torch.save(policy_net.state_dict(), 'policy_net_state_dict.pt')

#plot
plt.figure(dpi = 500) #set the resolution
plt.plot(episode_rewards, label='accumulated rewards')
plt.xlabel('Episodes')
plt.ylabel('Rewards')
plt.legend()
plt.title('DQN network training {} episodes'.format(num_episodes))
plt.savefig("./episode_rewards.png")
plt.close()