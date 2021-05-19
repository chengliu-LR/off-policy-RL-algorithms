import gym
import random
import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from collections import namedtuple
#from torchviz import make_dot

env = gym.make('CartPole-v0')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

BATCH_SIZE = 64
GAMMA = 0.999
EPS_START = 0.999
EPS_END = 0.001
EPS_DECAY = 1000
TARGET_UPDATE = 20

#get the number of actons from gym action space
n_actions = env.action_space.n
#steps counter for epsilon annealing
steps_done = 0
num_episodes = 1000
episode_rewards = np.zeros(num_episodes)
episode_count = 0

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
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            #pick action with the largest expected reward
            return policy_net(state).argmax(dim=1, keepdim=True)
    else:
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)

def update_policy():

    if len(memory) < BATCH_SIZE:
        return
        
    transitions = memory.sample(BATCH_SIZE)
    #convert the batch-array of transitions to Transition of batch-arrays
    batch = Transition(*zip(*transitions))
    #compute a mask of non-terminal states and concatenate the batch elements
    non_termi_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device, dtype=torch.bool)
                                            
    next_state_batch = torch.cat([s for s in batch.next_state
                                            if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)
    #compute Q(s_t, a) based on policy network and action selection for each batch state
    q_values = policy_net(state_batch).gather(1, action_batch)
    #compute expected values based on target network
    max_next_q = torch.zeros((BATCH_SIZE, 1), device=device)
    max_next_q[non_termi_mask] = target_net(next_state_batch).max(dim=1,
                                                                keepdim=True)[0].detach()
    expected_q_values = (max_next_q * GAMMA) + reward_batch
    #compute loss and optimizer update
    loss = F.smooth_l1_loss(q_values, expected_q_values)
    optimizer.zero_grad()
    loss.backward()
    #error clipping further improve the stability of the algorithm
    for param in policy_net.parameters():
        param.grad.data.clamp_(-2, 2)
    optimizer.step()

    #visualize the calculation graph
    #visual_graph = make_dot(loss, params=dict(policy_net.named_parameters()))
    #visual_graph.view()

#initialize the networks, optimizers and the memory
policy_net = FcDQN(4, n_actions).to(device)
target_net = FcDQN(4, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())
#target_net.eval()

optimizer = optim.RMSprop(policy_net.parameters(), lr=1e-2)

memory = ReplayMemory(5000)

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
        reward = torch.tensor([[reward]], device=device)    #tensor shape(1,1)
        #store the transition in memory
        memory.push(state, action, next_state, reward)
        #move to the next state
        state = next_state
        #perform one step of optimization
        update_policy()
        #env.render()

    if episode_rewards[epoch - 10 : epoch + 1].sum() > 1950:
        torch.save(policy_net.state_dict(), 'cartpole-trained-model.pt')
        episode_count = epoch
        break

    if epoch % 10 == 0:
        print('epoch:', epoch, 'accumulated reward:', episode_rewards[epoch], 'frames:', steps_done)

    if epoch % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())


#plot
plt.figure(dpi = 500) #set the resolution
plt.plot(episode_rewards[:episode_count], label='accumulated rewards')
plt.xlabel('Episodes')
plt.ylabel('Rewards')
plt.legend()
plt.title('DQN network training {} episodes'.format(num_episodes))
plt.savefig("./figures/train_dqn_cartpole.png")
plt.close()