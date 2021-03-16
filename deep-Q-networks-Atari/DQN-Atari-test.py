import gym
#import cv2
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import skimage.transform as transforms
from skimage import color
import datetime
#from torchviz import make_dot

num_episodes = 10
episode_rewards = np.zeros(num_episodes)

#frames between actions (input channel to ConvNet)
num_frames = 4
#steps counter for epsilon annealing
steps_done = 0

#make atari environment and set the device to GPU if supported
#env = gym.make('BreakoutNoFrameskip-v0')
env = gym.make('BreakoutNoFrameskip-v0')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#get the number of actons from gym action space
n_actions = env.action_space.n

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
            nn.ReLU())

        self.fc_net = nn.Sequential(
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(),
            nn.Linear(512, self.output_dim))

    def forward(self, observation):
        features = self.conv_net(observation)
        qvals = self.fc_net(features.view(-1, self.feature_dim))
        return qvals

def select_action(state):
    global steps_done
    steps_done += 1
    with torch.no_grad():
        #pick action with the largest expected reward
        return policy_net(state).argmax(dim=1, keepdim=True)
        #return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)

#convert the np ndarray RGB image to flickering removed and stacked frame chains
def observe_env(obs, action):
    total_reward = 0
    state = torch.zeros((num_frames, 84, 84))
    obs_buffer = np.zeros((2, )+env.observation_space.shape, dtype=np.uint8)
    obs_buffer[0] = obs
    # note that if this inner loop finished before state is filled, state is an incorrect for the 
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
policy_net.load_state_dict(torch.load('policy_net_state_dict.pt', map_location=torch.device('cpu')))

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
        #move to the next state
        state = next_state
        #perform one step of optimization
        env.render()

    print(datetime.datetime.now(), 'epoch:', epoch, 'accumulated reward:', episode_rewards[epoch], 'frames:', steps_done)

#plot
plt.figure(dpi = 500) #set the resolution
plt.plot(episode_rewards, label='accumulated rewards')
plt.xlabel('Episodes')
plt.ylabel('Rewards')
plt.legend()
plt.title('DQN network training {} episodes'.format(num_episodes))
plt.savefig("./episode_rewards.png")
plt.close()