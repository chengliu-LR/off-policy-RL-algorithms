import os
import gym
import numpy as np
from utils import ReplayBuffer
from models import DDPG
import matplotlib.pyplot as plt
import seaborn
seaborn.set_style('darkgrid')

# hyperparameters
runs = 2
max_episode = 150
training_data = np.zeros((runs, max_episode))
# make pendulum environment
env = gym.make('Pendulum-v0')   # it is wrapped by the TimeLimit (200 steps) wrapper

def train():
    # save trained model under preTrained directory
    directory = "./preTrained"
    filename = "ddpg"
    # set epsilon exploration rate and decay rate
    epsilon = 0.2
    eps_min = 1e-3
    eps_decay = 2e-3
    gaussian_exploration_noise = 0.2
    # set learning rate and batch size
    lr = 1e-3
    batch_size = 128
    # initialize replay memory
    replay_buffer = ReplayBuffer(max_size=5e4)
    # rewards for each episode / for plot
    rewards = np.zeros(max_episode)
    # initialize DDPG agent
    agent = DDPG(state_dim=env.observation_space.shape[0] - 1,  # state_dim=env.observation_space.shape[0]
                    action_dim=env.action_space.shape[0],
                    action_bounds=env.action_space.high[0],
                    lr=lr)

    for epoch in range(max_episode):
        # reset environment
        state = env.reset()
        done = False
        # epsilon decay
        epsilon = eps_min if (epsilon - eps_decay) < 0 else (epsilon - eps_decay)

        while not done:
            if np.random.random_sample() > epsilon:
                action = agent.select_action(state)
                action = action + np.random.normal(0, gaussian_exploration_noise)
            else:
                action = np.array(np.random.uniform(env.action_space.low[0], env.action_space.high[0])).reshape(1,)
            # perform one step update on pendulum
            next_state, reward, done, _ = env.step(action)
            env.render()
            replay_buffer.add((state, action, reward, next_state, done))
            
            # go to next state
            state = next_state
            # store rewards
            rewards[epoch] += reward
            # update the DDPG agent sampled on replay buffer and n_iter times
            agent.update(buffer=replay_buffer, n_iter=10, batch_size=batch_size)

        if rewards[epoch] > -1.0:
            print("task solved!\n")
            # save trained agent
            if not os.path.exists(directory):
                os.mkdir(directory)
                agent.save(directory, filename)
        
        # print rewards of current episode
        if epoch % 10 == 0:
            print('train epoch:', epoch, 'rewards:', rewards[epoch])

    return rewards

if __name__ == '__main__':
    plt.figure(dpi = 500)
    # run for several independent trials
    for i in range(runs):
        training_data[i] = train()
    seaborn.tsplot(training_data, color='blue')
    plt.xlabel('Episodes')
    plt.ylabel('Rewards')
    plt.legend(labels=['ddpg with target network'])
    plt.title('training {} episodes'.format(max_episode))
    plt.savefig("./figures/ddpg_training.png")
    plt.close()