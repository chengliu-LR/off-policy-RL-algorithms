import os
import gym
from models import DDPG

# hyperparameters
max_episode = 5
# make pendulum environment
env = gym.make('Pendulum-v0')

def test():
    # trained model directory
    directory = "./preTrained"
    filename = "ddpg"
    # initialize DDPG agent
    agent = DDPG(state_dim=env.observation_space.shape[0] - 1, # state_dim=env.observation_space.shape[0]
                    action_dim=env.action_space.shape[0],
                    action_bounds=env.action_space.high[0],
                    lr=0)
                    
    # load trained agent
    assert os.path.exists(directory), "Trained model not exists, try running train.py first."
    agent.load(directory, filename)
    for epoch in range(max_episode):
        # reset environment
        state = env.reset()
        done = False
        rewards = 0

        while not done:
            action = agent.select_action(state)
            # perform one step update on pendulum
            next_state, reward, done, _ = env.step(action)
            # go to next state
            state = next_state
            rewards += reward
            # render envrionment
            env.render()
        print("Episode:{:2d}, Rewards:{:3f}".format(epoch, rewards))
if __name__ == '__main__':
    test()
