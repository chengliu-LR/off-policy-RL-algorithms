import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sb
import init
import getNextStateAndReward as step
from tqdm import tqdm
matplotlib.use('Agg')

#probability for exploration
EPSILON = 0.1
#step size
ALPHA = 0.8
#gama for Q-Learning
GAMMA = 1
#all possible actions
ACTIONS = [step.FORWARD, step.TURN_LEFT, step.TURN_RIGHT]
NUM_DIR = 4 #number of directions

#choose an action based on epsilon-greedy algorithm
def choose_action(state, q_value):
    if np.random.binomial(1, EPSILON) == 1:
        return np.random.choice(ACTIONS)
    else:
        values_ = q_value[state[0], state[1], state[2], :]
        return np.random.choice([action_ for action_, value_ in enumerate(values_) if value_ == np.max(values_)])   #sometimes there is not only one max

# an episode with Q-Learning
# q_value: values for state action pair, will be updated
# step_size: step size for updating
# return: total rewards within this episode
def q_learning(q_value, step_size = ALPHA):
    state = init.init_absolute_state(30, 38, step.EAST)
    rewards = 0.0
    while ~ step.goal_reached(state[0], state[1]):
        action = choose_action(state, q_value)
        #get next absolute state and reward
        next_state, reward = step.getNextAbsoluteStateAndReward(maze, state, action)
        rewards += reward
        q_value[state[0], state[1], state[2], action] += step_size * (
                reward + GAMMA * np.max(q_value[next_state[0], next_state[1], next_state[2], :]) - q_value[state[0], state[1], state[2], action])
        #q_value[terminal_state, action] is always zero
        state = next_state
    return rewards

#maze initialization
maze = init.init_maze()
#maze size
MAZE_HEIGHT = maze.shape[0]
MAZE_WIDTH  = maze.shape[1]

episodes = 300
runs = 50
rewards_q_learning = np.zeros(episodes)

#learning process
for r in tqdm(range(runs)):
    q_value = np.zeros((MAZE_HEIGHT, MAZE_WIDTH, NUM_DIR, len(ACTIONS)))
    for i in tqdm(range(episodes)):
        rewards_q_learning[i] += q_learning(q_value)
rewards_q_learning /= runs

#plot
fig = plt.figure(dpi = 600) #set the resolution
plt.plot(rewards_q_learning, label='Q-Learning')
plt.xlabel('Episodes')
plt.ylabel('Sum of rewards during episode')
plt.legend()
plt.title('ParrsMaze Q-learning')
plt.savefig("../ParrsMaze/figures/Q-learning.png")
plt.close()
