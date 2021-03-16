import numpy as np
import init
### this function take a maze, current state, and action as 3 inputs, and will get the next state and the gotten reward based on the action.

#actions
FORWARD    = 0
TURN_LEFT  = 1
TURN_RIGHT = 2

#rewards
DEFAULT_REWARD = -1
WALL_PENALTY   = -2
GOAL_REWARD    = 10


#headings
NORTH  = 0
EAST   = 1
SOUTH  = 2
WEST   = 3

# judge if goal reached
def goal_reached(r, c):
    return r >= 75 and c >= 75

#judge if state is out of border
def out_of_border(maze, currentState):
    return currentState[0] < 0 or currentState[0] >= maze.shape[0] or \
    currentState[1] < 0 or currentState[1] >= maze.shape[1]

#judge if state in wall
def state_in_wall(maze, currentState):
    return maze[currentState[0], currentState[1]]

#method of getting next absolute state and reward
def getNextAbsoluteStateAndReward(maze, currentState, action):
    #default reward
    reward = DEFAULT_REWARD
    #state dimension incorrect
    if currentState.shape[0] != 3:
        print('\nWarning: state dimension incorrect!\n')
    #state out of border, initiate state and reward
    if out_of_border(maze, currentState):
        print('\nWarning: current state is out of border!')
        return np.array([currentState[0], currentState[1], currentState[2]]), reward
    #state in wall, initial state and reward
    if state_in_wall(maze, currentState):
        print('\nWarning: current state in wall!')
        return np.array([currentState[0], currentState[1], currentState[2]]), reward

    [r, c, h] = currentState
    #reward & nextState calculation
    if action == FORWARD:
        if h == NORTH:
            if maze[r - 1, c]:  #run into wall
                reward = WALL_PENALTY
                h = SOUTH    #moved into wall, keep state, reverse direction.
            else:
                r -= 1
                if goal_reached(r, c):
                    reward = GOAL_REWARD    #goal reached
        elif h == EAST:
            if maze[r, c + 1]:
                reward = WALL_PENALTY
                h = WEST
            else:
                c += 1
                if goal_reached(r, c):
                    reward = GOAL_REWARD
        elif h == SOUTH:
            if maze[r + 1, c]:
                reward = WALL_PENALTY
                h = NORTH
            else:
                r += 1
                if goal_reached(r, c):
                    reward = GOAL_REWARD
        elif h == WEST:
            if maze[r, c - 1]:
                reward = WALL_PENALTY
                h = EAST
            else:
                c -= 1
                if goal_reached(r, c):
                    reward = GOAL_REWARD

    elif action == TURN_LEFT:
        if h == NORTH:
            h = WEST
        elif h == WEST:
            h = SOUTH
        elif h == SOUTH:
            h = EAST
        elif h == EAST:
            h = NORTH
        else:
            print('Incorrect heading')

    elif action == TURN_RIGHT:
        if h == NORTH:
            h = EAST
        elif h == WEST:
            h = NORTH
        elif h == SOUTH:
            h = WEST
        elif h == EAST:
            h = SOUTH
        else:
            print('Incorrect heading')
    else:
        print('Incorrect action')

    nextState = np.array([r, c, h], dtype = np.short)
    return nextState, reward
