import numpy as np
import init
import getNextStateAndReward
#Relative state indicates distance to walls/obstacles in forward, left and right directions and the current absolute heading. relativeState = [LEFT, FORWARD, RIGHT, HEADING]. Distance can be 1,2,3 when a wall/obstacle is detected 1,2,3 blocks away. If distance is 0, then no wall/obstacle was detected within the sense-range of 3.

#look direction
LEFT_DIR = 0
FORWARD_DIR = 1
RIGHT_DIR = 2

def getRelativeState(maze, absoluteState):
    #state dimension incorrect
    if absoluteState.shape[0] != 3:
        print('\nWarning: state dimension incorrect!')
    #state out of border, initial state
    if getNextStateAndReward.out_of_border(maze, absoluteState):
        print('\nWarning: state is out of border!')
    #state in wall, initial state
    if getNextStateAndReward.state_in_wall(maze, absoluteState):
        print('\nWarning: state in wall!')

    [r, c, h] = absoluteState
    relativeState = np.zeros(4, dtype = np.short)
    relativeState[3] = h #HEADING
    if h == getNextStateAndReward.NORTH:
        for left_dist in range(1,4):
            if c - left_dist >= 0 and maze[r, c - left_dist]:
                relativeState[LEFT_DIR] = left_dist
                break
        for forward_dist in range(1,4):
            if r - forward_dist >= 0 and maze[r - forward_dist, c]:
                relativeState[FORWARD_DIR] = forward_dist
                break
        for right_dist in range(1,4):
            if c + right_dist < maze.shape[1] and maze[r, c + right_dist]:
                relativeState[RIGHT_DIR] = right_dist
                break
    elif h == getNextStateAndReward.EAST:
        for left_dist in range(1,4):
            if r - left_dist >= 0 and maze[r - left_dist, c]:
                relativeState[LEFT_DIR] = left_dist
                break
        for forward_dist in range(1,4):
            if c + forward_dist < maze.shape[1] and maze[r, c + forward_dist]:
                relativeState[FORWARD_DIR] = forward_dist
                break
        for right_dist in range(1,4):
            if r + right_dist < maze.shape[0] and maze[r + right_dist, c]:
                relativeState[RIGHT_DIR] = right_dist
                break
    elif h == getNextStateAndReward.SOUTH:
        for left_dist in range(1,4):
            if c + left_dist < maze.shape[1] and maze[r, c + left_dist]:
                relativeState[LEFT_DIR] = left_dist
                break
        for forward_dist in range(1,4):
            if r + forward_dist < maze.shape[0] and maze[r + forward_dist, c]:
                relativeState[FORWARD_DIR] = forward_dist
                break
        for right_dist in range(1,4):
            if c - right_dist >= 0 and maze[r, c - right_dist]:
                relativeState[RIGHT_DIR] = right_dist
                break
    elif h == getNextStateAndReward.WEST:
        for left_dist in range(1,4):
            if r + left_dist < maze.shape[0] and maze[r + left_dist, c]:
                relativeState[LEFT_DIR] = left_dist
                break
        for forward_dist in range(1,4):
            if c - forward_dist >= 0 and maze[r, c - forward_dist]:
                relativeState[FORWARD_DIR] = forward_dist
                break
        for right_dist in range(1,4):
            if r - right_dist >= 0 and maze[r - right_dist, c]:
                relativeState[RIGHT_DIR] = right_dist
                break
    else:
        print('\nWarning: heading incorrect!')
    return relativeState
