import numpy as np
#watch out the corner note diffrence from matlab!!!!!
#Load Maze
#raw maze_data[i,j] == 0: WALL
#raw paze_data[i,j] == 255: BLANK

#headings
NORTH  = 0
EAST   = 1
SOUTH  = 2
WEST   = 3

def init_maze():
    #transfer to bool numpy array
    maze_dir = "../ParrsMaze/maze_data.txt"
    maze = np.loadtxt(maze_dir, dtype = np.bool)
    #make walls true
    WALL_TEST = maze[0,0]
    if WALL_TEST is not True:
        maze = ~ maze
    return maze

def init_absolute_state(row, column, heading):
    #currentAbsoluteState initialization
    return  np.array([row, column, heading], dtype = np.short)
