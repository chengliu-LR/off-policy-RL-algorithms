import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sb
import init
import getNextStateAndReward
import getRelativeState
import pandas as pd
matplotlib.use('Agg')
#maze initialization
maze = init.init_maze()
currentAbsoluteState = init.init_absolute_state(32, 29, getNextStateAndReward.WEST)
#get next absolute state and reward
nextAbsoluteState, reward = getNextStateAndReward.getNextAbsoluteStateAndReward(
                            maze = maze,
                            currentState = currentAbsoluteState,
                            action = getNextStateAndReward.FORWARD)
relativeState = getRelativeState.getRelativeState(
                            maze = maze,
                            absoluteState = nextAbsoluteState)
#terminal print
print("current absolute state:", currentAbsoluteState)
print("next absolute state:", nextAbsoluteState, '\treward:', reward)
print("relative state:", relativeState)
#text in figure
next_heading = nextAbsoluteState[2]
if next_heading == getNextStateAndReward.NORTH:
    heading_label = 'North'
elif next_heading == getNextStateAndReward.EAST:
    heading_label = 'East'
elif next_heading == getNextStateAndReward.SOUTH:
    heading_label = 'South'
elif next_heading == getNextStateAndReward.WEST:
    heading_label = 'West'
else:
    heading_label = None
#plot
fig = plt.figure(dpi = 600) #set the resolution
fig = sb.heatmap(pd.DataFrame(data = maze),
                 cmap = "Greys",
                 cbar = False)
plt.plot(currentAbsoluteState[1], currentAbsoluteState[0], '*r', label = 'Current State')  #red star
plt.plot(nextAbsoluteState[1], nextAbsoluteState[0], '.b', label = 'Next State')  #blue squared
plt.text(x = nextAbsoluteState[1] + 2,
         y = nextAbsoluteState[0] - 2,
         s = heading_label,
         size = 6,
         bbox = dict(boxstyle = 'round',
         ec=(1., 0.5, 0.5),
         fc=(1., 0.8, 0.8)))
plt.legend()
plt.title('ParrsMaze')
plt.savefig("../ParrsMaze/figures/ParrsMaze.png")
plt.close()
