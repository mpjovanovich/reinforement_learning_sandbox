## Mike Jovanovich
## Basic RL demos with python
## Command line options:
## Example Usage:
## ...

STATE_SIZE_X = 10
STATE_SIZE_Y = 1
EPSILON = .05       ## chance of random move
GAMMA = .5         ## discount factor
## LAMBDA = .5         ## credit assignment rate
ALPHA = .5          ## learning rate
GOAL_REWARD = 1

import numpy as np
import random

## It's easier to just use a vector and translate 2d points
def translatePoint( p ):
    return p[1] * STATE_SIZE_X + p[0] 

def printVals( M ):
    for i in range( STATE_SIZE_Y ):
        for j in range( STATE_SIZE_X ):
            print( M[translatePoint( [j,i] )], end=" " )
        print()

def TD_zero( cur_state_index, next_state_index, reward, V ):
    target = reward + GAMMA * V[next_state_index]
    td_error = target - V[cur_state_index]
    V[cur_state_index] += ALPHA * td_error

##         V[cur_state_index] += V[cur_state_index] + ALPHA * td_error
##         print(target)
##         print(td_error)
##         print(new_val)

## def TD_lambda( cur_state_index, next_state_index, reward, V ):
##     delta = reward + GAMMA * V[next_state_index] - V[cur_state_index]
##     ALPHA * (reward + GAMMA * delta
##     print(delta)
## 

def main():
    ## Pick the goal state at random
    goal = [random.randint( 0,STATE_SIZE_X-1 ), random.randint( 0,STATE_SIZE_Y-1 )]
    print("Goal: " + str(goal))
    print()

    ## I'm calling this V, assuming TD(0) algorithm
    ## If we use Q learning this is really the Q function

    ## TODO: modify to use optimistic critic
##     V = np.ones( STATE_SIZE_X * STATE_SIZE_Y)
    V = np.zeros( STATE_SIZE_X * STATE_SIZE_Y)

    for trial in range(10):
        ## Start at point 0,0 for now
        cur_point = [0,0]
        new_point = [0,0]

        while True:

            ## Determine if we are in the goal state
            ## And perform value function update
            if cur_point[0] == goal[0] and cur_point[1] == goal[1]:
                V[translatePoint(cur_point)] = GOAL_REWARD
    ##             TD_zero( translatePoint( cur_point ), translatePoint( new_point ), 0, V )
                break
            else:
                TD_zero( translatePoint( cur_point ), translatePoint( new_point ), 0, V )

            ## Update cur_point to new_point
            cur_point = new_point

            ## Get possible next states/actions
            candidates = []

            ## Left, right, down, up
            if cur_point[0]-1 >= 0:
                candidates.append( [cur_point[0]-1,cur_point[1]] )
            if cur_point[0]+1 < STATE_SIZE_X:
                candidates.append( [cur_point[0]+1,cur_point[1]] )
            if cur_point[1]-1 >= 0:
                candidates.append( [cur_point[0],cur_point[1]-1] )
            if cur_point[1]+1 < STATE_SIZE_Y:
                candidates.append( [cur_point[0],cur_point[1]+1] )

            ## Init the state/action choice
            candidate_choice = 0
            max_val = -99


            if random.uniform(0,1) < EPSILON:
                ## Explore
                candidate_choice = random.randint(0,len(candidates)-1)
                pass
            else:
                ## Take the greedy state/action
                for i in range(len(candidates)):
                    if V[translatePoint(candidates[i])] > max_val:
                        max_val = V[translatePoint(candidates[i])]
                        candidate_choice = i
                pass

            new_point = candidates[candidate_choice]

            ## Perform value function update
            TD_zero( translatePoint( cur_point ), translatePoint( new_point ), 0, V )

            ## Update cur_point to new_point
            cur_point = new_point

    ##         printVals( V )
    ##         break

        printVals( V )


main()
