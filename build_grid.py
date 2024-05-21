# Main file for building gridworlds
import warnings
import numpy as np
import argparse
from pathlib import Path


# Defines a Boxpushing domain from an obstacle map
class boxpush:

    FREE = 0
    WALL = 1
    RUG = 2
    GOAL = 3

    MOVES = {
            0: (-1, 0), #UP
            1: (1, 0),  #DOWN
            2: (0, -1), #LEFT
            3: (0, 1),   #RIGHT
            4: (-1, -1), # UP LEFT
            5: (1, 1), # DOWN RIGHT
            6: (-1, 1), # UP RIGHT
            7: (1, -1,) # DOWN LEFT
        }
    
    UP = 0
    DOWN = 1
    LEFT = 2 
    RIGHT = 3
    UPLEFT = 4
    DOWNRIGHT = 5
    UPRIGHT = 6
    DOWNLEFT = 7

    def __init__(self, map, num_actions, gamma=.95, H=75, stochastic=False):
        # Convert the state space into a numpy array and create action space
       
        self.map = np.asarray(map, dtype='c').astype('int')
        
        self.S = self.map.shape[0] * self.map.shape[1]
        self.A = num_actions

        # Define the goal state and init state
        self.goal_states = self.to_s(np.argwhere(self.map==boxpush.GOAL).flatten())
        self.obstacle_states = self.to_s(np.argwhere(self.map==boxpush.RUG).flatten())
        self.init_state = self.to_s(init_state)
        self.curr_state = self.init_state

        if stochastic:
                self.P = self.build_stochastic_dynamics()
                self.R = self.build_stochastic_reward()
        else:
            self.P = self.build_dynamics()
            self.R = self.build_reward()

        self.gamma = gamma
        self.H = H

        self.rmax = np.max(self.R)
        self.rmin = np.min(self.R)

        self.action_space = np.arange(self.A)
        self.state_space = self.build_state_feats()

        # Convert a state tuple to an int
    def to_s(self, state):
        return state[0] * self.map.shape[1] + state[1]

    # Convert state int to map tuple
    def to_map(self, state):
        return np.array([state // self.map.shape[1], state % self.map.shape[1]])
    
    # Build the state features
    def build_state_feats(self):
        state_feats = np.zeros((self.S,5))
        goal_feat = self.to_s(self.goal_states)
        for i in range(self.S):
            state = self.to_map(i)
            state_feats[i,0] = state[0]
            state_feats[i,1] = state[1]
            state_feats[i,2] = goal_feat[0]
            state_feats[i,3] = goal_feat[1]
            state_feats[i,4] = self.map[state[0],state[1]] == boxpush.RUG
        
        return state_feats

    # Sample a trajectory on the MDP Dgiven a policy
    def sample_trajectory(self, policy):

        sars = []

        self.curr_state = self.init_state
        g = 0
        i = 0

        while np.all(self.curr_state != self.goal_states):

            curr_action = np.random.choice(np.arange(self.A), 1, p=policy[self.curr_state,:])[0]

            next_state = np.random.choice(np.arange(self.S), 1, p=self.P[self.curr_state,curr_action,:])[0]

            if len(self.R.shape) == 3:
                reward = self.R[self.curr_state, curr_action, next_state]
            else: 
                reward = self.R[self.curr_state, curr_action]

            sars.append( [self.curr_state, curr_action, reward, next_state] )

            g += self.gamma**i * reward
            i += 1

            if len(sars) >= self.H:
                break

            self.curr_state = next_state
            
        return np.array(sars), g
    
    # Return the uniformly random policy
    def rand_pol(self):
        return (1 / self.A) * np.ones((self.S,self.A))
        

    # Build stochastic gridworld dynamics 
    def build_stochastic_dynamics(self):

        x = self.map.shape[1]

        map = np.copy(self.map)
        map[map != boxpush.WALL] = boxpush.FREE
        map = map.flatten()

        dynamics = np.zeros((map.shape[0], self.A, map.shape[0]))

        for i in range(dynamics.shape[0]):
            
            # If in Wall, we stay there
            if map[i] == boxpush.WALL:
                dynamics[i,:,i] = 1

            else:
                
                # If a up move is valid
                if (i - x) >= 0 and map[i-x] != boxpush.WALL:
                    dynamics[i,boxpush.UP,i-x] = .75
                    dynamics[i,boxpush.DOWN,i-x] = 0
                    dynamics[i,boxpush.LEFT,i-x] = .05
                    dynamics[i,boxpush.RIGHT,i-x] = .05
                    dynamics[i,boxpush.UPLEFT,i-x] = .05
                    dynamics[i,boxpush.DOWNRIGHT,i-x] = .025
                    dynamics[i,boxpush.UPRIGHT,i-x] = .05
                    dynamics[i,boxpush.DOWNLEFT,i-x] = .025
                else:
                    dynamics[i,boxpush.UP,i] += .75
                    dynamics[i,boxpush.DOWN,i] += 0
                    dynamics[i,boxpush.LEFT,i] += .05
                    dynamics[i,boxpush.RIGHT,i] += .05
                    dynamics[i,boxpush.UPLEFT,i] += .05
                    dynamics[i,boxpush.DOWNRIGHT,i] += .025
                    dynamics[i,boxpush.UPRIGHT,i] += .05
                    dynamics[i,boxpush.DOWNLEFT,i] += .025

                # If a down move is valid
                if (i + x) < dynamics.shape[0] and map[i+x] != boxpush.WALL: 
                    dynamics[i,boxpush.UP,i+x] = 0
                    dynamics[i,boxpush.DOWN,i+x] = .75
                    dynamics[i,boxpush.LEFT,i+x] = .05
                    dynamics[i,boxpush.RIGHT,i+x] = .05
                    dynamics[i,boxpush.UPLEFT,i+x] = .025
                    dynamics[i,boxpush.DOWNRIGHT,i+x] = .05
                    dynamics[i,boxpush.UPRIGHT,i+x] = .025
                    dynamics[i,boxpush.DOWNLEFT,i+x] = .05
                else:
                    dynamics[i,boxpush.UP,i] += 0
                    dynamics[i,boxpush.DOWN,i] += .75
                    dynamics[i,boxpush.LEFT,i] += .05
                    dynamics[i,boxpush.RIGHT,i] += .05
                    dynamics[i,boxpush.UPLEFT,i] += .025
                    dynamics[i,boxpush.DOWNRIGHT,i] += .05
                    dynamics[i,boxpush.UPRIGHT,i] += .025
                    dynamics[i,boxpush.DOWNLEFT,i] += .05

                # If left move is valid
                if (i-1) % x != (x-1) and map[i-1] != boxpush.WALL:
                    dynamics[i,boxpush.UP,i-x] = .05
                    dynamics[i,boxpush.DOWN,i-x] = .05
                    dynamics[i,boxpush.LEFT,i-x] = .75
                    dynamics[i,boxpush.RIGHT,i-x] = 0
                    dynamics[i,boxpush.UPLEFT,i-x] = .05
                    dynamics[i,boxpush.DOWNRIGHT,i-x] = .025
                    dynamics[i,boxpush.UPRIGHT,i-x] = .025
                    dynamics[i,boxpush.DOWNLEFT,i-x] = .05
                else:
                    dynamics[i,boxpush.UP,i] += .05
                    dynamics[i,boxpush.DOWN,i] += .05
                    dynamics[i,boxpush.LEFT,i] += .75
                    dynamics[i,boxpush.RIGHT,i] += 0
                    dynamics[i,boxpush.UPLEFT,i] += .05
                    dynamics[i,boxpush.DOWNRIGHT,i] += .025
                    dynamics[i,boxpush.UPRIGHT,i] += .025
                    dynamics[i,boxpush.DOWNLEFT,i] += .05

                # If right move is valid 
                if (i+1) % x != 0 and map[i+1] != boxpush.WALL:
                    dynamics[i,boxpush.UP,i-x] = .05
                    dynamics[i,boxpush.DOWN,i-x] = .05
                    dynamics[i,boxpush.LEFT,i-x] = 0
                    dynamics[i,boxpush.RIGHT,i-x] = .75
                    dynamics[i,boxpush.UPLEFT,i-x] = .025
                    dynamics[i,boxpush.DOWNRIGHT,i-x] = .05
                    dynamics[i,boxpush.UPRIGHT,i-x] = .05
                    dynamics[i,boxpush.DOWNLEFT,i-x] = .025
                else:
                    dynamics[i,boxpush.UP,i] += .05
                    dynamics[i,boxpush.DOWN,i] += .05
                    dynamics[i,boxpush.LEFT,i] += 0
                    dynamics[i,boxpush.RIGHT,i] += .75
                    dynamics[i,boxpush.UPLEFT,i] += .025
                    dynamics[i,boxpush.DOWNRIGHT,i] += .05
                    dynamics[i,boxpush.UPRIGHT,i] += .05
                    dynamics[i,boxpush.DOWNLEFT,i] += .025

                # If up and left move is valid
                if (i - x) >= 0 and (i-1) % x != (x-1)and map[i-x] != boxpush.WALL:
                    dynamics[i,boxpush.UP,i-x] = .05
                    dynamics[i,boxpush.DOWN,i-x] = .025
                    dynamics[i,boxpush.LEFT,i-x] = .025
                    dynamics[i,boxpush.RIGHT,i-x] = .05
                    dynamics[i,boxpush.UPLEFT,i-x] = .75
                    dynamics[i,boxpush.DOWNRIGHT,i-x] = 0
                    dynamics[i,boxpush.UPRIGHT,i-x] = .05
                    dynamics[i,boxpush.DOWNLEFT,i-x] = .5
                else: 
                    dynamics[i,boxpush.UP,i] += .05
                    dynamics[i,boxpush.DOWN,i] += .025
                    dynamics[i,boxpush.LEFT,i] += .025
                    dynamics[i,boxpush.RIGHT,i] += .05
                    dynamics[i,boxpush.UPLEFT,i] += .75
                    dynamics[i,boxpush.DOWNRIGHT,i] += 0
                    dynamics[i,boxpush.UPRIGHT,i] += .05
                    dynamics[i,boxpush.DOWNLEFT,i] += .5

                # If down and right move is valid
                if (i + x) < dynamics.shape[0] and (i+1) % x != 0 and map[i+x] != boxpush.WALL:
                    dynamics[i,boxpush.UP,i-x] = .025
                    dynamics[i,boxpush.DOWN,i-x] = .05
                    dynamics[i,boxpush.LEFT,i-x] = .025
                    dynamics[i,boxpush.RIGHT,i-x] = .05
                    dynamics[i,boxpush.UPLEFT,i-x] = 0
                    dynamics[i,boxpush.DOWNRIGHT,i-x] = .75
                    dynamics[i,boxpush.UPRIGHT,i-x] = .05
                    dynamics[i,boxpush.DOWNLEFT,i-x] = .05
                else:
                    dynamics[i,boxpush.UP,i] += .025
                    dynamics[i,boxpush.DOWN,i] += .05
                    dynamics[i,boxpush.LEFT,i] += .025
                    dynamics[i,boxpush.RIGHT,i] += .05
                    dynamics[i,boxpush.UPLEFT,i] += 0
                    dynamics[i,boxpush.DOWNRIGHT,i] += .75
                    dynamics[i,boxpush.UPRIGHT,i] += .05
                    dynamics[i,boxpush.DOWNLEFT,i] += .05

                # If up and right move is valid 
                if (i - x) >= 0 and (i+1) % x != 0 and map[i-x] != boxpush.WALL:
                    dynamics[i,boxpush.UP,i-x] = .05
                    dynamics[i,boxpush.DOWN,i-x] = .025
                    dynamics[i,boxpush.LEFT,i-x] = .025
                    dynamics[i,boxpush.RIGHT,i-x] = .05
                    dynamics[i,boxpush.UPLEFT,i-x] = .05
                    dynamics[i,boxpush.DOWNRIGHT,i-x] = .05
                    dynamics[i,boxpush.UPRIGHT,i-x] = .75
                    dynamics[i,boxpush.DOWNLEFT,i-x] = 0
                else: 
                    dynamics[i,boxpush.UP,i] += .05
                    dynamics[i,boxpush.DOWN,i] += .025
                    dynamics[i,boxpush.LEFT,i] += .025
                    dynamics[i,boxpush.RIGHT,i] += .05
                    dynamics[i,boxpush.UPLEFT,i] += .05
                    dynamics[i,boxpush.DOWNRIGHT,i] += .05
                    dynamics[i,boxpush.UPRIGHT,i] += .75
                    dynamics[i,boxpush.DOWNLEFT,i] += 0

                # If down and left move is valid 
                if (i + x) < dynamics.shape[0] and (i-1) % x != (x-1) and map[i+x] != boxpush.WALL:
                    dynamics[i,boxpush.UP,i-x] = .025
                    dynamics[i,boxpush.DOWN,i-x] = .05
                    dynamics[i,boxpush.LEFT,i-x] = .05
                    dynamics[i,boxpush.RIGHT,i-x] = .025
                    dynamics[i,boxpush.UPLEFT,i-x] = .05
                    dynamics[i,boxpush.DOWNRIGHT,i-x] = .05
                    dynamics[i,boxpush.UPRIGHT,i-x] = 0
                    dynamics[i,boxpush.DOWNLEFT,i-x] = .75
                else:
                    dynamics[i,boxpush.UP,i] += .025
                    dynamics[i,boxpush.DOWN,i] += .05
                    dynamics[i,boxpush.LEFT,i] += .05
                    dynamics[i,boxpush.RIGHT,i] += .025
                    dynamics[i,boxpush.UPLEFT,i] += .05
                    dynamics[i,boxpush.DOWNRIGHT,i] += .05
                    dynamics[i,boxpush.UPRIGHT,i] += 0
                    dynamics[i,boxpush.DOWNLEFT,i] += .75


        return dynamics

    # Build deterministic dynamics model
    def build_dynamics(self):

        x = self.map.shape[1]

        map = np.copy(self.map)
        map[map != boxpush.WALL] = boxpush.FREE
        map = map.flatten()

        dynamics = np.zeros((map.shape[0], self.A, map.shape[0]))

        for i in range(dynamics.shape[0]):
            
            # If state is wall, do not move
            if map[i] == boxpush.WALL:
                dynamics[i,:,i] = 1
            else:
                # If up move is valid
                if (i - x) >= 0 and map[i-x] != boxpush.WALL:
                    dynamics[i,boxpush.UP,i-x] = 1
                else: 
                    dynamics[i,boxpush.UP,i] = 1
                
                # If down move is valid
                if (i + x) < dynamics.shape[0] and map[i+x] != boxpush.WALL:
                    dynamics[i,boxpush.DOWN,i+x] = 1
                else:
                    dynamics[i,boxpush.DOWN,i] = 1

                # If left move is valid
                if (i-1) % x != (x-1) and map[i-1] != boxpush.WALL:
                    dynamics[i,boxpush.LEFT,i-1] = 1
                else: 
                    dynamics[i,boxpush.LEFT,i] = 1

                # If right move is valid
                if (i+1) % x != 0 and map[i+1] != boxpush.WALL:
                    dynamics[i,boxpush.RIGHT,i+1] = 1
                else: 
                    dynamics[i,boxpush.RIGHT,i] = 1

                # If up and left move is valid
                if (i - x) >= 0 and (i-1) % x != (x-1)and map[i-x] != boxpush.WALL:
                    dynamics[i,boxpush.UPLEFT,i-x-1] = 1
                else: 
                    dynamics[i,boxpush.UPLEFT,i] = 1

                # If down and right move is valid
                if (i + x) < dynamics.shape[0] and (i+1) % x != 0 and map[i+x] != boxpush.WALL:
                    dynamics[i,boxpush.DOWNRIGHT,i+x+1] = 1
                else:
                    dynamics[i,boxpush.DOWNRIGHT,i] = 1

                # If up and right move is valid 
                if (i - x) >= 0 and (i+1) % x != 0 and map[i-x] != boxpush.WALL:
                    dynamics[i,boxpush.UPRIGHT,i-x+1] = 1
                else: 
                    dynamics[i,boxpush.UPRIGHT,i] = 1

                # If down and left move is valid 
                if (i + x) < dynamics.shape[0] and (i-1) % x != (x-1) and map[i+x] != boxpush.WALL:
                    dynamics[i,boxpush.DOWNLEFT,i+x-1] = 1
                else:
                    dynamics[i,boxpush.DOWNLEFT,i] = 1

        return dynamics
    
    # Build stochastic reward model 
    def build_stochastic_reward(self):

        rewards = np.zeros((self.S, self.A, self.S))-1
        
        # Set all goal states to 0
        rewards[:,:,self.goal_states] = 0

        return rewards

    # Build deterministc reward model
    def build_reward(self):
        map = self.map.flatten()
        rewards = np.zeros((self.S, self.A)) - 1

        # If the next state is a goal, zero reward
        for i in range(rewards.shape[0]):
            for j in range(rewards.shape[1]):
                next_states = np.argwhere(self.P[i,j] != 0)
                for ns in next_states:
                    if map[ns] == boxpush.GOAL:
                        rewards[i,j] = 0
        return rewards
        

# Defines a GridWorld from an obstacle map 
class grid_mdp:

    FREE = 0
    WALL = 1
    LAVA = 2
    GOAL = 3
    MOVES = {
            0: (-1, 0), #UP
            1: (1, 0),  #DOWN
            2: (0, -1), #LEFT
            3: (0, 1)   #RIGHT
        }

    # Set indices of trajectory tuples 
    S = 0
    A = 1
    R = 2
    S_P = 3


    def __init__(self, map, num_actions, init_state, gamma=.95, H=75, stochastic=False):
        
        # Convert the state space to a numpy array and create action space
        self.map = np.asarray(map, dtype='c').astype('int')
        
        self.S = self.map.shape[0] * self.map.shape[1]
        self.A = num_actions

        # Define the goal state and init state
        self.goal_states = self.to_s(np.argwhere(self.map==grid_mdp.GOAL).flatten())
        self.obstacle_states = self.to_s(np.argwhere(self.map==grid_mdp.LAVA).flatten())
        self.init_state = self.to_s(init_state)
        self.curr_state = self.init_state

        if stochastic:
                self.P = self.build_stochastic_dynamics()
                self.R = self.build_stochastic_reward()
        else:
            self.P = self.build_dynamics()
            self.R = self.build_reward()
        self.gamma = gamma
        self.H = H

        self.rmax = np.max(self.R)
        self.rmin = np.min(self.R)

        self.action_space = np.arange(self.A)
        self.state_space = self.build_state_feats()

    # Convert a state tuple to an int
    def to_s(self, state):
        return state[0] * self.map.shape[1] + state[1]

    # Convert state int to map tuple
    def to_map(self, state):
        return np.array([state // self.map.shape[1], state % self.map.shape[1]])
    
    # Build the state features
    def build_state_feats(self):
        state_feats = np.zeros((self.S,2))
        for i in range(self.S):
            state = self.to_map(i)
            state_feats[i,0] = np.linalg.norm(state - self.to_map(self.goal_states),ord=1)
            state_feats[i,1] = self.map[state[0],state[1]] == grid_mdp.LAVA
        
        return state_feats


    # Sample a trajectory on the MDP Dgiven a policy
    def sample_trajectory(self, policy):

        sars = []

        self.curr_state = self.init_state
        g = 0
        i = 0

        while np.all(self.curr_state != self.goal_states):

            curr_action = np.random.choice(np.arange(self.A), 1, p=policy[self.curr_state,:])[0]

            next_state = np.random.choice(np.arange(self.S), 1, p=self.P[self.curr_state,curr_action,:])[0]

            if len(self.R.shape) == 3:
                reward = self.R[self.curr_state, curr_action, next_state]
            else: 
                reward = self.R[self.curr_state, curr_action]

            sars.append( [self.curr_state, curr_action, reward, next_state] )

            g += self.gamma**i * reward
            i += 1

            if len(sars) >= self.H:
                break

            self.curr_state = next_state
            
        return np.array(sars), g
    
    # Return the uniformly random policy
    def rand_pol(self):
        return (1 / self.A) * np.ones((self.S,self.A))
        

    # Build stochastic gridworld dynamics 
    def build_stochastic_dynamics(self):

        x = self.map.shape[1]

        map = np.copy(self.map)
        map[map != grid_mdp.WALL] = grid_mdp.FREE
        map = map.flatten()

        dynamics = np.zeros((map.shape[0], self.A, map.shape[0]))

        for i in range(dynamics.shape[0]):
            
            # If in Wall, we stay there
            if map[i] == grid_mdp.WALL:
                dynamics[i,:,i] = 1

            else:
                
                # If a up move is valid
                if (i - x) >= 0 and map[i-x] != grid_mdp.WALL:
                    dynamics[i,0,i-x] = .75
                    dynamics[i,1,i-x] = .05
                    dynamics[i,2,i-x] = .1
                    dynamics[i,3,i-x] = .1
                else:
                    dynamics[i,0,i] += .75
                    dynamics[i,1,i] += .05
                    dynamics[i,2,i] += .1
                    dynamics[i,3,i] += .1

                # If a down move is valid
                if (i + x) < dynamics.shape[0] and map[i+x] != grid_mdp.WALL: 
                    dynamics[i,0,i+x] = .05
                    dynamics[i,1,i+x] = .75
                    dynamics[i,2,i+x] = .1
                    dynamics[i,3,i+x] = .1
                else:
                    dynamics[i,0,i] += .05
                    dynamics[i,1,i] += .75
                    dynamics[i,2,i] += .1
                    dynamics[i,3,i] += .1

                # If left move is valid
                if (i-1) % x != (x-1) and map[i-1] != grid_mdp.WALL:
                    dynamics[i,0,i-1] = .1
                    dynamics[i,1,i-1] = .1
                    dynamics[i,2,i-1] = .75
                    dynamics[i,3,i-1] = .05
                else:
                    dynamics[i,0,i] += .1
                    dynamics[i,1,i] += .1
                    dynamics[i,2,i] += .75
                    dynamics[i,3,i] += .05

                # If right move is valid 
                if (i+1) % x != 0 and map[i+1] != grid_mdp.WALL:
                    dynamics[i,0,i+1] = .1
                    dynamics[i,1,i+1] = .1
                    dynamics[i,2,i+1] = .05
                    dynamics[i,3,i+1] = .75
                else:
                    dynamics[i,0,i] += .1
                    dynamics[i,1,i] += .1
                    dynamics[i,2,i] += .05
                    dynamics[i,3,i] += .75


        return dynamics

    # Build deterministic dynamics model
    def build_dynamics(self):

        x = self.map.shape[1]

        map = np.copy(self.map)
        map[map != grid_mdp.WALL] = grid_mdp.FREE
        map = map.flatten()

        dynamics = np.zeros((map.shape[0], self.A, map.shape[0]))

        for i in range(dynamics.shape[0]):
            
            # If state is wall, do not move
            if map[i] == grid_mdp.WALL:
                dynamics[i,:,i] = 1

            else:
                
                # If up move is valid
                if (i - x) >= 0 and map[i-x] != grid_mdp.WALL:
                        dynamics[i,0,i-x] = 1
                else: 
                    dynamics[i,0,i] = 1
                
                # If down move is valid
                if (i + x) < dynamics.shape[0] and map[i+x] != grid_mdp.WALL:
                        dynamics[i,1,i+x] = 1
                else:
                    dynamics[i,1,i] = 1

                # If left move is valid
                if (i-1) % x != (x-1) and map[i-1] != grid_mdp.WALL:
                        dynamics[i,2,i-1] = 1
                else: 
                    dynamics[i,2,i] = 1

                # If right move is valid
                if (i+1) % x != 0 and map[i+1] != grid_mdp.WALL:
                        dynamics[i,3,i+1] = 1
                else: 
                    dynamics[i,3,i] = 1

        return dynamics
    
    # Build stochastic reward model 
    def build_stochastic_reward(self):

        rewards = np.zeros((self.S, self.A, self.S))-1
        
        # Set all goal states to +1
        rewards[:,:,self.goal_states] = 0

        # Set all obstacle states to -1
        if self.obstacle_states != None:
            rewards[:,:,self.obstacle_states] = -1

        return rewards

    # Build deterministc reward model
    def build_reward(self):

        map = self.map.flatten()
        
        rewards = np.zeros((self.S, self.A)) - 1

        for i in range(rewards.shape[0]):
            for j in range(rewards.shape[1]):

                next_states = np.argwhere(self.P[i,j] != 0)
                
                for ns in next_states:

                    if map[ns] == grid_mdp.GOAL:
                        rewards[i,j] = 0
                    elif map[ns] == grid_mdp.LAVA:
                        rewards[i,j] = -1 
        return rewards

# Return a variety of obstacle maps for GridWorld MDPs
def obstacle_map(fname):

    if fname == '7x8':
        return     [
        "01000000",
        "01000000",
        "01001000",
        "00001000",
        "00001000",
        "00001000",
        "00001003"
                    ]
    elif fname == '7x8_lava':
        return     [
        "01000000",
        "01000000",
        "01001000",
        "00001000",
        "00001000",
        "00002000",
        "00002003"
                    ] 
    elif fname == '5x5_lava':
        return  [
        "00000",
        "00000",
        "00100",
        "02000",
        "20003",
                ]
    elif fname == '5x5':
        return  [
        "00000",
        "00000",
        "00100",
        "00000",
        "00003",
                ]
    elif fname == '25x25_lava':
        return [
            "0000000001000000000000000",
            "0010000001000000000000000",
            "0010000001000000000000000",
            "0010000001000000000000000",
            "0010000000000000000000000",
            "0010001000000000000000000",
            "0010001000000001111110000",
            "0000001000000000000000000",
            "0000001000000000000000000",
            "0000001000000000000000000",
            "0220000000002000011100000",
            "0000000000001100020000000",
            "0000112211002000000000003",
            "0000000000002000100000000",
            "0000200000000000100000000",
            "0000200000010000100000000",
            "0100000000010000000000000",
            "0100000000010000000010000",
            "0100000000010000000010000",
            "0100000000000000000010000",
            "0100000000111000000010000",
            "0000000000000000000010000",
            "0000100000000000000000000",
            "0000100000000011111000000",
            "0000100000000000000000000",
        ]   
    elif fname == '25x25':
        return [
            "0000000001000000000000000",
            "0010000001000000000000000",
            "0010000001000000000000000",
            "0010000001000000000000000",
            "0010000000000000000000000",
            "0010001000000000000000000",
            "0010001000000001111110000",
            "0000001000000000000000000",
            "0000001000000000000000000",
            "0000001000000000000000000",
            "0000000000000000011100000",
            "0000000000000000000000000",
            "0000111111000000100000000",
            "0000000000000000100000000",
            "0000000000000000100000000",
            "0000000000010000100000000",
            "0100000000010000000000000",
            "0100000000010000000010000",
            "0100000000010000000010000",
            "0100000000000000000010000",
            "0100000000111000000010000",
            "0000000000000000000010000",
            "0000100000000000000000000",
            "0000100000000011111000000",
            "0000100000000000000000003",
        ]   
    elif fname == '6x6_lava':
        return [
            "000000",
            "000000",
            "222200",
            "222200",
            "000000",
            "300000"
        ]

    elif fname == '8x10_lava':
        return [
            "0000000000",
            "0000000000",
            "0000000000",
            "2222220000",
            "2222220000",
            "0000000000",
            "0000000000",
            "3000000000"
        ]
    elif fname == '10x10_rug':
        return [
            "2200000000",
            "2200000220",
            "0200000020",
            "0002222000",
            "0002222000",
            "1002222003",
            "0002222000",
            "0002222000",
            "0000000000",
            "0002220000",
        ]

    


warnings.filterwarnings("ignore")
np.set_printoptions(precision=3, suppress=True)

NUM_ACTIONS = 4

parser = argparse.ArgumentParser()
parser.add_argument("--env_type", type=str, default="5x5_lava", help="Grid name")
parser.add_argument("--env", type=str, default="5x5_lava", help="Grid name")
parser.add_argument("--stochastic", type=bool, default=False)
args = parser.parse_args()


map = obstacle_map(args.env)

init_state = (0,0)

mdp = grid_mdp(map, NUM_ACTIONS, init_state,stochastic=args.stochastic)

Path(f'configs/{args.env_type}/').mkdir(parents=True,exist_ok=True)

if args.stochastic:
    np.save(f'configs/{args.env_type}/stochastic_{args.env}_P.npy', mdp.P)
    np.save(f'configs/{args.env_type}/stochastic_{args.env}_R.npy', mdp.R)
    np.save(f'configs/{args.env_type}/stochastic_{args.env}_init.npy', mdp.init_state)
    np.save(f'configs/{args.env_type}/stochastic_{args.env}_goal.npy', mdp.goal_states)
    np.save(f'configs/{args.env_type}/stochastic_{args.env}_state_space.npy', mdp.state_space)
    np.save(f'configs/{args.env_type}/stochastic_{args.env}_action_space.npy', mdp.action_space)
else:

    np.save(f'configs/{args.env_type}/{args.env}_P.npy', mdp.P)
    np.save(f'configs/{args.env_type}/{args.env}_R.npy', mdp.R)
    np.save(f'configs/{args.env_type}/{args.env}_init.npy', mdp.init_state)
    np.save(f'configs/{args.env_type}/{args.env}_goal.npy', mdp.goal_states)
    np.save(f'configs/{args.env_type}/{args.env}_state_space.npy', mdp.state_space)
    np.save(f'configs/{args.env_type}/{args.env}_action_space.npy', mdp.action_space)



