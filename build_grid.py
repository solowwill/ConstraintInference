# Main file for SPI-CHRL

import gym
import warnings
import numpy as np

import lib.grid_mdp as GRIDMDP
import lib.build as build

# Defines a GridWorld from an obstacle map 
class grid_mdp():

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

        self.S = self.map.shape[0] * self.map.shape[1]
        self.A = num_actions
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
        return [state // self.map.shape[1], state % self.map.shape[1]]
    
    # Build the state features
    def build_state_feats(self):
        state_feats = np.zeros((self.S,2))
        for i in range(self.S):
            state = self.to_map[i]
            state_feats[i,0] = np.linalg.norm(state - self.to_map(self.goal_states))
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
            rewards[:,:,self.obstacle_states] = -100

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
                        rewards[i,j] = -100 
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


warnings.filterwarnings("ignore")
np.set_printoptions(precision=3, suppress=True)

NUM_ACTIONS = 4
fname='25x25_lava'


map = obstacle_map(fname)

init_state = (13,0)

mdp = GRIDMDP.grid_mdp(map, NUM_ACTIONS, init_state)

np.save(f'mdp_data/configs/grid_lava/{fname}_P.npy', mdp.P)
np.save(f'mdp_data/configs/grid_lava/{fname}_R.npy', mdp.R)
np.save(f'mdp_data/configs/grid_lava/{fname}_init.npy', mdp.init_state)
np.save(f'mdp_data/configs/grid_lava/{fname}_goal.npy', mdp.goal_states)
np.save(f'mdp_data/configs/grid_lava/{fname}_goal.npy', mdp.state_space)
np.save(f'mdp_data/configs/grid_lava/{fname}_goal.npy', mdp.act_space)

'''env = gym.make('SimpleGrid-v0', 
        obstacle_map=obstacle_map)

env.reset()
env.render()
'''

