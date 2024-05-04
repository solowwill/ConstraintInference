# grid_mdp.py
# Contains all code for tabular MDP

import numpy as np


# General class for an MDP 
class mdp():

    # Set indices of trajectory tuples 
    S = 0
    A = 1
    R = 2
    S_P = 3

    def __init__(self, P, R, init_state, goal_states, gamma=.95, H=75):

        self.goal_states = goal_states
        self.init_state = init_state

        self.S = P.shape[0]
        self.A = P.shape[1]
        self.P = P
        self.R = R
        self.gamma = gamma

        self.H = H

        self.rmax = np.max(self.R)
        self.rmin = np.min(self.R)

    # Sample a trajectory given a policy
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


            g += (self.gamma**i * reward)
            i += 1

            if len(sars) >= self.H:
                break

            self.curr_state = next_state
            
        return np.array(sars), g
    

    # Return the uniformly random policy 
    def rand_pol(self):
        return (1 / self.A) * np.ones((self.S,self.A))



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


    def __init__(self, map, num_actions, init_state, gamma=.95, H=75):
        
        # Convert the state space to a numpy array and create action space
        self.state_space = np.asarray(map, dtype='c').astype('int')
        self.action_space = np.arange(num_actions)

        # Define the goal state and init state
        self.goal_states = self.to_s(np.argwhere(self.state_space==grid_mdp.GOAL).flatten())
        self.obstacle_states = self.to_s(np.argwhere(self.state_space==grid_mdp.LAVA).flatten())
        self.init_state = self.to_s(init_state)
        self.curr_state = self.init_state


        self.S = self.state_space.shape[0] * self.state_space.shape[1]
        self.A = num_actions
        self.P = self.build_stochastic_dynamics()
        self.R = self.build_stochastic_reward()
        self.gamma = gamma
        self.H = H

        self.rmax = np.max(self.R)
        self.rmin = np.min(self.R)

    # Convert a state tuple to an int
    def to_s(self, state):
        return state[0] * self.state_space.shape[1] + state[1]
    

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

        x = self.state_space.shape[1]

        map = np.copy(self.state_space)
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

        x = self.state_space.shape[1]

        map = np.copy(self.state_space)
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

        rewards = np.zeros((self.S, self.A, self.S))
        
        # Set all goal states to +1
        rewards[:,:,self.goal_states] = 1

        # Set all obstacle states to -1
        if self.obstacle_states != None:
            rewards[:,:,self.obstacle_states] = -1

        return rewards

    # Build deterministc reward model
    def build_reward(self):

        map = self.state_space.flatten()
        
        rewards = np.zeros((self.S, self.A))

        for i in range(rewards.shape[0]):
            for j in range(rewards.shape[1]):

                next_states = np.argwhere(self.P[i,j] != 0)
                
                for ns in next_states:

                    if map[ns] == grid_mdp.GOAL:
                        rewards[i,j] = 1
                    elif map[ns] == grid_mdp.LAVA:
                        rewards[i,j] = -1 
        return rewards
    

