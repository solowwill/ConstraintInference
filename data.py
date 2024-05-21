# Main data file for creating data from MDP
import numpy as np
import argparse
from pathlib import Path

# General class for an MDP 
class mdp:

    # Set indices of trajectory tuples 
    I = 0
    S = 1
    A = 2
    R = 3
    S_P = 4

    def __init__(self, P, R, init_state, goal_states, state_space=None, action_space=None, gamma=.95, H=100):

        self.goal_states = goal_states
        self.init_state = init_state

        self.S = P.shape[0]
        self.A = P.shape[1]
        self.P = P
        self.R = R
        self.gamma = gamma

        self.H = H

        self.rmax = np.max(np.abs(self.R))
        self.rmin = -np.max(np.abs(self.R))

        self.state_space = state_space
        self.action_space = action_space

    # Sample a trajectory given a policy
    def sample_trajectory(self, policy):

        sars = []

        self.curr_state = self.init_state

        g = 0
        i = 0

        while len(sars) < self.H:

            curr_action = np.random.choice(np.arange(self.A), 1, p=policy[self.curr_state,:])[0]

            next_state = np.random.choice(np.arange(self.S), 1, p=self.P[self.curr_state,curr_action,:])[0]

            if len(self.R.shape) == 3:
                reward = self.R[self.curr_state, curr_action, next_state]
            else: 
                reward = self.R[self.curr_state, curr_action]

            sars.append( [i, self.curr_state, curr_action, reward, next_state] )


            g += (self.gamma**i * reward)
            i += 1

            if (self.curr_state == self.goal_states).any():
                break

            self.curr_state = next_state

            
        return np.array(sars), g
    

    # Return the uniformly random policy 
    def rand_pol(self):
        return (1 / self.A) * np.ones((self.S,self.A))


# General class for a CMDP
# Constaints are a k+|S|+|A| ndarray with a 1 indicating if a 
# state/action is a constraint, or a non-zero value indicating a threshold 
# for a feature being a constraint
class cmdp:
    FEAT = 0
    STATE = 1
    ACTION = 2
    def __init__(self, mdp, constraints):

        self.mdp = mdp
        self.feat_constraints = constraints[cmdp.FEAT]
        self.k = self.mdp.state_space.shape[-1]
        self.constraints = self.build_constraints(constraints)

        self.mu0 = np.zeros(self.mdp.S)
        self.mu0[self.mdp.init_state] = 1

        self.build_constrained_reward()
        #self.build_constrained_dynamics()

    # Build constraints, constraints is list of three ndarrays for feature/state/action
    def build_constraints(self, constraints):
        feat_c = []
        for i in range(self.k):
            if constraints[cmdp.FEAT][i] > 0:
                feat_k = np.argwhere(self.mdp.state_space[:,i] >= constraints[cmdp.FEAT][i]).flatten()
                [feat_c.append(feat_k[j]) for j in range(len(feat_k))]

        feat_s = np.argwhere(constraints[cmdp.STATE] > 0).flatten()
        feat_a = np.argwhere(constraints[cmdp.ACTION] > 0).flatten()

        return [feat_c, feat_s, feat_a]


    # Build the constrained dynamics 
    def build_constrained_dynamics(self):

        # State-feature constraints
        for s in self.constraints[cmdp.FEAT]:
            self.mdp.P[:,:,s] = np.zeros((self.mdp.S, self.mdp.A))
            self.mdp.P[s,:,:] = np.zeros((self.mdp.A, self.mdp.S))
        # State constraints
        for s in self.constraints[cmdp.STATE]:
            self.mdp.P[:,:,s] = np.zeros((self.mdp.S, self.mdp.A))
            self.mdp.P[s,:,:] = np.zeros((self.mdp.A, self.mdp.S))
        # Action constraints
        for a in self.constraints[cmdp.ACTION]:
            self.mdp.A[:,a,:] = np.zeros((self.mdp.S, self.mdp.S))

    # Build the constrained reward function 
    def build_constrained_reward(self):

        # Build deterministc reward model
        if len(self.mdp.R.shape) == 2:
            for s in range(self.mdp.S):
                for a in range(self.mdp.A):
                    
                    next_states = np.argwhere(self.mdp.P[s,a] != 0)
                    if a in self.constraints[cmdp.ACTION]:
                        self.mdp.R[:,a] = - self.mdp.rmax / (1-self.mdp.gamma)
                    
                    for ns in next_states:
                        if ns in self.constraints[cmdp.FEAT] or ns in self.constraints[cmdp.STATE]:
                            self.mdp.R[s,a] = - self.mdp.rmax / (1-self.mdp.gamma)

        if len(self.mdp.R.shape) == 3:
            for ns in self.constraints[cmdp.FEAT]:
                self.mdp.R[:,:,ns] = - self.mdp.rmax / (1-self.mdp.gamma)
            for ns in self.constraints[cmdp.STATE]:
                self.mdp.R[:,:,ns] = - self.mdp.rmax / (1-self.mdp.gamma)
            if a in self.constraints[cmdp.ACTION]:
                        self.mdp.R[:,a,:] = - self.mdp.rmax / (1-self.mdp.gamma)

    # Generate a baseline policy
    @staticmethod
    def gen_baseline(cmdp, targ_performance, dec=.05):

        mdp = cmdp.mdp

        opt_pol, q = utils.q_improvement(mdp)
        v_opt = np.sum(opt_pol * q, axis=-1)
        v_rand = utils.v_eval(mdp,mdp.rand_pol())

        pi_b = np.copy(opt_pol)
        pib_perf = utils.compute_performance(mdp,pi_b, v_opt, v_rand)

        # Iteratively remove action weight from the best action
        while pib_perf > targ_performance:

            actions = np.arange(mdp.A)
            flag = True
            
            i = 0
            while flag:

                s = np.random.choice(np.arange(mdp.S), 1)[0]
                i += 1

                # Find the best action in that state
                best_a = np.argmax(q[s,:])

                if pi_b[s,best_a] - dec >= 0:
                    flag = False
                else:
                    break

                if i > mdp.S:
                    print('Cannot find baseline')
                    return None

                pi_b[s,best_a] -= dec

                a = np.random.choice(actions[actions != best_a],1)[0]
                pi_b[s,a] += dec

            pib_perf = utils.compute_performance(mdp, pi_b, v_opt, v_rand)
        
        return pi_b
            
    # Generate data
    # Build a dataset of trajectories
    @staticmethod
    def build_D(cmdp, policy, num_traj):

        trajectories = []

        for _ in range(num_traj):
            sars, g = cmdp.mdp.sample_trajectory(policy)

            trajectories.append(sars)

        return trajectories


class utils:
    @staticmethod
    def parse_args():
        
        parser = argparse.ArgumentParser()

        # Envs 
        # ['highway', 'agaid', 'grid', 'grid_lava']
        parser.add_argument("--env_type", type=str, default="grid_lava", help="grid_lava/grid/highway/agaid")

        # Env Types 
        #    [['highway', 'highway-fast', 'merge', 'roundabout'],
        #    ['intvn28_act4_prec1'],
        #    ['5x5','7x8'],
        #    ['5x5_lava', '7x8_lava]]
        parser.add_argument("--env", type=str, default="5x5_lava")

        # Path to data
        parser.add_argument("--path", type=str, default="")

        # Dataset Params
        parser.add_argument("--baseline", type=float, default=.5, help="Baseline Performance")
        parser.add_argument("--d_size", type=int, default=10, help="Init Dataset Size")
        
        # Experiement params
        parser.add_argument("--num_baselines", type=int, default=10, help="Number of Baselines")
        parser.add_argument("--num_datasets", type=int, default=10, help="Number of Datasets") 

        args = parser.parse_args()
        return args
    
    # Perform Value Iteration given an MDP and a policy
    @staticmethod
    def v_eval(mdp, policy, epsilon=.01):

        v = np.zeros(mdp.S)
        
        while True:
            
            old_v = v

            # Perform Bellman Backups depending on if the reward is R(s,a,s') or R(s,a)
            if len(mdp.R.shape) == 3:
                v = np.sum( np.sum( mdp.P * (mdp.R + mdp.gamma * v), axis=-1) * policy, axis=-1)
                v[mdp.goal_states] = 0
            else:
                v = np.sum( np.sum( mdp.P * (mdp.R[:,:,np.newaxis] + mdp.gamma * v), axis=-1) * policy, axis=-1)
                v[mdp.goal_states] = 0

            if np.max(np.abs(old_v-v)) < epsilon:
                return v


    # Perform Q Improvement given an MDP
    @staticmethod 
    def q_improvement(mdp, epsilon=.01):

        q = np.zeros((mdp.S,mdp.A))
        
        while True:

            old_q = q

            # Perform Bellman Backups depending on if the reward is R(s,a,s') or R(s,a)
            if len(mdp.R.shape) == 3:
                q = np.sum(mdp.P * (mdp.R + mdp.gamma * np.max(q, axis=-1)), axis=-1)
                q[mdp.goal_states,:] = 0
            else:
                q = np.sum(mdp.P * (mdp.R[:,:,np.newaxis] + mdp.gamma * np.max(q, axis=-1)), axis=-1)
                q[mdp.goal_states,:] = 0

            if np.max(np.abs(old_q-q)) < epsilon:
                break
                
        # Determine the best actions with bellman backups 
        if len(mdp.R.shape) == 3:
            action_reward = np.sum(mdp.P * (mdp.R + mdp.gamma * np.max(q, axis=-1)), axis=-1)
        else:
            action_reward = np.sum(mdp.P * (mdp.R[:,:,np.newaxis] + mdp.gamma * np.max(q, axis=-1)), axis=-1)

        # Compute the optimal policy
        best_actions = (action_reward - np.max(action_reward, axis=1)[:,np.newaxis]) == 0
        policy = best_actions * (1 / np.sum( best_actions, axis=1)[:,np.newaxis])

        return policy, q
    

    # Compute the normalized performance 
    # Where optimal policy = 1 and uniform random policy = 0
    @staticmethod
    def compute_performance(mdp, policy, v_opt, v_rand):
        return (utils.v_eval(mdp,policy)[mdp.init_state] - v_rand[mdp.init_state]) / \
                (v_opt[mdp.init_state] - v_rand[mdp.init_state])
    
def main():

    np.set_printoptions(suppress=True)
    args = utils.parse_args()

    env_type = args.env_type
    env = args.env

    path = args.path
    #path = '/nfs/hpc/share/soloww/spi_hpc/'

    baseline = args.baseline
    d_size = args.d_size

    num_baselines = args.num_baselines
    num_d = args.num_datasets

    # Load MDP configuration
    P = np.load(f'{path}configs/{env_type}/{env}_P.npy')
    R = np.load(f'{path}configs/{env_type}/{env}_R.npy')
    init_state = np.load(f'{path}configs/{env_type}/{env}_init.npy')
    goal_states = np.load(f'{path}configs/{env_type}/{env}_goal.npy')
    state_space = np.load(f'{path}configs/{env_type}/{env}_state_space.npy')
    action_space = np.load(f'{path}configs/{env_type}/{env}_action_space.npy')
    loaded_mdp = mdp(P, R, init_state, goal_states, state_space=state_space, action_space=action_space)

    # Build the constraints
    constraints = [np.array([0,1]), np.zeros(loaded_mdp.S), np.zeros(loaded_mdp.A) ]

    constrained_mdp = cmdp(loaded_mdp, constraints)

    b = cmdp.gen_baseline(constrained_mdp, 1)

    opt_pol, q = utils.q_improvement(constrained_mdp.mdp)

    D = cmdp.build_D(constrained_mdp, opt_pol, d_size)

    Path(f'{path}config_data/{env_type}/{env}/b{baseline}/').mkdir(parents=True,exist_ok=True)  
    np.save(f'{path}config_data/{env_type}/{env}/b{baseline}/ds{d_size}.npy', np.array(D,dtype=object))


if __name__ == '__main__':
    main()
