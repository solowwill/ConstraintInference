# Main class for running ICRL agent experiments

import argparse
from pathlib import Path
import numpy as np
import agents
import sys
import copy
import matplotlib.pyplot as plt

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

        while np.all(self.curr_state != self.goal_states):

            curr_action = np.random.choice(np.arange(self.A), 1, p=policy[self.curr_state,:])[0]

            next_state = np.random.choice(np.arange(self.S), 1, p=self.P[self.curr_state,curr_action,:])[0]

            if len(self.R.shape) == 3:
                reward = self.R[self.curr_state, curr_action, next_state]
            else: 
                reward = self.R[self.curr_state, curr_action]

            sars.append( [i, self.curr_state, curr_action, reward, next_state] )


            g += (self.gamma**i * reward)
            i += 1

            if len(sars) >= self.H:
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

        #self.build_constrained_reward()
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

    num_feats = 2
    iterations = [100, 500, 1000, 2500, 5000]
    dsizes = [10, 25, 50, 75, 100]

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

    # Build the true constraints
    true_constraints = np.concatenate((constraints[0] != 0, constraints[1] != 0, constraints[2] != 0))
    true_constraints[np.array(constrained_mdp.constraints[0])+num_feats] = True

    data = np.empty(shape=(len(dsizes), len(iterations), num_d, 5))

    for k in range(len(dsizes)):
       
        for j in range(len(iterations)):
            print(f'\n\n{dsizes[k]}, {iterations[j]}')
            for i in range(num_d):
                print(f'Iteration {i}')
                D = np.load(f'{path}config_data/{env_type}/{env}/b{baseline}/ds{dsizes[k]}_{i}.npy', allow_pickle=True)

                exp_mdp = copy.deepcopy(constrained_mdp)
                bicrl_agent = agents.BICRL_Agent(exp_mdp, D, num_samples=iterations[j])

                c_hat = bicrl_agent.icrl()
                learned_constraints = c_hat

                # Rates of accuracy
                tp = np.sum((learned_constraints != 0) * (true_constraints != 0))
                tn = np.sum((learned_constraints == 0) * (true_constraints == 0))
                fp = np.sum((learned_constraints != 0) * (true_constraints == 0))
                fn = np.sum((learned_constraints == 0) * true_constraints != 0)
                prec = tp / (tp + fp)
                recall = tp / (tp + fn)
                acc = (tp + tn) / (tp + tn + fp + fn)
                fpr = fp / (fp + tn)
                fnr = fn / (fn + tp)

                data[k, j, i, 0] = prec
                data[k, j, i, 1] = recall
                data[k, j, i, 2] = acc
                data[k, j, i, 3] = fpr
                data[k, j, i, 4] = fnr

    Path(f'data/{env_type}/{env}').mkdir(exist_ok=True,parents=True)
    np.save(f'data/{env_type}/{env}/bayesian_b{baseline}_data.npy', data)


    data_mean = np.mean(data,axis=-2)
    data_std = np.std(data,axis=-2)

    fig,ax = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=False, figsize=(8,10))
    fig.add_subplot(111, frameon=False)
    
    for i in range(len(iterations)):
        ax[0,0].plot(dsizes, data_mean[:,i,0], label=f'DKL={iterations[i]}')
        ax[0,0].fill_between(dsizes, data_std[:,i,0], alpha=.5)
        ax[0,0].set_ylabel('Precision')

        ax[0,1].plot(dsizes, data_mean[:,i,2], label=f'DKL={iterations[i]}')
        ax[0,1].fill_between(dsizes, data_std[:,i,2], alpha=.5)
        ax[0,0].set_ylabel('Accuracy')

        ax[1,0].plot(dsizes, data_mean[:,i,3], label=f'DKL={iterations[i]}')
        ax[1,0].fill_between(dsizes, data_std[:,i,3], alpha=.5)
        ax[0,0].set_ylabel('False Positive Rate')

        ax[1,1].plot(dsizes, data_mean[:,i,4], label=f'DKL={iterations[i]}')
        ax[1,1].fill_between(dsizes, data_std[:,i,4], alpha=.5)
        ax[0,0].set_ylabel('False Negative Rate')

    plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
    plt.xlabel("Number of Demonstrations")
    plt.title('Bayesian ICRL')

    plt.savefig(f'data/{env_type}/{env}/bayesian_fig.png')
    plt.show()


    





    '''bicrl_agent = agents.BICRL_Agent(constrained_mdp,D)
    bicrl_agent.run_mcmc_bern_constraint(np.zeros(constrained_mdp.k+loaded_mdp.S))

    cnstr = bicrl_agent.get_mean_solution()

    print(np.mean(cnstr,axis=0))'''

   
if __name__ == "__main__":
    main()