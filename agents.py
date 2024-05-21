# Main class for ICRL agents
import numpy as np
import copy
'''import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim'''

# Implements MaxEnt ICRL
class MAXEntICRL_Agent:

    def __init__(self, cmdp, D, dkl=.1):
        self.cmdp = cmdp
        self.D = D

        self.d_kl = dkl


    def icrl(self):

        # Store the computed constraints from the hypothesis space
        C_hat = []

        curr_P = self.cmdp.mdp.P

        p_D = self.state_visitation_data()

        # Go through up to every possible minimal constraint of features (k)
        # States S, and actions A
        for i in range(self.cmdp.k+self.cmdp.mdp.S+self.cmdp.mdp.A):

            # Compute the next most feasible constraints
            features, p_MC = self.feature_accrual(curr_P)
            ci = self.find_best_constraint(features[-1], C_hat)
            if ci < 0:
                #print('No valid constraints to add')
                break
            # If feature, append all constrained states
            elif ci < self.cmdp.k:
                c_states = np.argwhere(self.cmdp.mdp.state_space[:,ci] >= self.cmdp.feat_constraints[ci]).flatten()
                [C_hat.append(c+self.cmdp.k) for c in c_states]
            

            # Compute the new dynamcis
            new_P = self.update_dynamics(curr_P, ci)

            _, new_p_MC = self.feature_accrual(new_P)

            # Compute the change of the KL divergence between the previous set and
            # adding the new constraint
            change_dkl = np.abs(self.compute_KL(p_D, p_MC) - self.compute_KL(p_D, new_p_MC))
            #print(f'Change DKL: {change_dkl}')

            # If the change in the KL divergence is less than the divergence threshold
            # Exit
            if change_dkl <= self.d_kl:
                break
            
            # Otherwise, we should add this constraint
            C_hat.append(ci)
            curr_P = new_P

            #print(C_hat)

        return np.array(C_hat)
    
    # Find the most likely constraint such that the dataset is feasible
    def find_best_constraint(self, feats, C_hat):
        mdp = self.cmdp.mdp
        sorted_feats = np.argsort(feats)[::-1]

        for f in sorted_feats:
            if f in C_hat:
                continue
            valid = True
            for traj in self.D:
                for sars in traj:
                    # If the sorted arg index is an action
                    if f >= len(feats) - mdp.A:
                        if sars[utils.A] == f - (mdp.S + self.cmdp.k):
                            valid = False
                            break
                    # If the sorted arg index in a state
                    elif f >= len(feats) - (mdp.A+mdp.S):
                        if sars[utils.S] == f - self.cmdp.k:
                            valid = False
                            break
                    # The sorted arg index is a feature
                    else:
                        if mdp.state_space[int(sars[utils.S])][f] >= self.cmdp.feat_constraints[f]:
                            valid = False
                            break
            if valid == True:
                #print(f'Found valid:{f}')
                return f
            
        return -1
    
    # Compute the state visitation frequency
    def state_visitation_data(self,):
        d_sh = np.zeros((self.cmdp.mdp.S,self.cmdp.mdp.H))
        for traj in self.D:
            for sars in traj:
                h = int(sars[utils.H])
                s = int(sars[utils.S])

                d_sh[s,h] +=1

        return np.nan_to_num(d_sh / np.sum(d_sh,axis=0))
    
    # Compute the KL Divergence 
    def compute_KL(self, p, q):
        # Constant to avoid dividing by 0
        epsilon = .0000001

        # Add to distributions p and q to avoid divide by 0 errors
        P = p + epsilon
        Q = q + epsilon 

        # Compute the KL Divergence as the log of the quotient times P
        return np.sum(P * np.log( P / Q ))
        
    # Compute the i-th constraint from M_C_hat, constraint hypothesis space, and dataset D
    def feature_accrual(self, P):

        mdp = self.cmdp.mdp
        n_phi = self.cmdp.k + mdp.S + mdp.A

        # Define the feature tracker array as a Hx(n+S+A) array
        phi_hat = np.zeros((mdp.H,n_phi))
        phi_hat_s = np.zeros((mdp.S, mdp.H, n_phi))
        phi_hat_as = np.zeros((mdp.A,mdp.S,mdp.H,n_phi))
        
        # Define the state visitation counts
        D_s = np.zeros((mdp.S,mdp.H))

        pi_stat = self.local_action_probabilities()
        pi = np.stack([pi_stat for _ in range(mdp.H)])

        for s in range(mdp.S):
            D_s[s,0] = self.cmdp.mu0[s]
        for h in range(mdp.H-1):
            for s in range(mdp.S):
                for a in range(mdp.A):
                    # Compute new feature accruals
                    phi_hat_as[a,s,h] = self.phi_ind(s,a) * (D_s[s,h] * np.ones(n_phi) - phi_hat_s[s,h])

            # Compute for the next possible stateand timestep
            for s_p in range(mdp.S):
                for s in range(mdp.S):
                    for a in range(mdp.A):
                        # Compute state visitation counts for the next timestep
                        D_s[s_p,h+1] += D_s[s,h] * pi[h,s,a] * P[s,a,s_p]

                        # Compute the phi state feature accrual for the next timestep
                        phi_hat_s[s_p,h+1] += (phi_hat_s[s,h] + phi_hat_as[a,s,h]) * pi[h,s,a] * P[s,a,s_p]

            # Sum across all states to get feature accruals for that timestep
            phi_hat[h+1] = np.sum(phi_hat_s[:,h+1], axis=0)

        return phi_hat, D_s
    

    # Compute local action probabilities for edge frequency calculation
    def local_action_probabilities(self):
        mdp = self.cmdp.mdp

        if len(mdp.R.shape) == 3:
            R = np.sum(mdp.P*mdp.R,axis=(0,1)) / mdp.A
        else:
            R = np.sum(mdp.P * mdp.R[:,:,np.newaxis],axis=(0,1)) / mdp.A
        er = np.exp(R)
        p = [np.array(mdp.P[:, a, :]) for a in range(mdp.A)]

        # Initialize at terminal states
        zs = np.zeros(mdp.S)
        zs[mdp.goal_states] = 1.0

        # Perform backward pass
        for _ in range(2 * mdp.S):
            za = np.array([er * p[a].dot(zs) for a in range(mdp.A)]).T
            zs = za.sum(axis=1)

        # Compute local action probabilities
        return np.nan_to_num(za / zs[:, None], nan=0)
    
    
    # Take the current dynamics and a constraint 
    def update_dynamics(self, P, ci):
        mdp = self.cmdp.mdp

        # If the constraint is an action
        if ci >= (self.cmdp.k + mdp.S):
            # Get the index of the action
            a = ci - (mdp.S + self.cmdp.k)
            P[:,a,:] = np.zeros((mdp.S, mdp.S))

        # If the constraint is a state
        elif ci >= self.cmdp.k:
            s = ci - self.cmdp.k
            P[:,:,s] = np.zeros((mdp.S, mdp.A))
            P[s,:,:] = np.zeros((mdp.A, mdp.S))

        # The constraint is a feature
        else:
            c_states = np.argwhere(mdp.state_space[:,ci] >= self.cmdp.feat_constraints[ci]).flatten()
            for s in c_states:
                P[:,:,s] = np.zeros((mdp.S, mdp.A))
                P[s,:,:] = np.zeros((mdp.A, mdp.S))

        return P


    # Build the feature indicators as 1 if a feature, state, or action has been seen
    # by the i-th timestep, 0 otherwise
    # Will return a k+|S|+|A| array 
    def phi_ind(self, s, a):
        features = np.zeros(self.cmdp.k + self.cmdp.mdp.S + self.cmdp.mdp.A)

        # Check for feature constraints
        for f in range(self.cmdp.k):
            if self.cmdp.mdp.state_space[s,f] >= self.cmdp.feat_constraints[f]:
                features[f] = 1

        # Then set the indicator features for the state and the action
        features[self.cmdp.k+s] = 1
        features[self.cmdp.k+self.cmdp.mdp.S+a] = 1
        return features
    

    # Implements MaxEnt ICRL

class MAXCasualEntICRL_Agent:

    def __init__(self, cmdp, D, dkl=.1):
        self.cmdp = cmdp
        self.D = D

        self.d_kl = dkl


    def icrl(self):

        # Store the computed constraints from the hypothesis space
        C_hat = []

        curr_P = self.cmdp.mdp.P

        p_D = self.state_visitation_data()

        # Go through up to every possible minimal constraint of features (k)
        # States S, and actions A
        for i in range(self.cmdp.k+self.cmdp.mdp.S+self.cmdp.mdp.A):

            # Compute the next most feasible constraints
            features, p_MC = self.feature_accrual(curr_P)
            ci = self.find_best_constraint(features[-1], C_hat)
            if ci < 0:
                #print('No valid constraints to add')
                break
            # If feature, append all constrained states
            elif ci < self.cmdp.k:
                c_states = np.argwhere(self.cmdp.mdp.state_space[:,ci] >= self.cmdp.feat_constraints[ci]).flatten()
                [C_hat.append(c+self.cmdp.k) for c in c_states]
            

            # Compute the new dynamcis
            new_P = self.update_dynamics(curr_P, ci)

            _, new_p_MC = self.feature_accrual(new_P)

            # Compute the change of the KL divergence between the previous set and
            # adding the new constraint
            change_dkl = np.abs(self.compute_KL(p_D, p_MC) - self.compute_KL(p_D, new_p_MC))
            #print(f'Change DKL: {change_dkl}')

            # If the change in the KL divergence is less than the divergence threshold
            # Exit
            if change_dkl <= self.d_kl:
                break
            
            # Otherwise, we should add this constraint
            C_hat.append(ci)
            curr_P = new_P

            #print(C_hat)

        return np.array(C_hat)
    
    # Find the most likely constraint such that the dataset is feasible
    def find_best_constraint(self, feats, C_hat):
        mdp = self.cmdp.mdp
        sorted_feats = np.argsort(feats)[::-1]

        for f in sorted_feats:
            if f in C_hat:
                continue
            valid = True
            for traj in self.D:
                for sars in traj:
                    # If the sorted arg index is an action
                    if f >= len(feats) - mdp.A:
                        if sars[utils.A] == f - (mdp.S + self.cmdp.k):
                            valid = False
                            break
                    # If the sorted arg index in a state
                    elif f >= len(feats) - (mdp.A+mdp.S):
                        if sars[utils.S] == f - self.cmdp.k:
                            valid = False
                            break
                    # The sorted arg index is a feature
                    else:
                        if mdp.state_space[int(sars[utils.S])][f] >= self.cmdp.feat_constraints[f]:
                            valid = False
                            break
            if valid == True:
                #print(f'Found valid:{f}')
                return f
            
        return -1
    
    # Compute the state visitation frequency
    def state_visitation_data(self,):
        d_sh = np.zeros((self.cmdp.mdp.S,self.cmdp.mdp.H))
        for traj in self.D:
            for sars in traj:
                h = int(sars[utils.H])
                s = int(sars[utils.S])

                d_sh[s,h] +=1

        return np.nan_to_num(d_sh / np.sum(d_sh,axis=0))
    
    # Compute the KL Divergence 
    def compute_KL(self, p, q):
        # Constant to avoid dividing by 0
        epsilon = .0000001

        # Add to distributions p and q to avoid divide by 0 errors
        P = p + epsilon
        Q = q + epsilon 

        # Compute the KL Divergence as the log of the quotient times P
        return np.sum(P * np.log( P / Q ))
        
    # Compute the i-th constraint from M_C_hat, constraint hypothesis space, and dataset D
    def feature_accrual(self, P):

        mdp = self.cmdp.mdp
        n_phi = self.cmdp.k + mdp.S + mdp.A

        
    

    # Compute local action probabilities for edge frequency calculation
    def local_action_probabilities(self):
        mdp = self.cmdp.mdp

        if len(mdp.R.shape) == 3:
            R = np.sum(mdp.P*mdp.R,axis=(0,1)) / mdp.A
        else:
            R = np.sum(mdp.P * mdp.R[:,:,np.newaxis],axis=(0,1)) / mdp.A
        er = np.exp(R)
        p = [np.array(mdp.P[:, a, :]) for a in range(mdp.A)]

        # Initialize at terminal states
        zs = np.zeros(mdp.S)
        zs[mdp.goal_states] = 1.0

        # Perform backward pass
        for _ in range(2 * mdp.S):
            za = np.array([er * p[a].dot(zs) for a in range(mdp.A)]).T
            zs = za.sum(axis=1)

        # Compute local action probabilities
        return np.nan_to_num(za / zs[:, None], nan=0)
    
    
    # Take the current dynamics and a constraint 
    def update_dynamics(self, P, ci):
        mdp = self.cmdp.mdp

        # If the constraint is an action
        if ci >= (self.cmdp.k + mdp.S):
            # Get the index of the action
            a = ci - (mdp.S + self.cmdp.k)
            P[:,a,:] = np.zeros((mdp.S, mdp.S))

        # If the constraint is a state
        elif ci >= self.cmdp.k:
            s = ci - self.cmdp.k
            P[:,:,s] = np.zeros((mdp.S, mdp.A))
            P[s,:,:] = np.zeros((mdp.A, mdp.S))

        # The constraint is a feature
        else:
            c_states = np.argwhere(mdp.state_space[:,ci] >= self.cmdp.feat_constraints[ci]).flatten()
            for s in c_states:
                P[:,:,s] = np.zeros((mdp.S, mdp.A))
                P[s,:,:] = np.zeros((mdp.A, mdp.S))

        return P


    # Build the feature indicators as 1 if a feature, state, or action has been seen
    # by the i-th timestep, 0 otherwise
    # Will return a k+|S|+|A| array 
    def phi_ind(self, s, a):
        features = np.zeros(self.cmdp.k + self.cmdp.mdp.S + self.cmdp.mdp.A)

        # Check for feature constraints
        for f in range(self.cmdp.k):
            if self.cmdp.mdp.state_space[s,f] >= self.cmdp.feat_constraints[f]:
                features[f] = 1

        # Then set the indicator features for the state and the action
        features[self.cmdp.k+s] = 1
        features[self.cmdp.k+self.cmdp.mdp.S+a] = 1
        return features
    
# Implement Bayesian ICRL
class BICRL_Agent:
    def __init__(self, cmdp, D, num_samples=4000, beta=1, num_cnstr=0, epsilon=0.001):

        self.cmdp = cmdp
        self.D = D
        self.beta = beta
        self.epsilon = epsilon
        self.num_samples = num_samples
    
        self.num_cnstr = num_cnstr
        
        self.num_mcmc_dims = self.cmdp.k + self.cmdp.mdp.S + self.cmdp.mdp.A


    # Calculate the log likelihood of a reward function given constraints
    def calc_log_likelihood(self, curr_rew):
        mdp = self.cmdp.mdp

        # Hypotheical reward analysis    
        _, q = utils.approx_q_improvement(mdp,mdp.P,curr_rew)

        # Calculate the log likelihood of the reward hypothesis given the demonstrations
        # Assumpe uniformative prior
        log_prior = 0.0  
        log_sum = log_prior
        for traj in self.D:
            for sarcs in traj:
                # Do not consider counterfactuals in goal state
                if (sarcs[utils.S] not in mdp.goal_states):  
                    Z_exponents = self.beta * q[int(sarcs[utils.S])]
                    log_sum += self.beta * q[int(sarcs[utils.S])][int(sarcs[utils.A])] - utils.logsumexp(Z_exponents)         
        return log_sum

    # Generate a proposed feature map and reward
    def gen_reward_proposal(self, W_old, rew_old, ind, stdev=0.1):
        rew_new = copy.deepcopy(rew_old)
        W_new = copy.deepcopy(W_old)

        if ind % 20 == 0:  
            rew_new = rew_new + stdev * np.random.randn() 
        else:
            index = np.random.randint(len(W_old))
            W_new[index] = 1 if W_old[index] == 0 else 0
        
        return W_new, rew_new
                

    # Initialize problem solution for MCMC to all zeros
    def gen_init_proposal(self):
        
        reward_penalty = np.random.randint(-self.cmdp.mdp.rmax / (1-self.cmdp.mdp.gamma), -5)
        curr_constraints = np.zeros(self.num_mcmc_dims)
        return curr_constraints, reward_penalty

    # Get the current reward from the feature vector and reward penalty
    def get_curr_reward(self, curr_constraints, reward_penalty):
        mdp = self.cmdp.mdp
        constraint_inds = np.argwhere(curr_constraints == 1)

        R = copy.deepcopy(mdp.R)

        if len(mdp.R.shape) == 3:
            for c in constraint_inds:
                # Action constraint
                if c >= (self.cmdp.k + mdp.S):
                    c -= (self.cmdp.k + mdp.S)
                    R[:,c,:] = reward_penalty
                # State constraint
                elif c >= self.cmdp.k:
                    c -= self.cmdp.k
                    R[:,:,c] = reward_penalty
                # Feature constraint
                else:
                    constrained_states = np.argwhere(mdp.state_space[:,c] >= self.cmdp.feat_constraints[c]).flatten()
                    R[:,:,constrained_states] = reward_penalty
        else:
            for c in constraint_inds:
                # Action constraint
                if c >= (self.cmdp.k + mdp.S):
                    c -=(self.cmdp.k + mdp.S)
                    R[:,c] = reward_penalty
                # State constraint
                elif c >= self.cmdp.k:
                    c -= self.cmdp.k
                    for i in range(mdp.R.shape[0]):
                        for j in range(mdp.R.shape[1]):
                            next_states = np.argwhere(mdp.P[i,j] != 0)
                            for ns in next_states:
                                if ns == c:
                                    R[i,j] = reward_penalty
                # Feature constraint
                else:
                    constrained_states = np.argwhere(mdp.state_space[:,c] >= self.cmdp.feat_constraints[c]).flatten()
                    for i in range(mdp.R.shape[0]):
                        for j in range(mdp.R.shape[1]):
                            next_states = np.argwhere(mdp.P[i,j] != 0)
                            for ns in next_states:
                                if ns in constrained_states:
                                    R[i,j] = reward_penalty                           
        return R
                                

    # Run MCMC with Gaussian symmetric proposal and uniform prior
    def icrl(self, stdev=.1):
        accept_cnt = 0 
    
        # Store rewards found for BICRL 
        self.chain_cnstr = np.zeros((self.num_samples, self.num_mcmc_dims)) 
        self.chain_rew = np.zeros((self.num_samples, self.num_mcmc_dims)) 
        curr_constraints, reward_penalty = self.gen_init_proposal()

        # Get current rewards, function of state
        curr_rewards = self.get_curr_reward(curr_constraints, reward_penalty)

        curr_ll = self.calc_log_likelihood(curr_rewards) 

        #keep track of MAP loglikelihood and solution
        map_ll = curr_ll  
        map_constr = curr_rewards
        map_list = []
        perf_list = []
      
        for i in range(self.num_samples):
            print(i)
            # Sample from proposed distribution        
            proposed_constraints, proposed_reward_penalty = self.gen_reward_proposal(curr_constraints, reward_penalty, i, stdev)

            proposed_reward = self.get_curr_reward(proposed_constraints, proposed_reward_penalty)
            
            # Calculate proposed log likelihood ratio test
            prop_ll = self.calc_log_likelihood(proposed_reward)
           
           # If proposed log likelihood greater than current log likelihood
           # Accept the constraint
            if prop_ll > curr_ll:
                self.chain_cnstr[i,:] = proposed_constraints
                self.chain_rew[i,:] = proposed_reward_penalty
                accept_cnt += 1
                curr_reward = proposed_reward_penalty
                curr_constraints = proposed_constraints
                curr_ll = prop_ll

                # If proposed log likelihood greater than map ll, accept
                # as MAP constraint
                if prop_ll > map_ll: 
                    map_constr = proposed_constraints
                    map_rew = proposed_reward_penalty
            else:
                # Accept as current constraint with probability 
                # exp(prop_ll - curr_ll)
                if np.random.rand() < np.exp(prop_ll - curr_ll):
                    self.chain_cnstr[i,:] = proposed_constraints
                    self.chain_rew[i,:] = proposed_reward_penalty
                    accept_cnt += 1
                    curr_reward = proposed_reward_penalty
                    curr_ll = prop_ll
                    curr_constraints = proposed_constraints
                else:
                    self.chain_cnstr[i,:] = curr_constraints               
                    self.chain_rew[i,:] = curr_reward   
     
        #print("accept rate:", accept_cnt / num_samples)
        self.accept_rate = accept_cnt / self.num_samples
        self.map_rew = map_rew
        self.map_constr = map_constr
        self.map_list = map_list
        self.perf_list = perf_list
        #print(self.map_constr)

        return curr_constraints
      
         
    # Get the MaP solution
    def get_map_solution(self):
        return self.map_constr, self.map_rew


    # Get the average solution over the posterior
    def get_mean_solution(self, burn_frac=0.1, skip_rate=1):
        return self.chain_cnstr[int(burn_frac*len(self.chain_cnstr))::skip_rate]

# Implement Bayesian ICRL
class FeatureBICRL_Agent:
    def __init__(self, cmdp, D, num_samples=4000, beta=1, num_cnstr=0, epsilon=0.001):

        self.cmdp = cmdp
        self.D = D
        self.beta = beta
        self.epsilon = epsilon
        self.num_samples = num_samples
    
        self.num_cnstr = num_cnstr
        
        self.num_mcmc_dims = self.cmdp.k + self.cmdp.mdp.A


    # Calculate the log likelihood of a reward function given constraints
    def calc_log_likelihood(self, curr_rew):
        mdp = self.cmdp.mdp

        # Hypotheical reward analysis    
        _, q = utils.approx_q_improvement(mdp,mdp.P,curr_rew)

        # Calculate the log likelihood of the reward hypothesis given the demonstrations
        # Assumpe uniformative prior
        log_prior = 0.0  
        log_sum = log_prior
        for traj in self.D:
            for sarcs in traj:
                # Do not consider counterfactuals in goal state
                if (sarcs[utils.S] not in mdp.goal_states):  
                    Z_exponents = self.beta * q[int(sarcs[utils.S])]
                    log_sum += self.beta * q[int(sarcs[utils.S])][int(sarcs[utils.A])] - utils.logsumexp(Z_exponents)         
        return log_sum

    # Generate a proposed feature map and reward
    def gen_reward_proposal(self, W_old, rew_old, ind, stdev=0.1):
        rew_new = copy.deepcopy(rew_old)
        W_new = copy.deepcopy(W_old)

        if ind % 20 == 0:  
            rew_new = rew_new + stdev * np.random.randn() 
        else:
            index = np.random.randint(len(W_old))
            W_new[index] = 1 if W_old[index] == 0 else 0
        
        return W_new, rew_new
                

    # Initialize problem solution for MCMC to all zeros
    def gen_init_proposal(self):
        
        reward_penalty = np.random.randint(-self.cmdp.mdp.rmax / (1-self.cmdp.mdp.gamma), -5)
        curr_constraints = np.zeros(self.num_mcmc_dims)
        return curr_constraints, reward_penalty

    # Get the current reward from the feature vector and reward penalty
    def get_curr_reward(self, curr_constraints, reward_penalty):
        mdp = self.cmdp.mdp
        constraint_inds = np.argwhere(curr_constraints == 1)

        R = copy.deepcopy(mdp.R)

        if len(mdp.R.shape) == 3:
            for c in constraint_inds:
                # Action constraint
                if c >= (self.cmdp.k ):
                    c -= (self.cmdp.k)
                    R[:,c,:] = reward_penalty
                # Feature constraint
                else:
                    constrained_states = np.argwhere(mdp.state_space[:,c] >= self.cmdp.feat_constraints[c]).flatten()
                    R[:,:,constrained_states] = reward_penalty
        else:
            for c in constraint_inds:
                # Action constraint
                if c >= (self.cmdp.k):
                    c -=(self.cmdp.k)
                    R[:,c] = reward_penalty
                # Feature constraint
                else:
                    constrained_states = np.argwhere(mdp.state_space[:,c] >= self.cmdp.feat_constraints[c]).flatten()
                    for i in range(mdp.R.shape[0]):
                        for j in range(mdp.R.shape[1]):
                            next_states = np.argwhere(mdp.P[i,j] != 0)
                            for ns in next_states:
                                if ns in constrained_states:
                                    R[i,j] = reward_penalty                           
        return R
                                

    # Run MCMC with Gaussian symmetric proposal and uniform prior
    def icrl(self, stdev=.1):
        accept_cnt = 0 
    
        # Store rewards found for BICRL 
        self.chain_cnstr = np.zeros((self.num_samples, self.num_mcmc_dims)) 
        self.chain_rew = np.zeros((self.num_samples, self.num_mcmc_dims)) 
        curr_constraints, reward_penalty = self.gen_init_proposal()

        # Get current rewards, function of state
        curr_rewards = self.get_curr_reward(curr_constraints, reward_penalty)

        curr_ll = self.calc_log_likelihood(curr_rewards) 

        #keep track of MAP loglikelihood and solution
        map_ll = curr_ll  
        map_constr = curr_rewards
        map_list = []
        perf_list = []
      
        for i in range(self.num_samples):
            #print(i)
            # Sample from proposed distribution        
            proposed_constraints, proposed_reward_penalty = self.gen_reward_proposal(curr_constraints, reward_penalty, i, stdev)

            proposed_reward = self.get_curr_reward(proposed_constraints, proposed_reward_penalty)
            
            # Calculate proposed log likelihood ratio test
            prop_ll = self.calc_log_likelihood(proposed_reward)
           
           # If proposed log likelihood greater than current log likelihood
           # Accept the constraint
            if prop_ll > curr_ll:
                self.chain_cnstr[i,:] = proposed_constraints
                self.chain_rew[i,:] = proposed_reward_penalty
                accept_cnt += 1
                curr_reward = proposed_reward_penalty
                curr_constraints = proposed_constraints
                curr_ll = prop_ll

                # If proposed log likelihood greater than map ll, accept
                # as MAP constraint
                if prop_ll > map_ll: 
                    map_constr = proposed_constraints
                    map_rew = proposed_reward_penalty
            else:
                # Accept as current constraint with probability 
                # exp(prop_ll - curr_ll)
                if np.random.rand() < np.exp(prop_ll - curr_ll):
                    self.chain_cnstr[i,:] = proposed_constraints
                    self.chain_rew[i,:] = proposed_reward_penalty
                    accept_cnt += 1
                    curr_reward = proposed_reward_penalty
                    curr_ll = prop_ll
                    curr_constraints = proposed_constraints
                else:
                    self.chain_cnstr[i,:] = curr_constraints               
                    self.chain_rew[i,:] = curr_reward   
     
        #print("accept rate:", accept_cnt / num_samples)
        self.accept_rate = accept_cnt / self.num_samples
        self.map_rew = map_rew
        self.map_constr = map_constr
        self.map_list = map_list
        self.perf_list = perf_list
        #print(self.map_constr)

        return curr_constraints
      
         
    # Get the MaP solution
    def get_map_solution(self):
        return self.map_constr, self.map_rew


    # Get the average solution over the posterior
    def get_mean_solution(self, burn_frac=0.1, skip_rate=1):
        return self.chain_cnstr[int(burn_frac*len(self.chain_cnstr))::skip_rate]


# Implements MaxEnt ICRL
class FeatureMAXEntICRL_Agent:

    def __init__(self, cmdp, D, dkl=.1):
        self.cmdp = cmdp
        self.D = D

        self.d_kl = dkl


    def icrl(self):

        # Store the computed constraints from the hypothesis space
        C_hat = []

        curr_P = self.cmdp.mdp.P

        p_D = self.state_visitation_data()

        # Go through up to every possible minimal constraint of features (k)
        # States S, and actions A
        for i in range(self.cmdp.k+self.cmdp.mdp.S+self.cmdp.mdp.A):

            # Compute the next most feasible constraints
            features, p_MC = self.feature_accrual(curr_P)
            ci = self.find_best_constraint(features[-1], C_hat)
            if ci < 0:
                #print('No valid constraints to add')
                break
            # If feature, append all constrained states
            elif ci < self.cmdp.k:
                c_states = np.argwhere(self.cmdp.mdp.state_space[:,ci] >= self.cmdp.feat_constraints[ci]).flatten()
                [C_hat.append(c+self.cmdp.k) for c in c_states]
            

            # Compute the new dynamcis
            new_P = self.update_dynamics(curr_P, ci)

            _, new_p_MC = self.feature_accrual(new_P)

            # Compute the change of the KL divergence between the previous set and
            # adding the new constraint
            change_dkl = np.abs(self.compute_KL(p_D, p_MC) - self.compute_KL(p_D, new_p_MC))
            #print(f'Change DKL: {change_dkl}')

            # If the change in the KL divergence is less than the divergence threshold
            # Exit
            if change_dkl <= self.d_kl:
                break
            
            # Otherwise, we should add this constraint
            C_hat.append(ci)
            curr_P = new_P

            #print(C_hat)

        return np.array(C_hat)
    
    # Find the most likely constraint such that the dataset is feasible
    def find_best_constraint(self, feats, C_hat):
        mdp = self.cmdp.mdp
        sorted_feats = np.argsort(feats)[::-1]

        for f in sorted_feats:
            if f in C_hat:
                continue
            valid = True
            for traj in self.D:
                for sars in traj:
                    # If the sorted arg index is an action
                    if f >= len(feats) - mdp.A:
                        if sars[utils.A] == f - (mdp.S + self.cmdp.k):
                            valid = False
                            break
                    # If the sorted arg index in a state
                    elif f >= len(feats) - (mdp.A+mdp.S):
                        if sars[utils.S] == f - self.cmdp.k:
                            valid = False
                            break
                    # The sorted arg index is a feature
                    else:
                        if mdp.state_space[int(sars[utils.S])][f] >= self.cmdp.feat_constraints[f]:
                            valid = False
                            break
            if valid == True:
                #print(f'Found valid:{f}')
                return f
            
        return -1
    
    # Compute the state visitation frequency
    def state_visitation_data(self,):
        d_sh = np.zeros((self.cmdp.mdp.S,self.cmdp.mdp.H))
        for traj in self.D:
            for sars in traj:
                h = int(sars[utils.H])
                s = int(sars[utils.S])

                d_sh[s,h] +=1

        return np.nan_to_num(d_sh / np.sum(d_sh,axis=0))
    
    # Compute the KL Divergence 
    def compute_KL(self, p, q):
        # Constant to avoid dividing by 0
        epsilon = .0000001

        # Add to distributions p and q to avoid divide by 0 errors
        P = p + epsilon
        Q = q + epsilon 

        # Compute the KL Divergence as the log of the quotient times P
        return np.sum(P * np.log( P / Q ))
        
    # Compute the i-th constraint from M_C_hat, constraint hypothesis space, and dataset D
    def feature_accrual(self, P):

        mdp = self.cmdp.mdp
        n_phi = self.cmdp.k + mdp.S + mdp.A

        # Define the feature tracker array as a Hx(n+S+A) array
        phi_hat = np.zeros((mdp.H,n_phi))
        phi_hat_s = np.zeros((mdp.S, mdp.H, n_phi))
        phi_hat_as = np.zeros((mdp.A,mdp.S,mdp.H,n_phi))
        
        # Define the state visitation counts
        D_s = np.zeros((mdp.S,mdp.H))

        pi_stat = self.local_action_probabilities()
        pi = np.stack([pi_stat for _ in range(mdp.H)])

        for s in range(mdp.S):
            D_s[s,0] = self.cmdp.mu0[s]
        for h in range(mdp.H-1):
            for s in range(mdp.S):
                for a in range(mdp.A):
                    # Compute new feature accruals
                    phi_hat_as[a,s,h] = self.phi_ind(s,a) * (D_s[s,h] * np.ones(n_phi) - phi_hat_s[s,h])

            # Compute for the next possible stateand timestep
            for s_p in range(mdp.S):
                for s in range(mdp.S):
                    for a in range(mdp.A):
                        # Compute state visitation counts for the next timestep
                        D_s[s_p,h+1] += D_s[s,h] * pi[h,s,a] * P[s,a,s_p]

                        # Compute the phi state feature accrual for the next timestep
                        phi_hat_s[s_p,h+1] += (phi_hat_s[s,h] + phi_hat_as[a,s,h]) * pi[h,s,a] * P[s,a,s_p]

            # Sum across all states to get feature accruals for that timestep
            phi_hat[h+1] = np.sum(phi_hat_s[:,h+1], axis=0)

        return phi_hat, D_s
    

    # Compute local action probabilities for edge frequency calculation
    def local_action_probabilities(self):
        mdp = self.cmdp.mdp

        if len(mdp.R.shape) == 3:
            R = np.sum(mdp.P*mdp.R,axis=(0,1)) / mdp.A
        else:
            R = np.sum(mdp.P * mdp.R[:,:,np.newaxis],axis=(0,1)) / mdp.A
        er = np.exp(R)
        p = [np.array(mdp.P[:, a, :]) for a in range(mdp.A)]

        # Initialize at terminal states
        zs = np.zeros(mdp.S)
        zs[mdp.goal_states] = 1.0

        # Perform backward pass
        for _ in range(2 * mdp.S):
            za = np.array([er * p[a].dot(zs) for a in range(mdp.A)]).T
            zs = za.sum(axis=1)

        # Compute local action probabilities
        return np.nan_to_num(za / zs[:, None], nan=0)
    
    
    # Take the current dynamics and a constraint 
    def update_dynamics(self, P, ci):
        mdp = self.cmdp.mdp

        # If the constraint is an action
        if ci >= (self.cmdp.k + mdp.S):
            # Get the index of the action
            a = ci - (mdp.S + self.cmdp.k)
            P[:,a,:] = np.zeros((mdp.S, mdp.S))

        # If the constraint is a state
        elif ci >= self.cmdp.k:
            s = ci - self.cmdp.k
            P[:,:,s] = np.zeros((mdp.S, mdp.A))
            P[s,:,:] = np.zeros((mdp.A, mdp.S))

        # The constraint is a feature
        else:
            c_states = np.argwhere(mdp.state_space[:,ci] >= self.cmdp.feat_constraints[ci]).flatten()
            for s in c_states:
                P[:,:,s] = np.zeros((mdp.S, mdp.A))
                P[s,:,:] = np.zeros((mdp.A, mdp.S))

        return P


    # Build the feature indicators as 1 if a feature, state, or action has been seen
    # by the i-th timestep, 0 otherwise
    # Will return a k+|S|+|A| array 
    def phi_ind(self, s, a):
        features = np.zeros(self.cmdp.k + self.cmdp.mdp.S + self.cmdp.mdp.A)

        # Check for feature constraints
        for f in range(self.cmdp.k):
            if self.cmdp.mdp.state_space[s,f] >= self.cmdp.feat_constraints[f]:
                features[f] = 1

        # Then set the indicator features for the state and the action
        features[self.cmdp.k+s] = 1
        features[self.cmdp.k+self.cmdp.mdp.S+a] = 1
        return features
    

    # Implements MaxEnt ICRL

# Implement MB-CLPF
# Inferring constraints with proactive human feedback
'''class MBCLPF_Agent:

    def __init__(self, cmdp, k=100, h=5, w=[0.5,0.0,0.5], beta=.25, epsilon=0.05, mu=.5):
        self.cmdp = cmdp

        self.h = h
        self.beta = beta
        self.epsilon = epsilon
        self.k = k
        self.mu = mu
        self.w = w

        # Initialize neural nets
        # Constraint Net
        # TODO: Add loss functions
        self.constraint_net = nn.NeuralNet(self.cmdp.mdp.state_space.shape[0]+self.cmdp.mdp.action_space.shape[0])
        self.train_constraint = nn.TrainNet(self.constraint_net)

        self.intervention_net = nn.NeuralNet(self.cmdp.mdp.state_space.shape[0]+self.cmdp.mdp.action_space.shape[0])
        self.train_intervention = nn.TrainNet(self.intervention_net)

    # Run the Main MB CLPF loop
    def run_mbclpf(self, horizon=10000):
        mdp = self.cmdp.mdp

        # Compute policy, dataset
        pi, _ = utils.approx_q_improvement(self.mdp, self.mdp.P, self.mdp.R)

        D = np.array(shape=(0,2+mdp.S*mdp.A))
        for i in range(horizon):

            # Collect intervention feedback
            # In the form of sparse reward <state, {0,1}>
            for _ in range(self.k):
                sars, feedback, _ = self.sample_trajectory_with_intervention(pi)
                # Add feedback and policy to dataset
                D = np.concatenate((D,feedback),axis=0)

            # Train intervention and constraint model
            self.train_constraint.train(D[:,[0,1,3]],D[:,2])
            self.train_intervention.train(D[:,[0,1,3]],D[:,2])

            # Get the training reward
            R_T = self.compute_training_reward()

            # Compute a new policy based on the new training reward
            pi, _ = utils.approx_q_improvement(self.mdp, self.mdp.P, R_T)

            # if the termination conditon is met, break
            if i > horizon:
                break

    # Compute the shannon entropy given a set of probabilities of constraint pairs
    def compute_entropy(self, x):
        return -np.sum(x * np.log(x))
    
    # Compute the reward for training
    def compute_training_reward(self):
        # Build the consraint probabilities and reward
        self.build_constraint_from_network()
        E = self.compute_entropy(self.constraints)

        # Build intervention reward
        self.build_internvention_from_network()
        I = -(self.interventions > self.mu)

        # Compute training reward
        R_T = np.stack((self.cmdp.mdp.R, I, E),axis=-1)
        return np.sum(self.w * R_T,axis=0)
    
    # Build the constraint approximate function
    def build_constraint_from_network(self):
        mdp = self.cmdp.mdp
        self.constraints = np.zeros((mdp.S, mdp.A))

        for s in range(mdp.S):
            for a in range(mdp.A):
                features = np.concatenate((mdp.state_space[s], mdp.action_space[s]))
                # TODO: Fix this line
                self.constraints[s,a] = self.constraint_model.forward(features)

        return self.constraints
    
    # Build the intervention approximation function
    def build_internvention_from_network(self):
        mdp = self.cmdp.mdp
        self.interventions = np.zeros((mdp.S, mdp.A))

        for s in range(mdp.S):
            for a in range(mdp.A):
                features = np.concatenate((mdp.state_space[s], mdp.action_space[s]))
                self.interventions[s,a] = self.intervention_net.forward(features)

        return self.interventions

    # Compute the expected constraint violation 
    def compute_expected_constraint_violation(self, policy, constraints):

        # Perform expected constraint violation
        c = utils.horizon_approx_v_eval(self.mdp, self.mdp.P, constraints, policy, h=self.h)

        return c
    
    # Compute the corrupted human policy
    def compute_human_policy(self, policy):
        mdp = self.cmdp.mdp
        pi_hat = np.zeros((mdp.S, mdp.A))
        for s in range(mdp.S):
            a = np.argwhere(policy[s] != 0).flatten()
            pi_hat[s,a] = 1-self.epsilon

            new_a = np.random.choice(np.delete(np.arange(mdp.S), a, None))
            pi_hat[s,new_a] = self.epsilon

        self.corrupt_pi = pi_hat
        return pi_hat
    
    # Compute the human intervention function
    def compute_human_intervention(self, policy):

        pi_hat = self.compute_human_policy(policy)

        c = self.compute_expected_constraint_violation(pi_hat, self.cmdp.C)

        return c

    # Sample a trajectory given a policy
    def sample_trajectory_with_intervention(self, policy):

        mdp = self.cmdp.mdp

        human_intvn_criteria = self.compute_human_intervention(policy)

        sars = []
        feedback = [[mdp.init_state, 0]]

        mdp.curr_state = mdp.init_state

        g = 0
        i = 0

        while np.all(mdp.curr_state != mdp.goal_states):

            curr_action = np.random.choice(np.arange(mdp.A), 1, p=policy[mdp.curr_state,:])[0]

            next_state = np.random.choice(np.arange(mdp.S), 1, p=mdp.P[mdp.curr_state,curr_action,:])[0]

            if len(mdp.R.shape) == 3:
                reward = mdp.R[mdp.curr_state, curr_action, next_state]
            else: 
                reward = mdp.R[mdp.curr_state, curr_action]

            # Move to a new state if high likelihood of constraint violation
            if human_intvn_criteria[mdp.curr_state] > self.beta:
                next_state = np.random.choice(human_intvn_criteria < self.beta)
                feedback.append([mdp.state_space[mdp.curr_state],mdp.action_space[curr_action],1,np.flatten(policy)])
            else:
                feedback.append([mdp.state_space[mdp.curr_state],mdp.action_space[curr_action],0,np.flatten(policy)])

            sars.append( [i, mdp.curr_state, curr_action, reward, next_state] )

            g += (mdp.gamma**i * reward)
            i += 1

            if len(sars) >= mdp.H:
                break

            mdp.curr_state = next_state
            
        return np.array(sars), np.array(feedback), g
    
    # Compute J1 and J0
    def compute_constraint_violation_loss(self, D):
        j1 = D[np.argwhere(D[:,2] == 1)]
        j0 = D[np.argwhere(D[:,2] == 0)]

        j1_cv = []
        j0_cv = []

        # Build the constraint function from neural net
        constraints = self.build_constraint_from_network()
        for d in j1:
            policy = np.reshape(d[-1],(self.cmdp.mdp.S,self.cmdp.mdp.A))
            c = self.compute_expected_constraint_violation(policy, constraints)
            s = np.argwhere(np.all(self.cmdp.mdp.state_space == d[0]))[0]
            j1_cv.append(c[s])

        for d in j0:
            policy = np.reshape(d[-1],(self.cmdp.mdp.S,self.cmdp.mdp.A))
            c = self.compute_expected_constraint_violation(policy, constraints)
            s = np.argwhere(np.all(self.cmdp.mdp.state_space == d[0]))[0]
            j0_cv.append(c[s])

        return [np.mean(j1_cv), np.mean(j0_cv)]'''

class utils:
    H = 0
    S = 1
    A = 2
    R = 3
    S_P = 4

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
    
    # Approximate Q Improvement given an MLE MDP
    @staticmethod 
    def approx_q_improvement(mdp, P_hat, R_hat, epsilon=.01):

        q = np.zeros((P_hat.shape[0],P_hat.shape[1]))
        
        for _ in range(100):
            # Store the current value function
            old_q = q

            # Perform Bellman Backups depending on if the reward is R(s,a,s') or R(s,a)
            if len(R_hat.shape) == 3:
                q = np.sum(P_hat * (R_hat + mdp.gamma * np.max(q, axis=-1)), axis=-1)
                q[mdp.goal_states,:] = 0
            elif len(R_hat.shape) == 2: 
                q = np.sum(P_hat * (R_hat[:,:,np.newaxis] + mdp.gamma * np.max(q, axis=-1)), axis=-1)
                q[mdp.goal_states,:] = 0
            # R(s)
            else: 
                q = np.sum(P_hat * (R_hat[:,np.newaxis,np.newaxis] + mdp.gamma * np.max(q, axis=-1)), axis=-1)

            if np.max(np.abs(old_q-q)) < epsilon:
                break
                

        # Determine the best actions with bellman backups 
        if len(R_hat.shape) == 3:
            action_reward = np.sum(P_hat * (R_hat + mdp.gamma * np.max(q, axis=-1)), axis=-1)
        else:
            action_reward = np.sum(P_hat * (R_hat[:,:,np.newaxis] + mdp.gamma * np.max(q, axis=-1)), axis=-1)

        # Compute the optimal policy
        best_actions = (action_reward - np.max(action_reward, axis=1)[:,np.newaxis]) == 0
        policy = best_actions * (1 / np.sum( best_actions, axis=1)[:,np.newaxis])

        return policy, q
    
    # Perform Value Iteration given an MDP and a policy
    @staticmethod
    def horizon_approx_v_eval(mdp, P_hat, R_hat, policy, h=100, epsilon=.01):

        v = np.zeros(mdp.S)
        
        for _ in range(h):
            
            # Perform Bellman Backups depending on if the reward is R(s,a,s') or R(s,a)
            if len(mdp.R.shape) == 3:
                v = np.sum( np.sum( P_hat * (R_hat + v), axis=-1) * policy, axis=-1)
                v[mdp.goal_states] = 0
            else:
                v = np.sum( np.sum( P_hat * (R_hat[:,:,np.newaxis] + v), axis=-1) * policy, axis=-1)
                v[mdp.goal_states] = 0

        return v 
    
    # Perform log sum
    @staticmethod
    def logsumexp(x):
        max_x = np.max(x)
        sum_exp = 0.0
        for xi in x:
            sum_exp += np.exp(xi - max_x)
        return max(x) + np.log(sum_exp)