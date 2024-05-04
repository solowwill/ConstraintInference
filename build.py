# build.py
# Contains functions for generating datasets, baseline policies, and gridworlds
import numpy as np
import lib.utils as utils

# Build a dataset of trajectories
def build_D(mdp, policy, num_traj):

    trajectories = []

    for _ in range(num_traj):
        sars, g = mdp.sample_trajectory(policy)

        trajectories.append(sars)

    return trajectories


# Generate a baseline policy
def gen_baseline(mdp, targ_performance, dec=.05, epsilon = .001):

    opt_pol, q = utils.q_improvement(mdp)
    v_opt = np.sum(opt_pol * q, axis=-1)
    v_rand = utils.v_eval(mdp,mdp.rand_pol())

    pi_b = np.copy(opt_pol)
    pib_perf = compute_performance(mdp,pi_b, v_opt, v_rand)

    # Iteratively remove action weight from the best action
    while (pib_perf + epsilon) > targ_performance:

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

            if i > mdp.S:
                print('Cannot find baseline')
                return None

        pi_b[s,best_a] -= dec

        a = np.random.choice(actions[actions != best_a],1)[0]
        pi_b[s,a] += dec

        pib_perf = compute_performance(mdp, pi_b, v_opt, v_rand)
    
    return pi_b

# Compute the normalized performance 
# Where optimal policy = 1 and uniform random policy = 0
def compute_performance(mdp, policy, v_opt, v_rand):
    return (utils.v_eval(mdp,policy)[mdp.init_state] - v_rand[mdp.init_state]) / \
            (v_opt[mdp.init_state] - v_rand[mdp.init_state])


# Normalize the performance 
# Where optimal policy = 1 and uniform random policy = 0
def normalize_perf(mdp, perf, v_opt, v_rand):

    return (perf - v_rand[mdp.init_state]) / \
            (v_opt[mdp.init_state] - v_rand[mdp.init_state])


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