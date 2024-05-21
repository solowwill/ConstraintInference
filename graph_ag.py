# File for graphing the agriculture tree
import matplotlib.pyplot as plt
import numpy as np
import argparse

class utils:
    @staticmethod
    def parse_args():
        
        parser = argparse.ArgumentParser()

        # Envs 
        # ['highway', 'agaid', 'grid', 'grid_lava']
        parser.add_argument("--env_type", type=str, default="agaid", help="grid_lava/grid/highway/agaid")

        # Env Types 
        #    [['highway', 'highway-fast', 'merge', 'roundabout'],
        #    ['intvn28_act4_prec1'],
        #    ['5x5','7x8'],
        #    ['5x5_lava', '7x8_lava]]
        parser.add_argument("--env", type=str, default="intvn28_act4_prec1")

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

    # Load MDP configuration
    P = np.load(f'{path}configs/{env_type}/{env}_P.npy')
    R = np.load(f'{path}configs/{env_type}/{env}_R.npy')
    init_state = np.load(f'{path}configs/{env_type}/{env}_init.npy')
    goal_states = np.load(f'{path}configs/{env_type}/{env}_goal.npy')
    state_space = np.load(f'{path}configs/{env_type}/{env}_statespace.npy')
    action_space = np.load(f'{path}configs/{env_type}/{env}_actionspace.npy')

    print(state_space.shape)
    print(state_space[:,1])

    # Normalize state space
    normed_state_space = (state_space - np.min(state_space,axis=0)) / (np.max(state_space,axis=0)-np.min(state_space,axis=0))

    print(normed_state_space.shape)
    print(normed_state_space[:,1])

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(projection='3d')   

    ax.scatter(normed_state_space[:,0], normed_state_space[:,2], normed_state_space[:,3], c='r')

    ax.set_xlabel('DVS')
    ax.set_ylabel('NUPTT')
    ax.set_zlabel('WSO')

    lines = np.argwhere(P > 0)
    print(lines)

    for s in range(len(lines)):
        x = [normed_state_space[lines[s,0],0], normed_state_space[lines[s,2],0]]
        y = [normed_state_space[lines[s,0],1], normed_state_space[lines[s,2],1]]
        z = [normed_state_space[lines[s,0],2], normed_state_space[lines[s,2],2]]
        ax.plot(x,y,z, c='b', alpha=.7)

    plt.show()

if __name__ == "__main__":
    main()