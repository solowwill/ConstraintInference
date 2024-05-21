import matplotlib.pyplot as plt
import numpy as np
import argparse

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
    dkls = [5, 1, .1, .01, .001]

    data = np.load(f'data/{env_type}/{env}/bayesian_b{baseline}_data.npy')

    data_mean = np.mean(data,axis=-2)
    data_std = np.std(data,axis=-2)

    fig,ax = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=False, figsize=(10,6))
    fig.add_subplot(111, frameon=False)
    
    for i in range(len(iterations)):
        ax[0,0].plot(dsizes, data_mean[:,i,0], label=f'Iterations={iterations[i]}')
        ax[0,0].fill_between(dsizes, data_mean[:,i,0]-data_std[:,i,0], data_mean[:,i,0]+data_std[:,i,0], alpha=.5)
        ax[0,0].set_ylabel('Precision')
        
        ax[0,1].plot(dsizes, data_mean[:,i,2], label=f'Iterations={iterations[i]}')
        ax[0,1].fill_between(dsizes, data_mean[:,i,2]-data_std[:,i,2], data_mean[:,i,2]+data_std[:,i,2], alpha=.5)
        ax[0,1].set_ylabel('Accuracy')

        ax[1,0].plot(dsizes, data_mean[:,i,3], label=f'Iterations={iterations[i]}')
        ax[1,0].fill_between(dsizes,data_mean[:,i,3]-data_std[:,i,3], data_mean[:,i,3]+data_std[:,i,3], alpha=.5)
        ax[1,0].set_ylabel('False Positive Rate')

        ax[1,1].plot(dsizes, data_mean[:,i,4], label=f'Iterations={iterations[i]}')
        ax[1,1].fill_between(dsizes, data_mean[:,i,4]-data_std[:,i,4], data_mean[:,i,4]+data_std[:,i,4],alpha=.5)
        ax[1,1].set_ylabel('False Negative Rate')
        ax[1,1].legend(loc='lower right')

    plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
    plt.xlabel("Number of Demonstrations")
    plt.title(f'Bayesian ICRL with Baseline Policy {baseline}')

    plt.savefig(f'data/{env_type}/{env}/bayesian_b{baseline}_fig.png')
    plt.show()



    data = np.load(f'data/{env_type}/{env}/maxent_b{baseline}_data.npy')

    data_mean = np.mean(data,axis=-2)
    data_std = np.std(data,axis=-2)

    fig,ax = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=False, figsize=(10,6))
    fig.add_subplot(111, frameon=False)
    
    for i in range(len(dkls)):
        ax[0,0].plot(dsizes, data_mean[:,i,0], label=f'DKL={dkls[i]}')
        ax[0,0].fill_between(dsizes, data_mean[:,i,0]-data_std[:,i,0], data_mean[:,i,0]+data_std[:,i,0], alpha=.5)
        ax[0,0].set_ylabel('Precision')

        ax[0,1].plot(dsizes, data_mean[:,i,2], label=f'DKL={dkls[i]}')
        ax[0,1].fill_between(dsizes, data_mean[:,i,2]-data_std[:,i,2], data_mean[:,i,2]+data_std[:,i,2], alpha=.5)
        ax[0,1].set_ylabel('Accuracy')

        ax[1,0].plot(dsizes, data_mean[:,i,3], label=f'DKL={dkls[i]}')
        ax[1,0].fill_between(dsizes,data_mean[:,i,3]-data_std[:,i,3], data_mean[:,i,3]+data_std[:,i,3], alpha=.5)
        ax[1,0].set_ylabel('False Positive Rate')

        ax[1,1].plot(dsizes, data_mean[:,i,4], label=f'DKL={dkls[i]}')
        ax[1,1].fill_between(dsizes, data_mean[:,i,4]-data_std[:,i,4], data_mean[:,i,4]+data_std[:,i,4],alpha=.5)
        ax[1,1].set_ylabel('False Negative Rate')
        ax[1,1].legend(loc='lower right')

    plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
    plt.xlabel("Number of Demonstrations")
    plt.title(f'MaxEnt ICRL with Baseline Policy {baseline}')
    

    plt.savefig(f'data/{env_type}/{env}/maxent_b{baseline}_fig.png')
    plt.show()


if __name__ == "__main__":
    main()