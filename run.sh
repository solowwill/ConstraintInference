#!/bin/bash

#SBATCH -t 1-12:00:00
#SBATCH -J icrl_bayesian
#SBATCH -A eecs
#SBATCH -p share
#SBATCH -n 1
#SBATCH -o icrl_bayesian.out
#SBATCH -e icrl_bayesian.err
#SBATCH --mem=8G

python3 run_icrl.py --env_type grid_lava --env 6x6_lava --d_size 2000 --num_d 10 --path /nfs/hpc/share/soloww/icrl/ 




