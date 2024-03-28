#!/bin/bash
#SBATCH --time=7:59:00
#SBATCH --mem=10GB
#SBATCH --job-name=tune
#SBATCH --output=%A.out
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=4

# python run.py --path ethanol_ccsd_t-train.npz --test-path ethanol_ccsd_t-test.npz 
# python tune.py
python run.py
