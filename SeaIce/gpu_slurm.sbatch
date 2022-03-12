#!/bin/bash
#SBATCH --partition=lotus_gpu
#SBATCH --account=lotus_gpu
#SBATCH --gres=gpu:1
#SBATCH -o wd.out
#SBATCH -e wd.err
#SBATCH --time=24:00:00
#SBATCH --ntasks=32
#SBATCH --ntasks-per-node=32
#SBATCH --mem=32000
# executable
python3 tune_hyperparameters_wd.py
