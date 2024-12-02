#!/bin/bash

# Note: This is configured for Daytona's systems.

#SBATCH --job-name=trial_lr9en4_bs50
#SBATCH --open-mode=append
#SBATCH --output=./artifacts/outputs/slurm.out
#SBATCH --error=./artifacts/outputs/slurm.err
#SBATCH --time=00:30:00
#SBATCH --mem=12GB
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1

python -u -m train_label_classifier