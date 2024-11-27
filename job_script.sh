#!/bin/bash

#SBATCH --job-name=train_bert
#SBATCH --open-mode=append
#SBATCH --output=./artifacts/training/train_bert.out
#SBATCH --error=./artifacts/training/train_bert.err
#SBATCH --time=05:00:00
#SBATCH --mem=12GB
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1

hostname
date