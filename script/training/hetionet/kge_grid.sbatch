#!/bin/bash

#SBATCH --job-name=kgehet
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=8G
#SBATCH --time=1-00:00:00
#SBATCH --array=0-191
#SBATCH --output=slurm/kgehet-%A-%a.log
#SBATCH --export=all

python script/training/hetionet/01_train_kge_grid.py $SLURM_ARRAY_TASK_ID
