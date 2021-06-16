#!/bin/bash

#SBATCH --job-name=dkrlhet
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:q6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --time=2-00:00:00
#SBATCH --array=0-26
#SBATCH --output=slurm/dkrlhet-%A-%a.log
#SBATCH --error=slurm/dkrlhet-%A-%a.log
#SBATCH --export=all

python script/training/hetionet/07_train_dkrl_grid.py $SLURM_ARRAY_TASK_ID
