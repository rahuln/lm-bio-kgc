#!/bin/bash

#SBATCH --job-name=kgbhetin
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=16G
#SBATCH --time=3-00:00:00
#SBATCH --array=0
#SBATCH --output=slurm/kgbhetin-%A-%a.log
#SBATCH --error=slurm/kgbhetin-%A-%a.log
#SBATCH --export=all

python script/training/hetionet/02_train_multitask_kgbert_grid_ind.py $SLURM_ARRAY_TASK_ID
