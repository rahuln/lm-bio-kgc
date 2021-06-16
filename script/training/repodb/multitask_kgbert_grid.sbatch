#!/bin/bash

#SBATCH --job-name=kgbrepo
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --time=36:00:00
#SBATCH --array=0-35
#SBATCH --output=slurm/kgbrepo-%A-%a.log
#SBATCH --error=slurm/kgbrepo-%A-%a.log
#SBATCH --export=all

python script/training/repodb/01_train_multitask_kgbert_grid.py $SLURM_ARRAY_TASK_ID
