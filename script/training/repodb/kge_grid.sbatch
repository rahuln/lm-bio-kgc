#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=8G
#SBATCH --time=1-00:00:00
#SBATCH --array=0-191
#SBATCH --output=slurm/kgerepo-%A-%a.log
#SBATCH --export=all

python script/training/repodb/01_train_kge_grid.py $SLURM_ARRAY_TASK_ID
