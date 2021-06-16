#!/bin/bash

#SBATCH --job-name=kgmsiind
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --time=1-00:00:00
#SBATCH --array=0-3
#SBATCH --output=slurm/kgmsiind-%A-%a.log
#SBATCH --export=all

python script/training/msi/02_train_kge_grid_ind.py $SLURM_ARRAY_TASK_ID
