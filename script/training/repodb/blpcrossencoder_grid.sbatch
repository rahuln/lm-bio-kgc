#!/bin/bash

#SBATCH --job-name=blpxrepo
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --time=2-00:00:00
#SBATCH --array=0-5
#SBATCH --output=slurm/blpxrepo-%A-%a.log
#SBATCH --error=slurm/blpxrepo-%A-%a.log
#SBATCH --export=all

python script/training/repodb/05_train_blpcrossencoder_grid.py $SLURM_ARRAY_TASK_ID
