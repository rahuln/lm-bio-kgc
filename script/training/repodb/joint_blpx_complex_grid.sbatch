#!/bin/bash

#SBATCH --job-name=jntrepo
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --time=2-00:00:00
#SBATCH --array=0-2
#SBATCH --output=slurm/jntrepo-%A-%a.log
#SBATCH --error=slurm/jntrepo-%A-%a.log
#SBATCH --export=all

python script/training/repodb/06_train_joint_blpx_complex_grid.py $SLURM_ARRAY_TASK_ID
