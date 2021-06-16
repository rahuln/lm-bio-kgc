#!/bin/bash

#SBATCH --job-name=routhet
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=2G
#SBATCH --time=1-00:00:00
#SBATCH --array=0-99
#SBATCH --output=slurm/routhet-%A-%a.log
#SBATCH --error=slurm/routhet-%A-%a.log
#SBATCH --export=all

python script/training/hetionet/04_router_ensemble.py $SLURM_ARRAY_TASK_ID
