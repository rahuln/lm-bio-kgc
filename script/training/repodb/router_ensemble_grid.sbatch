#!/bin/bash

#SBATCH --job-name=routrep
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=2G
#SBATCH --time=1-00:00:00
#SBATCH --array=0-99
#SBATCH --output=slurm/routrep-%A-%a.log
#SBATCH --error=slurm/routrep-%A-%a.log
#SBATCH --export=all

python script/training/repodb/04_router_ensemble.py $SLURM_ARRAY_TASK_ID
