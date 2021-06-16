#!/bin/bash

#SBATCH --job-name=msiensm
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=2G
#SBATCH --time=1-00:00:00
#SBATCH --array=0-39
#SBATCH --output=slurm/msiensm-%A-%a.log
#SBATCH --error=slurm/msiensm-%A-%a.log
#SBATCH --export=all

python script/training/msi/08_router_ensemble_multi.py $SLURM_ARRAY_TASK_ID
