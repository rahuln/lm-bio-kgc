#!/bin/bash

#SBATCH --job-name=avgrep
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=2G
#SBATCH --time=1-00:00:00
#SBATCH --array=0-719
#SBATCH --output=slurm/avgrep-%A-%a.log
#SBATCH --error=slurm/avgrep-%A-%a.log
#SBATCH --export=all

python script/training/repodb/09_average_ranking_ensemble.py $SLURM_ARRAY_TASK_ID
