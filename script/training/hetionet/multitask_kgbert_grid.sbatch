#!/bin/bash

#SBATCH --job-name=kgbhet
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=16G
#SBATCH --time=3-00:00:00
#SBATCH --array=0-35
#SBATCH --output=slurm/kgbhet-%A-%a.log
#SBATCH --error=slurm/kgbhet-%A-%a.log
#SBATCH --export=all

python script/training/hetionet/01_train_multitask_kgbert_grid.py $SLURM_ARRAY_TASK_ID
