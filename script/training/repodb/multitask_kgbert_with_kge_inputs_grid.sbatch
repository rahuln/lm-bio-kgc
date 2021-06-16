#!/bin/bash

#SBATCH --job-name=kgbe-repodb
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=16G
#SBATCH --time=1-00:00:00
#SBATCH --array=0-5
#SBATCH --output=slurm/kgbe-repodb-%A-%a.log
#SBATCH --error=slurm/kgbe-repodb-%A-%a.log
#SBATCH --export=all

python script/training/repodb/03_train_multitask_kgbert_with_kge_inputs_grid.py $SLURM_ARRAY_TASK_ID
