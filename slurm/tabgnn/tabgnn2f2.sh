#!/bin/bash

#SBATCH --job-name="tg2fmoco"
#SBATCH --time=72:00:00
#SBATCH --partition=gpu-a100
#SBATCH --gres=gpu:1
#SBATCH --gpu-bind=none
#SBATCH --ntasks=3
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8GB
#SBATCH --account=research-eemcs-st
#SBATCH --output=./%j.out # standard output of the job will be printed here
#SBATCH --error=./%j.err # standard error of the job will be printed here

previous=$(nvidia-smi --query-accounted-apps='gpu_utilization,mem_utilization,max_memory_usage,time' --format='csv' | /usr/bin/tail -n '+2')
nvidia-smi

module load miniconda3

unset CONDA_SHLVL
source "$(conda info --base)/etc/profile.d/conda.sh"

conda activate rel-mm

srun python /home/$USER/models-for-relational-multimodal-data/tabular-gnn2.py

conda deactivate

nvidia-smi --query-accounted-apps='gpu_utilization,mem_utilization,max_memory_usage,time' --format='csv' | /usr/bin/grep -v -F "$previous"

