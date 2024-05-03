#!/bin/bash

#SBATCH --job-name="ll"
#SBATCH --time=00:10:00
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --partition=gpu-a100
#SBATCH --mem-per-cpu=20GB
#SBATCH --account=education-eemcs-courses-cse3000
#SBATCH --output=./%j.out # standard output of the job will be printed here
#SBATCH --error=./%j.err # standard error of the job will be printed here

previous=$(nvidia-smi --query-accounted-apps='gpu_utilization,mem_utilization,max_memory_usage,time' --format='csv' | /usr/bin/tail -n '+2')
nvidia-smi

module load miniconda3

unset CONDA_SHLVL
source "$(conda info --base)/etc/profile.d/conda.sh"

conda activate rel-mm

srun python /home/$USER/cse3000/self-supervised.py

conda deactivate

nvidia-smi --query-accounted-apps='gpu_utilization,mem_utilization,max_memory_usage,time' --format='csv' | /usr/bin/grep -v -F "$previous"

