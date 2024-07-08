#!/bin/bash

#SBATCH --job-name="pna"
#SBATCH --time=48:00:00
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=5
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-task=24GB
#SBATCH --account=research-eemcs-st
#SBATCH --output=./%j.out # standard output of the job will be printed here
#SBATCH --error=./%j.err # standard error of the job will be printed here

previous=$(nvidia-smi --query-accounted-apps='gpu_utilization,mem_utilization,max_memory_usage,time' --format='csv' | /usr/bin/tail -n '+2')
nvidia-smi

module load miniconda3

unset CONDA_SHLVL
source "$(conda info --base)/etc/profile.d/conda.sh"

conda activate rel-mm

srun python /home/$USER/models-for-relational-multimodal-data/pna.py

conda deactivate

nvidia-smi --query-accounted-apps='gpu_utilization,mem_utilization,max_memory_usage,time' --format='csv' | /usr/bin/grep -v -F "$previous"

