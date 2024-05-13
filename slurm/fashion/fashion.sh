#!/bin/bash

#SBATCH --job-name="fashion"
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --partition=gpu-v100
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=80GB
#SBATCH --cpus-per-task=1          
#SBATCH --account=education-eemcs-courses-cse3000
#SBATCH --output=/home/%u/cse3000/slurm/fashion/%x_%j.out
#SBATCH --error=/home/%u/cse3000/slurm/fashion/%x_%j.err

module load miniconda3

unset CONDA_SHLVL
source "$(conda info --base)/etc/profile.d/conda.sh"

conda activate rel-mm-clean

srun python /home/$USER/cse3000/fashion.py --name=$SLURM_JOB_NAME --nrows=10000 --batch_size_embedder=5 --batch_size_tokenizer=1024 --epochs=10

conda deactivate
