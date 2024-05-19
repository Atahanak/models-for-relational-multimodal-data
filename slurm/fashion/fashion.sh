#!/bin/bash

name="fashion"
nrows=10000
batch_size=512
batch_size_embedder=5
batch_size_tokenizer=20000
epochs=100
finetune=true

# Construct the job name dynamically
job_name="${name}_nrows${nrows}_bs${batch_size}_bsemb${batch_size_embedder}_bstok${batch_size_tokenizer}_epochs${epochs}_finetune${finetune}"

#SBATCH --job-name=$job_name
#SBATCH --time=1:00:00
#SBATCH --ntasks=1
#SBATCH --partition=gpu-a100
#SBATCH --mem-per-cpu=8GB
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=15    
#SBATCH --account=education-eemcs-courses-cse3000
#SBATCH --output=/home/%u/cse3000/slurm/fashion/%x_%j.out
#SBATCH --error=/home/%u/cse3000/slurm/fashion/%x_%j.err

module load miniconda3

unset CONDA_SHLVL
source "$(conda info --base)/etc/profile.d/conda.sh"

conda activate rel-mm-clean

srun python /home/$USER/cse3000/text_example.py --name=$job_name --nrows=$nrows --batch_size=$batch_size --batch_size_embedder=$batch_size_embedder --batch_size_tokenizer=$batch_size_tokenizer --epochs=$epochs --finetune=$finetune

conda deactivate
