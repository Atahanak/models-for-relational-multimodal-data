#!/bin/bash

#SBATCH --job-name="fashion_nrows10000_bs512_bsemb5_bstok20000_epochs100_finetunetrue"
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

# Run the Python script with the specified parameters
srun python /home/cgriu/cse3000/text_example.py --name=fashion_nrows10000_bs512_bsemb5_bstok20000_epochs100_finetunetrue --nrows=10000 --batch_size=512 --batch_size_embedder=5 --batch_size_tokenizer=20000 --epochs=100 --finetune

conda deactivate
