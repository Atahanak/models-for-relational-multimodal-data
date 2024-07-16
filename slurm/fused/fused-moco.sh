#!/bin/bash

#SBATCH --job-name="fu-moco"
#SBATCH --time=96:00:00
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
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

srun python /home/$USER/models-for-relational-multimodal-data/fused.py --dataset /scratch/takyildiz/ibm-transactions-for-anti-money-laundering-aml/LI-Small_Trans_c.csv --wandb_dir /scratch/takyildiz/ --save_dir /scratch/takyildiz/ --testing False --moo moco  --group LI-Small,fused,mcm-lp,moco --run_name LI-Small,fused,mcm-lp,moco --epochs 30
srun python /home/$USER/models-for-relational-multimodal-data/fused.py --dataset /scratch/takyildiz/ibm-transactions-for-anti-money-laundering-aml/HI-Small_Trans-c.csv --wandb_dir /scratch/takyildiz/ --save_dir /scratch/takyildiz/ --testing False --moo moco  --group HI-Small,fused,mcm-lp,moco --run_name HI-Small,fused,mcm-lp,moco --epochs 30

conda deactivate

nvidia-smi --query-accounted-apps='gpu_utilization,mem_utilization,max_memory_usage,time' --format='csv' | /usr/bin/grep -v -F "$previous"

