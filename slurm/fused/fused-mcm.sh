#!/bin/bash

#SBATCH --job-name="fmcm"
#SBATCH --time=96:00:00
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --gpu-bind=none
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=16GB
#SBATCH --account=research-eemcs-st
#SBATCH --output=./%j.out # standard output of the job will be printed here
#SBATCH --error=./%j.err # standard error of the job will be printed here

previous=$(nvidia-smi --query-accounted-apps='gpu_utilization,mem_utilization,max_memory_usage,time' --format='csv' | /usr/bin/tail -n '+2')
nvidia-smi

module load miniconda3

unset CONDA_SHLVL
source "$(conda info --base)/etc/profile.d/conda.sh"

conda activate rel-mm

srun python /home/$USER/models-for-relational-multimodal-data/fused.py --dataset /scratch/takyildiz/ibm-transactions-for-anti-money-laundering-aml/LI-Small_Trans-l.csv --wandb_dir /scratch/takyildiz/ --save_dir /scratch/takyildiz/ --mode mcm --group LS-fused-mcm --run_name LS-fused-mcm --epochs 30 --ports --ego --reverse_mp --batch_size 1024
srun python /home/$USER/models-for-relational-multimodal-data/fused.py --dataset /scratch/takyildiz/ibm-transactions-for-anti-money-laundering-aml/HI-Small_Trans-l.csv --wandb_dir /scratch/takyildiz/ --save_dir /scratch/takyildiz/ --mode mcm --group HS-fused-mcm --run_name HS-fused-mcm --epochs 30 --ports --ego --reverse_mp --batch_size 1024
#srun python /home/$USER/models-for-relational-multimodal-data/fused.py --dataset /scratch/takyildiz/ibm-transactions-for-anti-money-laundering-aml/HI-Medium_Trans-l.csv --wandb_dir /scratch/takyildiz/ --save_dir /scratch/takyildiz/ --mode mcm --group HM-fused-mcm --run_name HM-fused-mcm --epochs 30 --ports --ego --reverse_mp --batch_size 1024
#srun python /home/$USER/models-for-relational-multimodal-data/fused.py --dataset /scratch/takyildiz/ibm-transactions-for-anti-money-laundering-aml/LI-Medium_Trans-l.csv --wandb_dir /scratch/takyildiz/ --save_dir /scratch/takyildiz/ --mode mcm --group LM-fused-mcm --run_name LM-fused-mcm --epochs 30 --ports --ego --reverse_mp --batch_size 1024

conda deactivate

nvidia-smi --query-accounted-apps='gpu_utilization,mem_utilization,max_memory_usage,time' --format='csv' | /usr/bin/grep -v -F "$previous"

