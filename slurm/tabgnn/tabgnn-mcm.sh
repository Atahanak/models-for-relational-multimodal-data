#!/bin/bash

#SBATCH --job-name="tmcm"
#SBATCH --time=120:00:00
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --gpu-bind=none
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
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

#srun python /home/$USER/models-for-relational-multimodal-data/data/prepare_AML_transactions.py -i /scratch/takyildiz/ibm-transactions-for-anti-money-laundering-aml/HI-Large_Trans.csv -o /scratch/takyildiz/ibm-transactions-for-anti-money-laundering-aml/HI-Large_Trans-l.csv
#srun python /home/$USER/models-for-relational-multimodal-data/tabgnn.py --dataset /scratch/takyildiz/ibm-transactions-for-anti-money-laundering-aml/HI-Large_Trans-l.csv --wandb_dir /scratch/takyildiz/ --save_dir /scratch/takyildiz/ --mode mcm --group HL-tabgnn-mcm --run_name HL-tabgnn-mcm --epochs 30 --batch_size 1024 --ports --ego --reverse_mp
#srun python /home/$USER/models-for-relational-multimodal-data/tabgnn.py --dataset /scratch/takyildiz/ibm-transactions-for-anti-money-laundering-aml/HI-Small_Trans-l.csv --wandb_dir /scratch/takyildiz/ --save_dir /scratch/takyildiz/ --mode mcm --group HS-tabgnn-mcm --run_name HS-tabgnn-mcm --epochs 30 --batch_size 4096 --ports --ego --reverse_mp
#srun python /home/$USER/models-for-relational-multimodal-data/tabgnn.py --dataset /scratch/takyildiz/ibm-transactions-for-anti-money-laundering-aml/LI-Small_Trans-l.csv --wandb_dir /scratch/takyildiz/ --save_dir /scratch/takyildiz/ --mode mcm --group LS-tabgnn-mcm --run_name LS-tabgnn-mcm --epochs 30 --batch_size 4096 --ports --ego --reverse_mp
srun python /home/$USER/models-for-relational-multimodal-data/tabgnn.py --testing False --dataset /scratch/takyildiz/ibm-transactions-for-anti-money-laundering-aml/HI-Medium_Trans-l.csv --wandb_dir /scratch/takyildiz/ --save_dir /scratch/takyildiz/ --mode mcm --group HM-tabgnn-mcm --run_name HM-tabgnn-mcm --epochs 30 --batch_size 1024 --ports --ego --reverse_mp
#srun python /home/$USER/models-for-relational-multimodal-data/tabgnn.py --dataset /scratch/takyildiz/ibm-transactions-for-anti-money-laundering-aml/LI-Medium_Trans-l.csv --wandb_dir /scratch/takyildiz/ --save_dir /scratch/takyildiz/ --mode mcm --group LM-tabgnn-mcm --run_name LM-tabgnn-mcm --epochs 30 --batch_size 1024 --ports --ego --reverse_mp

conda deactivate

nvidia-smi --query-accounted-apps='gpu_utilization,mem_utilization,max_memory_usage,time' --format='csv' | /usr/bin/grep -v -F "$previous"

