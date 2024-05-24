#!/bin/bash

#SBATCH --job-name="mask_type_comparison"
#SBATCH --time=14:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=compute-p2
#SBATCH --mem-per-cpu=16GB
#SBATCH --account=education-eemcs-courses-cse3000
#SBATCH --output=./%j.out # standard output of the job will be printed here
#SBATCH --error=./%j.err # standard error of the job will be printed here

previous=$(nvidia-smi --query-accounted-apps='gpu_utilization,mem_utilization,max_memory_usage,time' --format='csv' | /usr/bin/tail -n '+2')
nvidia-smi

module load miniconda3

unset CONDA_SHLVL
source "$(conda info --base)/etc/profile.d/conda.sh"

conda activate rel-mm

for i in {1..4}
do
   srun python /home/$USER/cse3000/mask_strategy_comparison.py --strategy="remove"

   srun python /home/$USER/cse3000/bert_mask.py --strategy="replace"

   srun python /home/$USER/cse3000/replace_mask.py --strategy="bert"

   rm "/scratch/$USER/masked_columns.npy"
done

conda deactivate

nvidia-smi --query-accounted-apps='gpu_utilization,mem_utilization,max_memory_usage,time' --format='csv' | /usr/bin/grep -v -F "$previous"

