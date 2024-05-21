#!/bin/bash

# Parameters
name="fa_st2"
nrows=100000
cpus_per_task=15
mem_per_cpu=8GB
st2_epochs=50
st2_batch_size=256
st2_batch_size_embedder=64
gpu=true
partition="gpu-a100"
time="01:00:00"
text_model="/home/$USER/cse3000/checkpoints/checkpoint-22500"

# Construct the job name dynamically, append gpu to the job name if GPU is used
job_name="${name}_nrows${nrows}_s2epochs${st2_epochs}_s2bs${st2_batch_size}_s2bs_embedder${st2_batch_size_embedder}_cpus${cpus_per_task}_mem${mem_per_cpu}"
if [ $gpu == true ]; then
    job_name="${job_name}_gpu"
fi

# Path for the generated SLURM script
generated_script_path="/home/$USER/cse3000/slurm/separate/scripts/${job_name}.sh"

# Create the SLURM script
cat <<EOT > $generated_script_path
#!/bin/bash

#SBATCH --job-name="$job_name"
#SBATCH --time=$time
#SBATCH --ntasks=1
#SBATCH --partition=$partition
#SBATCH --mem-per-cpu=$mem_per_cpu
#SBATCH --cpus-per-task=$cpus_per_task
$([ $gpu == true ] && echo "#SBATCH --gpus-per-task=1")
#SBATCH --account=education-eemcs-courses-cse3000
#SBATCH --output=/home/%u/cse3000/slurm/separate/fa_st2/%x_%j.out
#SBATCH --error=/home/%u/cse3000/slurm/separate/fa_st2/%x_%j.err

module load miniconda3

unset CONDA_SHLVL
source "\$(conda info --base)/etc/profile.d/conda.sh"

conda activate rel-mm-clean

# Run the Python script with the specified parameters
srun python /home/$USER/cse3000/s2.py --name=$job_name --nrows=$nrows --text_model=$text_model --task_type=regression --st2_epochs=$st2_epochs --st2_batch_size=$st2_batch_size --st2_batch_size_embedder=$st2_batch_size_embedder

conda deactivate
EOT

# Submit the generated SLURM script
sbatch $generated_script_path
