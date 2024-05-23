#!/bin/bash

# Parameters
name="1step"
nrows=100000
batch_size=256
batch_size_tokenizer=50000
epochs=50
finetune=true
cpus_per_task=15
mem_per_cpu=8GB
one_step_lora_alpha=1
one_step_lora_dropout=0.1
one_step_r=16
gpu=true
partition="gpu-a100"
time="08:00:00"
text_model="sentence-transformers/all-distilroberta-v1"

# Construct the job name dynamically
job_name="${name}_rows${nrows}_bs${batch_size}_bstok${batch_size_tokenizer}_ep${epochs}_cpus${cpus_per_task}_mem${mem_per_cpu}"
if [ $gpu == true ]; then
    job_name="${job_name}_gpu"
fi

# Path for the generated SLURM script
generated_script_path="/home/$USER/cse3000/slurm/one_step/scripts/${job_name}.sh"

# Create the SLURM script
cat <<EOT > $generated_script_path
#!/bin/bash

#SBATCH --job-name="$job_name"
#SBATCH --time=$time
#SBATCH --ntasks=1
#SBATCH --partition=$partition
#SBATCH --mem-per-cpu=$mem_per_cpu
$([ $gpu == true ] && echo "#SBATCH --gpus-per-task=1")
#SBATCH --cpus-per-task=$cpus_per_task
#SBATCH --account=education-eemcs-courses-cse3000
#SBATCH --output=/home/%u/cse3000/slurm/one_step/%x_%j.out
#SBATCH --error=/home/%u/cse3000/slurm/one_step/%x_%j.err

module load miniconda3

unset CONDA_SHLVL
source "\$(conda info --base)/etc/profile.d/conda.sh"

conda activate rel-mm

srun python /home/$USER/cse3000/one_step.py --name=$job_name --nrows=$nrows --batch_size=$batch_size --batch_size_tokenizer=$batch_size_tokenizer --epochs=$epochs --text_model=$text_model --task_type="regression" --lora_alpha=$one_step_lora_alpha --lora_dropout=$one_step_lora_dropout --lora_r=$one_step_r

conda deactivate
EOT

# Submit the generated SLURM script
sbatch $generated_script_path
