#!/bin/bash

# Parameters
name="fa_st1"
nrows=100000
batch_size=300
batch_size_embedder=5
batch_size_tokenizer=50000
epochs=20
cpus_per_task=1
mem_per_cpu=16GB
st1_epochs=10
st1_lora_alpha=1
st1_lora_dropout=0.1
st1_r=8
st1_per_device_train_batch_size=64
st1_per_device_eval_batch_size=64
st1_learning_rate=2e-5
st1_weight_decay=0.01
gpu=true
partition="gpu-a100"
time="04:00:00"
text_model="sentence-transformers/all-distilroberta-v1"


# Construct the job name dynamically, append gpu to the job name if GPU is used
job_name="${name}_nrows${nrows}_s1bs${st1_per_device_train_batch_size}_st1epochs${st1_epochs}_st1r${st1_r}_cpus${cpus_per_task}_mem${mem_per_cpu}" 
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
#SBATCH --output=/home/%u/cse3000/slurm/separate/fa_st1/%x_%j.out
#SBATCH --error=/home/%u/cse3000/slurm/separate/fa_st1/%x_%j.err

module load miniconda3

unset CONDA_SHLVL
source "\$(conda info --base)/etc/profile.d/conda.sh"

conda activate rel-mm-clean

# Run the Python script with the specified parameters
srun python /home/$USER/cse3000/s1.py --name=$job_name --nrows=$nrows --batch_size=$batch_size --batch_size_embedder=$batch_size_embedder --batch_size_tokenizer=$batch_size_tokenizer --epochs=$epochs --text_model=$text_model --task_type="regression" --st1_per_device_train_batch_size=$st1_per_device_train_batch_size --st1_per_device_eval_batch_size=$st1_per_device_eval_batch_size $([ $finetune == true ] && echo "--finetune") --st1_epochs=$st1_epochs --st1_lora_alpha=$st1_lora_alpha --st1_lora_dropout=$st1_lora_dropout --st1_r=$st1_r --st1_learning_rate=$st1_learning_rate --st1_weight_decay=$st1_weight_decay

conda deactivate
EOT

# Submit the generated SLURM script
sbatch $generated_script_path
