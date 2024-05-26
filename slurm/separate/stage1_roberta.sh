#!/bin/bash

name="st1"
nrows=2000
per_device_train_batch_size=128
per_device_eval_batch_size=128
epochs=50
lora_alpha=1
lora_dropout=0.1
lora_r=16
learning_rate=2e-5
weight_decay=0.01
model_name="roberta"
text_model="sentence-transformers/all-distilroberta-v1"
root="/scratch/$USER/AML_dataset/AMAZON_FASHION.csv"

cpus_per_task=1
mem_per_cpu=30GB
gpu=false
partition="gpu-a100"
time="00:10:00"

# Construct the job name dynamically
job_name="${name}_${model_name}_rows${nrows}_bs-emb${per_device_train_batch_size}_ep${epochs}_r${lora_r}_cpus${cpus_per_task}_mem${mem_per_cpu}" 
if [ $gpu == true ]; then
    job_name="${job_name}_gpu"
fi

checkpoint_dir="./checkpoint_dir/${job_name}"

# Path for the generated SLURM script
generated_script_path="/home/$USER/cse3000/slurm/separate/stage1/$model_name/scripts/${job_name}.sh"

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
#SBATCH --output=/home/%u/cse3000/slurm/separate/stage1/$model_name/logs/%x_%j.out
#SBATCH --error=/home/%u/cse3000/slurm/separate/stage1/$model_name/logs/%x_%j.err

module load miniconda3

unset CONDA_SHLVL
source "\$(conda info --base)/etc/profile.d/conda.sh"

conda activate rel-mm

srun python /home/$USER/cse3000/finetune_LLM.py --name=$job_name --nrows=$nrows --epochs=$epochs --text_model=$text_model --per_device_train_batch_size=$per_device_train_batch_size --per_device_eval_batch_size=$per_device_eval_batch_size --lora_alpha=$lora_alpha --lora_dropout=$lora_dropout --lora_r=$lora_r --learning_rate=$learning_rate --weight_decay=$weight_decay --checkpoint_dir=$checkpoint_dir --root=$root

conda deactivate
EOT

# Submit the generated SLURM script
sbatch $generated_script_path
