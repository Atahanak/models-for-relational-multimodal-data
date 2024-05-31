#!/bin/bash

name="no_finetune"
nrows=2000
batch_size=1
batch_size_embedder=1
epochs=50
task_type="regression"
model_name="roberta"
text_model="sentence-transformers/all-distilroberta-v1"
root="/scratch/$USER/AML_dataset/AMAZON_FASHION.csv"

cpus_per_task=15
mem_per_cpu=8GB
gpu=false
partition="gpu-a100"
time="00:10:00"

# Construct the job name dynamically
job_name="${name}_${model_name}_rows${nrows}_ep${epochs}_bs${batch_size}_bs-emb${batch_size_embedder}_cpus${cpus_per_task}_mem${mem_per_cpu}"
if [ $gpu == true ]; then
    job_name="${job_name}_gpu"
fi

# Path for the generated SLURM script
generated_script_path="/home/$USER/cse3000/slurm/$name/$model_name/scripts/${job_name}.sh"

mkdir -p /home/$USER/cse3000/slurm/$name/$model_name/scripts
mkdir -p /home/$USER/cse3000/slurm/$name/$model_name/logs

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
#SBATCH --output=/home/%u/cse3000/slurm/$name/$model_name/logs/%x_%j.out
#SBATCH --error=/home/%u/cse3000/slurm/$name/$model_name/logs/%x_%j.err

module load miniconda3

unset CONDA_SHLVL
source "\$(conda info --base)/etc/profile.d/conda.sh"

conda activate rel-mm

srun python /home/$USER/cse3000/downstream_model_LLM.py --name=$job_name --nrows=$nrows --text_model=$text_model --task_type=$task_type --epochs=$epochs --batch_size=$batch_size --batch_size_embedder=$batch_size_embedder --script_path=$generated_script_path --root=$root

conda deactivate
EOT

# Submit the generated SLURM script
sbatch $generated_script_path
