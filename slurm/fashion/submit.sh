#!/bin/bash

# Parameters
name="fashion"
nrows=10000
batch_size=128
batch_size_embedder=5
batch_size_tokenizer=20000
epochs=100
finetune=true

# Construct the job name dynamically
job_name="${name}_nrows${nrows}_bs${batch_size}_bsemb${batch_size_embedder}_bstok${batch_size_tokenizer}_epochs${epochs}_finetune${finetune}"

# Path for the generated SLURM script
generated_script_path="/home/$USER/cse3000/slurm/fashion/${job_name}.sh"

# Create the SLURM script
cat <<EOT > $generated_script_path
#!/bin/bash

#SBATCH --job-name="$job_name"
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
source "\$(conda info --base)/etc/profile.d/conda.sh"

conda activate rel-mm-clean

# Run the Python script with the specified parameters
srun python /home/$USER/cse3000/text_example.py --name=$job_name --nrows=$nrows --batch_size=$batch_size --batch_size_embedder=$batch_size_embedder --batch_size_tokenizer=$batch_size_tokenizer --epochs=$epochs $([ $finetune == true ] && echo "--finetune")

conda deactivate
EOT

# Submit the generated SLURM script
sbatch $generated_script_path
