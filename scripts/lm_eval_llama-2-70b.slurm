#!/bin/bash
#SBATCH --job-name=lm_eval_llama-2-70b
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=0
#SBATCH --mem=100GB
#SBATCH --gres=gpu:4
#SBATCH --output=.slurm/lm_eval_llama-2-70b.out
#SBATCH --error=.slurm/lm_eval_llama-2-70b.err

# activate virtual environment
source /gaueko0/users/jetxaniz007/phd/venv2/bin/activate

path="meta-llama"
model="Llama-2-70b-hf"
model_name=$path/$model
folder="llama-2"

# select tasks
tasks_selected=(
    "basquetrivia_eu"
    "basquetrivia_en"
    #"basquetrivia_en_mt"
)

num_fewshot=5

for group_name in "${tasks_selected[@]}"; do
    srun python3 -m lm_eval \
        --model hf \
        --model_args pretrained=$model_name,parallelize=True \
        --tasks $group_name \
        --device cuda \
        --output_path ../results/${folder}/${model}/${group_name}_${num_fewshot}-shot.json \
        --batch_size auto \
        --num_fewshot ${num_fewshot} \
        --log_samples
done
