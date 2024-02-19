#!/bin/bash
#SBATCH --job-name=lm_eval_llama-2_eustrivia_mt
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=0
#SBATCH --mem=50GB
#SBATCH --gres=gpu:1
#SBATCH --output=.slurm/lm_eval_llama-2_eustrivia_mt.out
#SBATCH --error=.slurm/lm_eval_llama-2_eustrivia_mt.err

# activate virtual environment
source /gaueko0/users/jetxaniz007/phd/venv2/bin/activate
export TRANSFORMERS_CACHE="/gaueko1/transformers_cache/"
export TOKENIZERS_PARALLELISM=false

path="meta-llama"
model="Llama-2-7b-hf"
model_name=$path/$model

# load tasks
source tasks.sh

# select tasks
tasks_selected=(
    #"eustrivia"
    #"eustrivia_mt"
    #"eustrivia_mt_itzuli"
    "eustrivia_mt_elhuyar"
)

num_fewshot=5

for group_name in "${tasks_selected[@]}"; do
    srun python3 -m lm_eval \
        --model hf \
        --model_args pretrained=$model_name \
        --tasks $group_name \
        --device cuda \
        --output_path ../results/llama-2/${model}/${model}_${group_name}_${num_fewshot}-shot.json \
        --batch_size auto \
        --num_fewshot ${num_fewshot} \
        --log_samples
done
