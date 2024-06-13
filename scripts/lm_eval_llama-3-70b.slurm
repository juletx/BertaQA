#!/bin/bash
#SBATCH --job-name=lm_eval_llama-3-70b
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=0
#SBATCH --mem=100GB
#SBATCH --gres=gpu:3
#SBATCH --output=.slurm/lm_eval_llama-3-70b.out
#SBATCH --error=.slurm/lm_eval_llama-3-70b.err

# activate virtual environment
source /gaueko0/users/jetxaniz007/phd/venv2/bin/activate

path="meta-llama"
model="Meta-Llama-3-70B"
model_name=$path/$model
folder="llama-3"

# select tasks
tasks_selected=(
    "bertaqa_eu"
    #"bertaqa_en"
)

num_fewshot=5

for group_name in "${tasks_selected[@]}"; do
    srun python3 -m lm_eval \
        --model hf \
        --model_args pretrained=$model_name,parallelize=True,attn_implementation=flash_attention_2 \
        --tasks $group_name \
        --device cuda \
        --output_path ../results/${folder}/${model}/${group_name}_${num_fewshot}-shot.json \
        --batch_size auto \
        --num_fewshot ${num_fewshot} \
        --log_samples
done
