#!/bin/bash
#SBATCH --job-name=lm_eval_qwen-2-7b-instruct
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=0
#SBATCH --mem=50GB
#SBATCH --gres=gpu:1
#SBATCH --output=.slurm/lm_eval_qwen-2-7b-instruct.out
#SBATCH --error=.slurm/lm_eval_qwen-2-7b-instruct.err

# activate virtual environment
source /gaueko0/users/jetxaniz007/phd/venv2/bin/activate

path="Qwen"
model="Qwen2-7B-Instruct"
model_name=$path/$model
folder="qwen"


# select tasks
tasks_selected=(
    "bertaqa_eu"
    "bertaqa_en"
)

num_fewshot=5

for group_name in "${tasks_selected[@]}"; do
    srun python3 -m lm_eval \
        --model hf \
        --model_args pretrained=$model_name,attn_implementation=flash_attention_2 \
        --tasks $group_name \
        --device cuda \
        --output_path ../results/${folder}/${model}/${group_name}_${num_fewshot}-shot.json \
        --batch_size auto \
        --num_fewshot ${num_fewshot} \
        --log_samples
done
