#!/bin/bash
#SBATCH --job-name=lm_eval_mgpt-13b
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=0
#SBATCH --mem=50GB
#SBATCH --gres=gpu:1
#SBATCH --output=.slurm/lm_eval_mgpt-13b.out
#SBATCH --error=.slurm/lm_eval_mgpt-13b.err

# activate virtual environment
source /gaueko0/users/jetxaniz007/phd/venv2/bin/activate

path="ai-forever"
model="mGPT-13B"
model_name=$path/$model
folder="mgpt"


# select tasks
tasks_selected=(
    "eustrivia"
    "eustrivia_mt"
    "eustrivia_mt_itzuli"
    "eustrivia_mt_elhuyar"
)

num_fewshot=5

for group_name in "${tasks_selected[@]}"; do
    srun python3 -m lm_eval \
        --model hf \
        --model_args pretrained=$model_name \
        --tasks $group_name \
        --device cuda \
        --output_path ../results/${folder}/${model}/${model}_${group_name}_${num_fewshot}-shot.json \
        --batch_size auto \
        --num_fewshot ${num_fewshot} \
        --log_samples
done
