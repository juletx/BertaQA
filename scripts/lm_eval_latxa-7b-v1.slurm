#!/bin/bash
#SBATCH --job-name=lm_eval_latxa-7b-v1
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=0
#SBATCH --mem=100GB
#SBATCH --gres=gpu:1
#SBATCH --output=.slurm/lm_eval_latxa-7b-v1.out
#SBATCH --error=.slurm/lm_eval_latxa-7b-v1.err

# activate virtual environment
source /gaueko0/users/jetxaniz007/phd/venv2/bin/activate

path="HiTZ"
model="latxa-7b-v1"
model_name=$path/$model
folder="latxa"

# select tasks
tasks_selected=(
    #"bertaqa_eu"
    #"bertaqa_en"
    "bertaqa_en_mt_nllb"
    "bertaqa_en_mt_madlad"
    "bertaqa_en_mt_hitz"
    "bertaqa_en_mt_itzuli"
    "bertaqa_en_mt_latxa-7b-v1"
)


num_fewshot=5

for group_name in "${tasks_selected[@]}"; do
    srun python3 -m lm_eval \
        --model hf \
        --model_args pretrained=$model_name \
        --tasks $group_name \
        --device cuda \
        --output_path ../results/${folder}/${model}/${group_name}_${num_fewshot}-shot.json \
        --batch_size auto \
        --num_fewshot ${num_fewshot} \
        --log_samples
done
