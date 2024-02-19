#!/bin/bash
#SBATCH --job-name=lm_eval_llama-2-70b-hf-eus-v0_eustrivia_mt
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=0
#SBATCH --mem=100GB
#SBATCH --gres=gpu:4
#SBATCH --output=.slurm/lm_eval_llama-2-70b-hf-eus-v0_eustrivia_mt.out
#SBATCH --error=.slurm/lm_eval_llama-2-70b-hf-eus-v0_eustrivia_mt.err

# activate virtual environment
source /gaueko0/users/jetxaniz007/phd/venv2/bin/activate
export TRANSFORMERS_CACHE="/gaueko1/transformers_cache/"
export TOKENIZERS_PARALLELISM=false

path="/gaueko1/hizkuntza-ereduak/llama-2-eus/"
model="Llama-2-70b-hf-eus-v0"
checkpoint="global_step1680"
model_name=$path/$model/$checkpoint
tokenizer="meta-llama/Llama-2-7b-hf"

# select tasks
tasks_selected=(
    "eustrivia_mt_itzuli"
    "eustrivia_mt_elhuyar"
)

for group_name in "${tasks_selected[@]}"; do
    if [[ $group_name == "xnli_eu" || $group_name == "xstorycloze_eu" ]]; then
        num_fewshot=0
    else
        num_fewshot=5
    fi
    srun python3 -m lm_eval \
        --model hf \
        --model_args pretrained=$model_name,tokenizer=$tokenizer,parallelize=True \
        --tasks $group_name \
        --device cuda \
        --output_path ../results/llama-2-eus/${model}/${model}_${group_name}_${num_fewshot}-shot.json \
        --batch_size auto \
        --num_fewshot ${num_fewshot} \
        --log_samples
done
