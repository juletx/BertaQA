#!/bin/bash
#SBATCH --job-name=lm_eval_llama-2-eu_eustrivia_mt
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=0
#SBATCH --mem=50GB
#SBATCH --gres=gpu:1
#SBATCH --output=.slurm/lm_eval_llama-2-eu_eustrivia_mt.out
#SBATCH --error=.slurm/lm_eval_llama-2-eu_eustrivia_mt.err

# activate virtual environment
source /gaueko0/users/jetxaniz007/phd/venv2/bin/activate
export TRANSFORMERS_CACHE="/gaueko0/transformers_cache/"
export TOKENIZERS_PARALLELISM=false

# path and model name
path="/trumoi_scratch3/aormazabal024/EU_FT/finetuned_models/"
model="7b_lr1e-4_fixed_warmup20_thepile100k/"

# load tasks
source tasks.sh

# select tasks
tasks_selected=(
    #"eustrivia_mt"
    "eustrivia_mt_itzuli"
)

num_fewshot=5

# model name from checkpoint 300 to 6600 with step 300
for i in {300..6600..300}; do
    checkpoint="checkpoint-${i}"
    model_name="${path}${model}${checkpoint}"
    for group_name in "${tasks_selected[@]}"; do
        srun python3 -m lm_eval \
            --model hf \
            --model_args pretrained=$model_name \
            --tasks $group_name \
            --device cuda \
            --output_path ../results/finetune_llama-2_raw/${model}${checkpoint}/${group_name}_${num_fewshot}-shot.json \
            --batch_size auto \
            --num_fewshot ${num_fewshot} \
            --log_samples
    done
done
