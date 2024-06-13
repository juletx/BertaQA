#!/bin/bash
#SBATCH --job-name=translate_bertaqa_latxa-70b-v1
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:5
#SBATCH --output=.slurm/translate_bertaqa_latxa-70b-v1.out
#SBATCH --error=.slurm/translate_bertaqa_latxa-70b-v1.err
#SBATCH --mem=64GB

# activate virtual environment
source /gaueko0/users/jetxaniz007/phd/venv2/bin/activate

model_name="HiTZ/latxa-70b-v1"
dataset=bertaqa
max_new_tokens=64

srun accelerate launch --main_process_port 29501 translate_dataset_few_shot.py \
    --dataset $dataset \
    --target_lang "eng_Latn" \
    --starting_batch_size 128 \
    --model_name $model_name \
    --max_length 1024 \
    --max_new_tokens $max_new_tokens \
    --num_beams 1 \
    --num_return_sequences 1 \
    --precision bf16 \
    --eos_token "\n"