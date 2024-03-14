#!/bin/bash
#SBATCH --job-name=translate_basquetrivia_llama-2-70b-hf
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:5
#SBATCH --output=.slurm/translate_basquetrivia_llama-2-70b-hf.out
#SBATCH --error=.slurm/translate_basquetrivia_llama-2-70b-hf.err
#SBATCH --mem=64GB

# activate virtual environment
source /gaueko0/users/jetxaniz007/phd/venv2/bin/activate

model_name="meta-llama/Llama-2-70b-hf"
dataset=basquetrivia
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