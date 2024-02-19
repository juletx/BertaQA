#!/bin/bash
#SBATCH --job-name=translate_eustrivia_nllb
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --output=.slurm/translate_eustrivia_nllb.out
#SBATCH --error=.slurm/translate_eustrivia_nllb.err
#SBATCH --mem=64GB

# activate virtual environment
source ../venv2/bin/activate

model_names=(
    "facebook/nllb-200-3.3B"
)

for model_name in "${model_names[@]}"; do
    srun accelerate launch --mixed_precision fp16 translate_dataset_nllb.py \
        --dataset eustrivia \
        --target_lang "eng_Latn" \
        --starting_batch_size 1536 \
        --model_name $model_name \
        --max_length 64 \
        --num_beams 1 \
        --num_return_sequences 1 \
        --precision fp16
done
