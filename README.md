# BertaQA: How Much Do Language Models Know About Local Culture?

Large Language Models (LLMs) exhibit extensive knowledge about the world, but most evaluations have been limited to global or anglocentric subjects. This raises the question of how well these models perform on topics relevant to other cultures, whose presence on the web is not that prominent. To address this gap, we introduce BertaQA, a multiple-choice trivia dataset that is parallel in English and Basque. The dataset consists of a local subset with questions pertinent to the Basque culture, and a global subset with questions of broader interest. We find that state-of-the-art LLMs struggle with local cultural knowledge, even as they excel on global topics. However, we show that continued pre-training in Basque significantly improves the models' performance on Basque culture, even when queried in English. To our knowledge, this is the first solid evidence of knowledge transfer from a low-resource to a high-resource language. Our analysis sheds light on the complex interplay between language and knowledge, and reveals that some prior findings do not fully hold when reassessed on local topics. Our dataset and evaluation code are available under open licenses at https://github.com/juletx/BertaQA.

Dataset: https://huggingface.co/HiTZ/BertaQA

Paper: https://arxiv.org/abs/2406.07302

# Get Examples and Statistics

Run the following `examples_statistics.ipynb` jupyter notebook in `analysis` folder to get examples and statistics of the dataset:

# Open Model Evaluation

## Install Evaluation Harness

You will need to install [LM Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness). Clone the repository and install the requirements:

```bash	
git clone https://github.com/EleutherAI/lm-evaluation-harness
cd lm-evaluation-harness
pip install -e .
```

## Run Model Evaluation

To run evaluation on open models, use the scripts in the `scripts` directory. Each script evaluates a model in all the tasks. For example, to run evaluation on Latxa v1.1 7b, run:

```bash
sbatch lm_eval_latxa-7b-v1.1.slurm
```

## Check Evaluation Results

Evaluation results are in the `results` directory. Each model has a directory with the results of the evaluation in each task. The results are in the form of a json file with the average scores of the model in each task.

## Results Analysis

To analyze the results, run the `bertaqa.ipynb` jupyter notebook in the `analysis` directory. This notebook will generate the tables of the paper.

# Commercial Model Evaluation

Commercial models from OpenAI and Anthropic are evaluated using the respective APIs. The evaluation scripts are in the `openai` and `anthropic` directories. The evaluation results are in the `results` directory.

## Run OpenAI Model Evaluation

To run evaluation on OpenAI models, use the scripts in the `openai` directory. There is a python script to evaluate each dataset, and a bash script for each model and dataset. For example, to run evaluation on GPT-3.5 Turbo on EusTrivia, run:

```bash
bash gpt-3.5-turbo-0125_eus_trivia.sh
```

## Check OpenAI Evaluation Results

Evaluation results are in the `results` directory. Each model has a directory with the results of the evaluation in each task. In this case, all the outputs of the models are saved for each task. Scores can be calculated using the `correct` field. For EusTrivia and EusExams, there are additional scripts to obtained detailed results by category.

## Results Analysis

To analyze the results, run the `bertaqa_openai.ipynb` and `bertaqa_anthropic.ipynb` jupyter notebooks in the `analysis` directory. These notebooks will generate the tables of the paper.

# Translate-test and Self-translate

We use the HuggingFace Transformers to translate the datasets. Translation scripts are in the `translate` directory. There is a folder for each model with the translation scripts that were used to generate the results in the paper. The resulting translated datasets are available in HuggingFace: https://huggingface.co/HiTZ/BertaQA.

- The script `dataset.py` contains the dataset classes.
- The script `dataset_configs.py` contains the dataset configuration.
- We use the `translate_dataset_nllb.py` script to translate the datasets with NLLB. This script uses the `translate.py` script to translate each field of the dataset.
- The `translate_dataset_few_shot.py` script is used to translate the datasets with XGLM. This script uses the `translate_few_shot.py` script to translate each field of the dataset.

## Translate-test

For example, to translate the EusTrivia dataset to English using NLLB-200-3.3B, run:

```bash
sbatch translate_bertaqa_nllb.slurm
```

## Self-translate

For example, to self-translate the EusTrivia dataset using Latxa v1.1 7b, run:

```bash
sbatch translate_bertaqa_latxa-7b-v1.1.slurm
```

# Citation

```bibtex
@misc{etxaniz2024bertaqa,
      title={BertaQA: How Much Do Language Models Know About Local Culture?}, 
      author={Julen Etxaniz and Gorka Azkune and Aitor Soroa and Oier Lopez de Lacalle and Mikel Artetxe},
      year={2024},
      eprint={2406.07302},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```