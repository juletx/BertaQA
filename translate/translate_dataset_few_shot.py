from collections import defaultdict
from datasets import load_dataset, DatasetDict
import translate_few_shot
import os
import argparse
import pandas as pd
from dataset_configs import dataset_configs
from typing import Dict, Any, List
import json


def get_dataset(dataset_args: Dict[str, Any]) -> DatasetDict:
    """
    Load the dataset specified in dataset_args and return a DatasetDict object.

    Args:
    - dataset_args: A dictionary containing the dataset name, dataset configurations, dataset split.

    Returns:
    - A DatasetDict object containing the loaded dataset.
    """
    dataset = DatasetDict()
    for config in dataset_args["dataset_configs"]:
        dataset[config] = load_dataset(
            dataset_args["dataset"], config, split=dataset_args["dataset_split"]
        )
    return dataset


def get_texts(
    dataset: DatasetDict, dataset_args: Dict[str, Any]
) -> Dict[str, Dict[str, Any]]:
    """
    Extract the texts from the dataset and return a dictionary containing the texts.

    Args:
    - dataset: A DatasetDict object containing the loaded dataset.
    - dataset_args: A dictionary containing the dataset configurations.

    Returns:
    - A dictionary containing the texts extracted from the dataset.
    """
    texts = defaultdict(dict)
    flatten = []
    for config in dataset_args["dataset_configs"]:
        for field in dataset_args["dataset_fields"]:
            texts[config][field] = dataset[config][field]
            # if field is a list of strings, we flatten it
            if isinstance(texts[config][field][0], list):
                flatten = [len(sentences) for sentences in texts[config][field]]
                texts[config][field] = [
                    sentence
                    for sentences in texts[config][field]
                    for sentence in sentences
                ]
    return texts, flatten


def get_few_shot_dataset(dataset_args: Dict[str, Any]) -> DatasetDict:
    """
    Load the few-shot dataset specified in dataset_args and return a DatasetDict object.

    Args:
    - dataset_args: A dictionary containing the few-shot dataset configurations.

    Returns:
    - A DatasetDict object containing the loaded few-shot dataset.
    """
    dataset = DatasetDict()
    dataset["en"] = load_dataset("facebook/flores", "eng_Latn", split="dev")
    for config in dataset_args["dataset_configs"]:
        dataset[config] = load_dataset(
            "facebook/flores", dataset_args["lang_codes"][config], split="dev"
        )
    return dataset


def get_few_shot_prompts(
    dataset: DatasetDict,
    dataset_args: Dict[str, Any],
    translate_args: Dict[str, Any],
    shots: int,
) -> Dict[str, str]:
    """
    Generate few-shot prompts for each language in dataset_args and return a dictionary containing the prompts.

    Args:
    - dataset: A DatasetDict object containing the few-shot dataset.
    - dataset_args: A dictionary containing the dataset configurations.
    - translate_args: A dictionary containing the translation configurations.
    - shots: An integer representing the number of few-shot prompts to generate.

    Returns:
    - A dictionary containing the few-shot prompts for each language.
    """
    prompts = {}
    for config in dataset_args["dataset_configs"]:
        prompts[config] = ""
        i, shot = 0, 0
        while shot < shots:
            if len(dataset[config][i]["sentence"]) < 100:
                prompts[
                    config
                ] += f'{dataset_args["lang_names"][config]}: {dataset[config][i]["sentence"]}{translate_args["eos_token"]}'
                prompts[
                    config
                ] += f'English: {dataset["en"][i]["sentence"]}{translate_args["eos_token"]}'
                shot += 1
            i += 1
        prompts[config] += f'{dataset_args["lang_names"][config]}:'
    return prompts


def text_with_prompt(text: str, prompt: str, translate_args: Dict[str, Any]) -> str:
    """
    Concatenate the text with the prompt and the eos_token.

    Args:
    - text: A string representing the text to be concatenated.
    - prompt: A string representing the prompt to be concatenated.
    - translate_args: A dictionary containing the translation configurations.

    Returns:
    - A string representing the concatenated text with the prompt and the eos_token.
    """
    return f"{prompt} {text}{translate_args['eos_token']}English:"


def map_texts_with_prompts(
    texts: Dict[str, Dict[str, List[str]]],
    prompts: Dict[str, str],
    translate_args: Dict[str, Any],
) -> Dict[str, Dict[str, List[str]]]:
    """
    Map the texts with the prompts.

    Args:
    - texts: A dictionary containing the texts to be mapped.
    - prompts: A dictionary containing the prompts to be mapped.
    - translate_args: A dictionary containing the translation configurations.

    Returns:
    - A dictionary containing the mapped texts with the prompts.
    """
    texts_with_prompts = defaultdict(dict)
    for config in texts:
        for field in dataset_args["dataset_fields"]:
            texts_with_prompts[config][field] = [
                text_with_prompt(
                    text, prompt=prompts[config], translate_args=translate_args
                )
                for text in texts[config][field]
            ]
    return texts_with_prompts


def extract_translations(
    translations: List[str], texts: List[str], translate_args: Dict[str, Any]
) -> List[str]:
    """
    Extract the translation from the output of the translation model.

    Args:
    - translations: A list containing the translations to be extracted.
    - texts: A list containing the texts to be extracted.
    - translate_args: A dictionary containing the translation configurations.

    Returns:
    - A list containing the extracted translations.
    """
    for i, text in enumerate(texts):
        if "xglm" in translate_args["model_name"]:
            translations[i] = translations[i].split("English:")[-1].strip()
        elif "bloom" in translate_args["model_name"]:
            translations[i] = translations[i][len(text) :].split(r"\n")[0].strip()
        elif "llama" in translate_args["model_name"].lower():
            translations[i] = translations[i][len(text) :].split(r"\n")[0].strip()
            if ": " in translations[i]:
                translations[i] = translations[i].split(": ")[1]
        elif "RedPajama" in translate_args["model_name"]:
            translations[i] = translations[i][len(text) :].split(r"\n")[0].strip()
        else:
            translations[i] = translations[i][len(text) :].split(r"\n")[0].strip()
    return translations


def translate_texts(
    dataset: DatasetDict,
    texts: Dict[str, Dict[str, List[str]]],
    translate_args: Dict[str, Any],
    dataset_args: Dict[str, Any],
    flatten: List[int],
) -> None:
    """
    Translate the texts.

    Args:
    - dataset: A DatasetDict object containing the dataset.
    - texts: A dictionary containing the texts to be translated.
    - translate_args: A dictionary containing the translation configurations.
    - dataset_args: A dictionary containing the dataset configurations.

    Returns:
    - None
    """
    translations = {}
    for config in dataset_args["dataset_configs"]:
        translations[config] = dataset[config].to_dict()
        translate_args["source_lang"] = dataset_args["lang_codes"][config]
        print(f"Translating from {config}")
        for field in dataset_args["dataset_fields"]:
            translations[config][field] = translate_few_shot.main(
                sentences_list=texts[config][field],
                return_output=True,
                **translate_args,
            )
            translations[config][field] = extract_translations(
                translations[config][field], texts[config][field], translate_args
            )
            # if field is a list of strings, we unflatten it with the same length as the original
            if field == "candidates":
                unflattened = []
                start = 0
                for val in flatten:
                    unflattened.append(translations[config][field][start : start + val])
                    start += val
                translations[config][field] = unflattened
        save_file(translations[config], config, translate_args, dataset_args)


def load_jsonl(input_path):
    data = []
    with open(input_path, 'r') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

def save_jsonl(data, output_path):
    with open(output_path, 'w') as f:
        for item in data:
            f.write(json.dumps(item))
            f.write('\n')

def save_file(
    translations: Dict[str, List[str]],
    config: str,
    translate_args: Dict[str, Any],
    dataset_args: Dict[str, Any],
) -> None:
    """
    Save the translations to a file.

    Args:
    - translations: A dictionary containing the translations to be saved.
    - config: A string representing the configuration.
    - translate_args: A dictionary containing the translation configurations.
    - dataset_args: A dictionary containing the dataset configurations.

    Returns:
    - None
    """
    name = translate_args["model_name"].split("/")[-1]
    dirname = f"{dataset_args['file_path']}/{name}"
    # create directory if it does not exist
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    translated_df = pd.DataFrame(translations)
    filename = f"{dirname}/{dataset_args['filename'].format(config=config)}"
    if filename.endswith(".tsv"):
        translated_df.to_csv(filename, sep="\t", index=False)
    elif filename.endswith(".jsonl"):
        translated_df.to_json(filename, orient="records", lines=True)
        # load and save as jsonl to fix spacing
        data = load_jsonl(filename)
        save_jsonl(data, filename)
    else:
        raise ValueError("Unknown file format")


def main(translate_args: Dict[str, Any], dataset_args: Dict[str, Any]) -> None:
    """
    Main function to translate the dataset.

    Args:
    - translate_args: A dictionary containing the translation configurations.
    - dataset_args: A dictionary containing the dataset configurations.

    Returns:
    - None
    """
    dataset = get_dataset(dataset_args)
    texts, flatten = get_texts(dataset, dataset_args)
    few_shot_dataset = get_few_shot_dataset(dataset_args)
    prompts = get_few_shot_prompts(
        few_shot_dataset, dataset_args, translate_args, shots=4
    )
    texts_with_prompts = map_texts_with_prompts(
        texts, prompts, translate_args=translate_args
    )
    # print(texts_with_prompts)
    translate_texts(dataset, texts_with_prompts, translate_args, dataset_args, flatten)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the translation of a dataset or dict"
    )

    parser.add_argument(
        "--dataset",
        type=str,
        help="Dataset to translate, dump file path os huggingface identifier is supported",
    )

    parser.add_argument(
        "--target_lang",
        type=str,
        required=True,
        help="Target language id. See: supported_languages.md",
    )

    parser.add_argument(
        "--starting_batch_size",
        type=int,
        default=128,
        help="Starting batch size, we will automatically reduce it if we find an OOM error."
        "If you use multiple devices, we will divide this number by the number of devices.",
    )

    parser.add_argument(
        "--model_name",
        type=str,
        default="facebook/m2m100_1.2B",
        help="Path to the model to use. See: https://huggingface.co/models",
    )

    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="Cache directory from which to load the model, or None to not cache",
    )

    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help="Maximum number of tokens in the source sentence and generated sentence. "
        "Increase this value to translate longer sentences, at the cost of increasing memory usage.",
    )

    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=128,
        help="Maximum number of tokens in the source sentence and generated sentence. "
        "Increase this value to translate longer sentences, at the cost of increasing memory usage.",
    )

    parser.add_argument(
        "--num_beams",
        type=int,
        default=5,
        help="Number of beams for beam search, m2m10 author recommends 5, but it might use too much memory",
    )

    parser.add_argument(
        "--num_return_sequences",
        type=int,
        default=1,
        help="Number of possible translation to return for each sentence (num_return_sequences<=num_beams).",
    )

    parser.add_argument(
        "--precision",
        type=str,
        default="32",
        choices=["bf16", "fp16", "32"],
        help="Precision of the model. bf16, fp16 or 32.",
    )

    parser.add_argument(
        "--do_sample",
        action="store_true",
        help="Use sampling instead of beam search.",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Temperature for sampling, value used only if do_sample is True.",
    )

    parser.add_argument(
        "--top_k",
        type=int,
        default=100,
        help="If do_sample is True, will sample from the top k most likely tokens.",
    )

    parser.add_argument(
        "--top_p",
        type=float,
        default=0.75,
        help="If do_sample is True, will sample from the top k most likely tokens.",
    )

    parser.add_argument(
        "--keep_special_tokens",
        action="store_true",
        help="Keep special tokens in the decoded text.",
    )

    parser.add_argument(
        "--eos_token",
        type=str,
        default="</s>",
        help="End of sentence token.",
    )

    args = parser.parse_args()

    translate_args = dict(
        target_lang=args.target_lang,
        starting_batch_size=args.starting_batch_size,
        model_name=args.model_name,
        cache_dir=args.cache_dir,
        max_length=args.max_length,
        max_new_tokens=args.max_new_tokens,
        num_beams=args.num_beams,
        num_return_sequences=args.num_return_sequences,
        precision=args.precision,
        do_sample=args.do_sample,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        keep_special_tokens=args.keep_special_tokens,
        eos_token=args.eos_token,
    )

    dataset_args = dataset_configs[args.dataset]

    main(translate_args, dataset_args)
