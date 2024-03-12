from anthropic import Anthropic
import os
from datasets import load_dataset
import random
import argparse
import json
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff

seed = 42
random.seed(seed)

# Set your Anthropic API key from environment variable
client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

answer2letter = {0: "A", 1: "B", 2: "C"}


def load_basquetrivia(config="eu"):
    dataset = load_dataset("HiTZ/basquetrivia", name=config, split="test")
    return dataset


def format_question(item, config="eu"):
    question = item["question"]
    candidates = item["candidates"]

    # Format the question with the given prompt
    if config == "eu":
        formatted_question = f"Galdera: {question}\nA: {candidates[0]}\nB: {candidates[1]}\nC: {candidates[2]}\nErantzuna:"
    elif config in ["en", "en_mt"]:
        formatted_question = f"Question: {question}\nA: {candidates[0]}\nB: {candidates[1]}\nC: {candidates[2]}\nAnswer:"
    else:
        raise ValueError("config must be 'eu', 'en' or 'en_mt'")
    return formatted_question


def few_shot_messages(few_shot_examples, config="eu"):
    # Add the few-shot examples to the prompt
    messages = []
    for example in few_shot_examples:
        messages.append({"role": "user", "content": format_question(example, config)})
        messages.append(
            {"role": "assistant", "content": answer2letter[example["answer"]]}
        )
    return messages


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def completion_with_backoff(**kwargs):
    return client.messages.create(**kwargs)


def anthropic_api_calculate_cost(usage, model="claude-3-sonnet-20240229"):
    pricing = {
        "claude-3-opus-20240229": {
            "prompt": 0.015,
            "completion": 0.075,
        },
        "claude-3-sonnet-20240229": {
            "prompt": 0.003,
            "completion": 0.015,
        },
    }

    try:
        model_pricing = pricing[model]
    except KeyError:
        raise ValueError("Invalid model specified")

    prompt_cost = usage.input_tokens * model_pricing["prompt"] / 1000
    completion_cost = usage.output_tokens * model_pricing["completion"] / 1000

    total_cost = prompt_cost + completion_cost
    # round to 6 decimals
    total_cost = round(total_cost, 6)

    return total_cost


def evaluate_basquetrivia(
    config="test", model="claude-3-sonnet-20240229", shots=5, limit=1, start=0
):
    # Load your dataset from Hugging Face
    print(f"Loading {config} config...")
    dataset = load_basquetrivia(config=config)

    # Create the results directory if it doesn't exist
    os.makedirs(f"../results/{model}", exist_ok=True)

    tokens = 0
    cost = 0
    
    system_prompt = "Respond always with a single letter: A, B or C."

    # Iterate over your dataset and use the API
    for i, item in enumerate(dataset):
        if i < start:
            continue

        # Get 5 random few-shot examples
        few_shot_examples = random.sample([ex for ex in dataset if ex != item], shots)

        # Add the few-shot examples to the prompt
        messages = few_shot_messages(few_shot_examples, config)

        messages.append({"role": "user", "content": format_question(item, config)})

        # Save messages along with the original dataset fields to a jsonl file
        item["system"] = system_prompt
        item["messages"] = messages

        # Use the chat models, which are better for multi-turn conversation
        completion = completion_with_backoff(
            model=model,
            max_tokens=1,
            system=system_prompt,
            messages=messages,
            temperature=0,
        )

        # convert completions to dict
        response = completion.model_dump()

        # Save whole response along with the original dataset fields to a jsonl file
        item["response"] = response

        # Check if the answer is correct
        item["correct"] = (
            response["content"][0]["text"]
            == answer2letter[item["answer"]]
        )

        # Calculate Anthropic API cost and add to the item
        item["cost"] = anthropic_api_calculate_cost(completion.usage, model)

        cost += item["cost"]
        tokens += completion.usage.input_tokens + completion.usage.output_tokens

        # Print details in a line: i, total tokens and total cost
        print(f"{i + 1}: ${cost:.4f} total cost, {tokens:,} tokens")

        with open(
            f"../results/{model}/basquetrivia_{config}_{shots}-shot.jsonl", "a"
        ) as f:
            json.dump(item, f)
            f.write("\n")

        if i == limit - 1:
            break


def main():
    # Define the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, default="eu", help="Dataset config to evaluate on"
    )
    parser.add_argument(
        "--model", type=str, default="claude-3-sonnet-20240229", help="Anthropic model to use"
    )
    parser.add_argument(
        "--shots", type=int, default=5, help="Number of few-shot examples"
    )
    parser.add_argument(
        "--limit", type=int, default=1, help="Number of examples to evaluate"
    )
    parser.add_argument(
        "--start", type=int, default=0, help="Start index of the examples to evaluate"
    )
    args = parser.parse_args()
    evaluate_basquetrivia(
        config=args.config, model=args.model, shots=args.shots, limit=args.limit
    )


if __name__ == "__main__":
    main()
