from openai import OpenAI
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

# Set your OpenAI API key from environment variable
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

answer2letter = {0: "A", 1: "B", 2: "C"}


def load_bertaqa(config="eu"):
    dataset = load_dataset("HiTZ/bertaqa", name=config, split="test")
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
    messages = [
        {
            "role": "system",
            "content": "Respond always with a single letter: A, B or C.",
        }
    ]
    for example in few_shot_examples:
        messages.append({"role": "user", "content": format_question(example, config)})
        messages.append(
            {"role": "assistant", "content": answer2letter[example["answer"]]}
        )
    return messages


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def completion_with_backoff(**kwargs):
    return client.chat.completions.create(**kwargs)


def openai_api_calculate_cost(usage, model="gpt-4-0125-preview"):
    pricing = {
        'gpt-3.5-turbo-0125': {
            'prompt': 0.0005,
            'completion': 0.0015,
        },
        'gpt-4-0125-preview': {
            'prompt': 0.01,
            'completion': 0.03,
        },
        'gpt-4-0613': {
            'prompt': 0.03,
            'completion': 0.06,
        }
    }

    try:
        model_pricing = pricing[model]
    except KeyError:
        raise ValueError("Invalid model specified")

    prompt_cost = usage.prompt_tokens * model_pricing['prompt'] / 1000
    completion_cost = usage.completion_tokens * model_pricing['completion'] / 1000

    total_cost = prompt_cost + completion_cost
    # round to 6 decimals
    total_cost = round(total_cost, 6)

    return total_cost


def evaluate_bertaqa(config="test", model="gpt-3.5-turbo", shots=5, limit=1, start=0):
    # Load your dataset from Hugging Face
    print(f"Loading {config} config...")
    dataset = load_bertaqa(config=config)

    # Create the results directory if it doesn't exist
    os.makedirs(f"../results/{model}", exist_ok=True)

    tokens = 0
    cost = 0 

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
        item["messages"] = messages

        # Use the chat models, which are better for multi-turn conversation
        completion = completion_with_backoff(
            model=model,
            messages=messages,
            temperature=0,
            seed=seed,
        )

        # convert completions to dict
        response = completion.model_dump()

        # Save whole response along with the original dataset fields to a jsonl file
        item["response"] = response

        # Check if the answer is correct
        item["correct"] = (
            response["choices"][0]["message"]["content"]
            == answer2letter[item["answer"]]
        )
        
                # Calculate OpenAI API cost and add to the item
        item["cost"] = openai_api_calculate_cost(completion.usage, model)
        
        cost += item["cost"]
        tokens += completion.usage.total_tokens
        
        # Print details in a line: i, total tokens and total cost
        print(f"{i + 1}: ${cost:.4f} total cost, {tokens:,} tokens")

        with open(f"../results/{model}/bertaqa_{config}_{shots}-shot.jsonl", "a") as f:
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
        "--model", type=str, default="gpt-3.5-turbo", help="OpenAI model to use"
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
    evaluate_bertaqa(
        config=args.config, model=args.model, shots=args.shots, limit=args.limit
    )


if __name__ == "__main__":
    main()
