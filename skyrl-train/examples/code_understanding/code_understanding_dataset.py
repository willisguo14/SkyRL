"""
Preprocess the synthetic-code-understanding dataset to parquet format
"""

import argparse
import json
import os
import re
from datasets import load_dataset


def extract_ground_truth(example):
    """Extract the ground truth answer from the dataset."""
    # Parse the gold_standard_solution which is a string representation of a dict
    try:
        gold_standard = json.loads(example["gold_standard_solution"].replace("'", '"'))
        return gold_standard["output"]
    except:
        # Fallback: try parsing verification_info
        try:
            verification = json.loads(example["verification_info"].replace("'", '"'))
            return verification["ground_truth"]
        except:
            # Last resort: return None
            return None


def modify_prompt(prompt_text):
    """Modify the prompt to use GSM8k-style instruction format."""
    # Remove the original JSON instruction
    # The instruction is typically at the end: "Return your response as a json with a field 'output'..."
    instruction_pattern = r"Return your response as a json.*?\.(\n|$)"
    modified_prompt = re.sub(instruction_pattern, "", prompt_text, flags=re.DOTALL)

    # Add GSM8k-style instruction
    new_instruction = 'Let\'s think step by step and output the final answer after "####".'
    modified_prompt = modified_prompt.strip() + "\n\n" + new_instruction

    return modified_prompt


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="/data/user_data/willisg/code_understanding")
    parser.add_argument("--train_ratio", type=float, default=0.9, help="Ratio of data to use for training")

    args = parser.parse_args()

    args.output_dir = os.path.expanduser(args.output_dir)

    data_source = "PrimeIntellect/synthetic-code-understanding"

    print(f"Loading dataset from {data_source}...")
    dataset = load_dataset(data_source)

    # Get the train split and split it into train/validation
    full_train = dataset["train"]

    # Calculate split sizes
    total_size = len(full_train)
    train_size = int(total_size * args.train_ratio)

    print(f"Total examples: {total_size}")
    print(f"Train size: {train_size}, Test size: {total_size - train_size}")

    # Split the dataset
    train_test_split = full_train.train_test_split(
        train_size=train_size,
        seed=42,
        shuffle=True
    )
    train_dataset = train_test_split["train"]
    val_dataset = train_test_split["test"]

    # Process function to transform data into required format
    def make_map_fn(split):
        def process_fn(example, idx):
            # Extract ground truth
            ground_truth = extract_ground_truth(example)

            if ground_truth is None:
                print(f"Warning: Could not extract ground truth for example {idx} in {split}")
                # Use empty string as fallback
                ground_truth = ""

            # Modify prompt to use GSM8k instruction format
            original_prompt = example["prompt"]
            modified_prompt_text = modify_prompt(original_prompt)

            # Parse metadata
            try:
                metadata_dict = json.loads(example["metadata"].replace("'", '"'))
            except:
                metadata_dict = {}

            data = {
                "data_source": data_source,
                "prompt": [
                    {
                        "role": "user",
                        "content": modified_prompt_text,
                    }
                ],
                "env_class": "code_understanding",
                "reward_spec": {
                    "method": "rule",
                    "ground_truth": ground_truth,
                },
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "problem_id": example.get("problem_id", ""),
                    "source": example.get("source", ""),
                    "task_type": example.get("task_type", ""),
                    "metadata": metadata_dict,
                    "original_prompt": original_prompt,
                },
            }
            return data

        return process_fn

    print("Processing train dataset...")
    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True)

    print("Processing validation dataset...")
    val_dataset = val_dataset.map(function=make_map_fn("test"), with_indices=True)

    # Save to parquet
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    print(f"Saving datasets to {output_dir}...")
    train_dataset.to_parquet(os.path.join(output_dir, "train.parquet"))
    val_dataset.to_parquet(os.path.join(output_dir, "validation.parquet"))

    print("Done!")
    print(f"Train dataset saved with {len(train_dataset)} examples")
    print(f"Validation dataset saved with {len(val_dataset)} examples")
