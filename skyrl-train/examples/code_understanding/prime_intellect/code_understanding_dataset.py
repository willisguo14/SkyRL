import argparse
import ast
import json
import os
import re
from datasets import load_dataset


def filter(example):
    if any("####" in example[k] for k in ["prompt", "gold_standard_solution"]):
        return False
    
    metadata = ast.literal_eval(example["metadata"])
    if metadata["complexify_iteration"] > 3:
        return False
    
    return True

def modify_prompt(prompt):
    old = """Return your response as a json with a field 'output' that contains the predicted output string."""
    new = """Let's think step by step and output a json with a field 'output' that contains the predicted output string after "####"."""
    assert prompt.strip().endswith(old)
    return prompt[:-len(old)] + new + "\n"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="/data/user_data/willisg/code_understanding")
    parser.add_argument("--test_size", type=float, default=0.1)
    
    args = parser.parse_args()
    args.output_dir = os.path.expanduser(args.output_dir)
    
    dataset_name = "PrimeIntellect/synthetic-code-understanding"
    dataset_dict = load_dataset(dataset_name)
        
    dataset_dict = dataset_dict["train"].filter(filter).train_test_split(test_size=args.test_size)
    
    train_dataset, test_dataset = dataset_dict["train"], dataset_dict["test"]
    
    def make_map_fn(split):
        def process_fn(example, idx):
            prompt = modify_prompt(example["prompt"])
            
            data = {
                "data_source": "dataset_name",
                "prompt": [{"role": "user", "content": prompt}],
                "env_class": "code_understanding",
                "reward_spec": {
                    "method": "rule",
                    "ground_truth": example["gold_standard_solution"]
                },
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "problem_id": example.get("problem_id", ""),
                    "metadata": example.get("metadata", "")
                }
            }
            
            return data
        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True)
    val_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True)
    
    print("train", len(train_dataset))
    print("val", len(val_dataset))
    
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    train_dataset.to_parquet(os.path.join(output_dir, "train.parquet"))
    val_dataset.to_parquet(os.path.join(output_dir, "validation.parquet"))