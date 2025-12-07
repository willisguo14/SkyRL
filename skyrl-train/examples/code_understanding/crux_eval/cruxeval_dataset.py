"""
Preprocess the CruxEval dataset to parquet format for SkyRL training.

This script loads the CruxEval dataset and creates two tasks for each example:
1. Input prediction: predict the function input given the output
2. Output prediction: predict the function output given the input
"""

import argparse
import os
from datasets import load_dataset, Dataset, DatasetDict

INPUT_PREDICTION_PROMPT_TEMPLATE = """You are given a Python function f and an assertion containing an output to the function. Your task is to find an input such that executing f on the input leads to the given output. Execute the program step by step in [THOUGHT] and [/THOUGHT] tags before arriving at an answer, and provide the full assertion with the correct input in [ANSWER] and [/ANSWER] tags, following the examples. Do NOT output any extra information. 

[PYTHON]
def f(x):
    return x + 1
assert f(??) == 17
[/PYTHON]
[THOUGHT]
To find an input such that executing f on the input leads to the given output, we can work backwards from the given assertion. We know that f(??) == 17. 

Since the function f(x) returns x + 1, for f(??) to be equal to 17, the value of ?? should be 16. 
[/THOUGHT]
[ANSWER]
assert f(16) == 17
[/ANSWER]

[PYTHON]
{code}
{assert_statement}
[/PYTHON]
[THOUGHT]
"""


OUTPUT_PREDICTION_PROMPT_TEMPLATE = """You are given a Python function f and an assertion containing an input to the function. Your task is to find the output when executing f on the input. Execute the program step by step in [THOUGHT] and [/THOUGHT] tags before arriving at an answer, and provide the full assertion with the correct output in [ANSWER] and [/ANSWER] tags, following the examples. Do NOT output any extra information.

[PYTHON]
def f(s):
    s = s + s
    return "b" + s + "a"
assert f("hi") == ??
[/PYTHON]
[THOUGHT]
Let's execute the code step by step:

1. The function f is defined, which takes a single argument s.
2. The function is called with the argument "hi", so within the function, s is initially "hi".
3. Inside the function, s is concatenated with itself, so s becomes "hihi".
4. The function then returns a new string that starts with "b", followed by the value of s (which is now "hihi"), and ends with "a".
5. The return value of the function is therefore "bhihia".
[/THOUGHT]
[ANSWER]
assert f("hi") == "bhihia"
[/ANSWER]

[PYTHON]
{code}
{assert_statement}
[/PYTHON]
[THOUGHT]
"""


def create_input_prediction_example(example, idx, split):
    """Create an input prediction task from a CruxEval example."""
    code = example["code"]
    input_val = example["input"]
    output_val = example["output"]
    example_id = example["id"]

    # Create assert statement with ?? for input
    assert_statement = f"assert f(??) == {output_val}"

    # Create the prompt
    prompt_text = INPUT_PREDICTION_PROMPT_TEMPLATE.format(
        code=code,
        assert_statement=assert_statement
    )

    # Ground truth is the full assertion with correct input
    ground_truth = f"assert f({input_val}) == {output_val}"

    data = {
        "data_source": "cruxeval-org/cruxeval",
        "prompt": [
            {
                "role": "user",
                "content": prompt_text,
            }
        ],
        "env_class": "cruxeval",
        "reward_spec": {
            "method": "rule",
            "ground_truth": ground_truth,
        },
        "extra_info": {
            "task_type": "input",
            "split": split,
            "index": idx,
            "original_id": example_id,
            "code": code,
            "input": input_val,
            "output": output_val,
        },
    }
    return data


def create_output_prediction_example(example, idx, split):
    """Create an output prediction task from a CruxEval example."""
    code = example["code"]
    input_val = example["input"]
    output_val = example["output"]
    example_id = example["id"]

    # Create assert statement with ?? for output
    assert_statement = f"assert f({input_val}) == ??"

    # Create the prompt
    prompt_text = OUTPUT_PREDICTION_PROMPT_TEMPLATE.format(
        code=code,
        assert_statement=assert_statement
    )

    # Ground truth is the full assertion with correct output
    ground_truth = f"assert f({input_val}) == {output_val}"

    data = {
        "data_source": "cruxeval-org/cruxeval",
        "prompt": [
            {
                "role": "user",
                "content": prompt_text,
            }
        ],
        "env_class": "cruxeval",
        "reward_spec": {
            "method": "rule",
            "ground_truth": ground_truth,
        },
        "extra_info": {
            "task_type": "output",
            "split": split,
            "index": idx,
            "original_id": example_id,
            "code": code,
            "input": input_val,
            "output": output_val,
        },
    }
    return data


def process_dataset(dataset_split, split_name, train_ratio=0.8):
    """
    Process the CruxEval dataset split into train and validation sets.
    Each example is converted into two tasks: input prediction and output prediction.

    Args:
        dataset_split: The dataset split to process
        split_name: Name of the split (e.g., 'train', 'test')
        train_ratio: Ratio of examples to use for training (default 0.8)

    Returns:
        Tuple of (train_dataset, val_dataset)
    """
    all_examples = []

    # Process each example and create both input and output prediction tasks
    for idx, example in enumerate(dataset_split):
        # Create input prediction example
        input_pred_example = create_input_prediction_example(
            example, idx * 2, split_name
        )
        all_examples.append(input_pred_example)

        # Create output prediction example
        output_pred_example = create_output_prediction_example(
            example, idx * 2 + 1, split_name
        )
        all_examples.append(output_pred_example)

    # Split into train and validation sets
    total_examples = len(all_examples)
    train_size = int(total_examples * train_ratio)

    train_examples = all_examples[:train_size]
    val_examples = all_examples[train_size:]

    train_dataset = Dataset.from_list(train_examples)
    val_dataset = Dataset.from_list(val_examples)

    return train_dataset, val_dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="/data/user_data/willisg/cruxeval",
                        help="Directory to save the processed dataset")
    parser.add_argument("--train_ratio", type=float, default=0.9,
                        help="Ratio of examples to use for training (default: 0.8)")

    args = parser.parse_args()

    args.output_dir = os.path.expanduser(args.output_dir)

    # Load the CruxEval dataset
    print("Loading CruxEval dataset...")
    dataset = load_dataset("cruxeval-org/cruxeval")

    # The dataset only has a 'test' split, so we'll use that and split it ourselves
    test_dataset = dataset["test"]
    print(f"Loaded {len(test_dataset)} examples from the test split")

    # Process the dataset and split into train/validation
    print(f"Processing dataset with train_ratio={args.train_ratio}...")
    train_dataset, val_dataset = process_dataset(
        test_dataset, "train", train_ratio=args.train_ratio
    )

    print(f"Created {len(train_dataset)} training examples")
    print(f"Created {len(val_dataset)} validation examples")

    # Save datasets to parquet format
    os.makedirs(args.output_dir, exist_ok=True)
    train_output_path = os.path.join(args.output_dir, "train.parquet")
    val_output_path = os.path.join(args.output_dir, "validation.parquet")

    train_dataset.to_parquet(train_output_path)
    val_dataset.to_parquet(val_output_path)

    print(f"\nDataset saved to {args.output_dir}")
    print(f"  - Training data: {train_output_path}")
    print(f"  - Validation data: {val_output_path}")

    # Print example
    print("\n" + "="*80)
    print("Example training sample (Input Prediction):")
    print("="*80)
    example = train_dataset[0]
    print(f"Task type: {example['extra_info']['task_type']}")
    print(f"\nPrompt:\n{example['prompt'][0]['content']}")
    print(f"\nGround truth: {example['reward_spec']['ground_truth']}")

    print("\n" + "="*80)
    print("Example training sample (Output Prediction):")
    print("="*80)
    example = train_dataset[1]
    print(f"Task type: {example['extra_info']['task_type']}")
    print(f"\nPrompt:\n{example['prompt'][0]['content']}")
    print(f"\nGround truth: {example['reward_spec']['ground_truth']}")
