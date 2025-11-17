import argparse
from tqdm import tqdm
import torch
import os
import time
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer, AutoModelForCausalLM

def prep_pissa(
    base_save_path,
    hf_path,
    rank,
):
    save_path = os.path.join(base_save_path, hf_path)

    # Check if path already exists
    if os.path.exists(save_path):
        print(f"Save path {save_path} already exists. Skipping model preparation.")
        return

    config = LoraConfig(
        init_lora_weights="pissa",
        r=rank,
        lora_alpha=32,
        lora_dropout=0,
        target_modules="all-linear",
        task_type="CAUSAL_LM",
    )

    tokenizer = AutoTokenizer.from_pretrained(hf_path)
    model = AutoModelForCausalLM.from_pretrained(
        hf_path, torch_dtype=torch.bfloat16, device_map="auto"
    )

    model = get_peft_model(model, config)

    save_model = model.unload()

    # Create directory if it doesn't exist
    os.makedirs(save_path, exist_ok=True)

    tokenizer.save_pretrained(save_path)
    save_model.save_pretrained(save_path)
    print(f"Model and tokenizer saved to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare PiSSA base model")
    parser.add_argument("--base_save_path", type=str, required=True,
                       help="Base directory to save the model")
    parser.add_argument("--rank", type=int, required=True,
                       help="LoRA rank")
    parser.add_argument("--hf_path", type=str, required=True,
                       help="HuggingFace model path")

    args = parser.parse_args()

    prep_pissa(args.base_save_path, args.hf_path, args.rank)
   