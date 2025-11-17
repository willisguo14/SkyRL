
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
    
    save_path = os.path.join(base_save_path, hf_path)
    
    tokenizer.save_pretrained(save_path)
    save_model.save_pretrained(save_path)
    
    


if __name__ == "__main__":
    base_save_path = "/data/user_data/willisg/pissa"
    rank = 32
    hf_path = "Qwen/Qwen2.5-0.5B-Instruct"
    
    prep_pissa(base_save_path, hf_path, rank)
   