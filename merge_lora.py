"""
Merge LoRA adapter with base model and prepare for GGUF conversion
"""
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
LORA_PATH = "insurance-lora-output"
MERGED_OUTPUT = "insurance-model-merged"

print("Step 1: Loading base model...")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float32,
    low_cpu_mem_usage=True
)

print("Step 2: Loading LoRA adapter...")
model = PeftModel.from_pretrained(base_model, LORA_PATH)

print("Step 3: Merging LoRA with base model...")
model = model.merge_and_unload()

print("Step 4: Saving merged model...")
model.save_pretrained(MERGED_OUTPUT)

print("Step 5: Saving tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
tokenizer.save_pretrained(MERGED_OUTPUT)

print(f"""
✓ Done! Merged model saved to: {MERGED_OUTPUT}/

Next step - Convert to GGUF:
  python convert_to_gguf.py
""")
