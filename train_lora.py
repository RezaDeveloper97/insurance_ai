import os
import json
import torch
from datasets import Dataset

# Use HuggingFace mirror for faster downloads (useful in regions with slow HF access)
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType


# ---------- Config for 8GB RAM ----------
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DATASET_PATH = "insurance_lora.json"
OUTPUT_DIR = "insurance-lora-output"
MAX_LENGTH = 256  # Reduced for memory


# ---------- Device setup for Apple Silicon ----------
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using Apple MPS (Metal Performance Shaders)")
else:
    device = torch.device("cpu")
    print("Using CPU")


# ---------- Load dataset ----------
with open(DATASET_PATH, "r", encoding="utf-8") as f:
    raw_data = json.load(f)

def format_example(example):
    prompt = (
        f"### Instruction:\n{example['instruction']}\n\n"
        f"### Question:\n{example['input']}\n\n"
        f"### Answer:\n{example['output']}"
    )
    return {"text": prompt}

dataset = Dataset.from_list(raw_data)
dataset = dataset.map(format_example)
print(f"Dataset size: {len(dataset)} examples")


# ---------- Tokenizer ----------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

def tokenize(example):
    return tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH
    )

tokenized_dataset = dataset.map(tokenize, remove_columns=dataset.column_names)


# ---------- Model ----------
print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float32
)

# Enable gradient checkpointing to save memory
model.gradient_checkpointing_enable()

model = model.to(device)
print("Model loaded!")


# ---------- LoRA config (minimal for 8GB RAM) ----------
lora_config = LoraConfig(
    r=4,  # Reduced from 8
    lora_alpha=8,
    target_modules=["q_proj", "v_proj"],  # Only 2 modules
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()


# ---------- Training ----------
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    learning_rate=2e-4,
    logging_steps=1,
    save_strategy="epoch",
    report_to="none",
    fp16=False,
    optim="adamw_torch",
    dataloader_pin_memory=False,  # Disable for MPS
    gradient_checkpointing=True
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator
)

print("Starting training...")
trainer.train()

model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("LoRA training finished! Output saved to:", OUTPUT_DIR)
