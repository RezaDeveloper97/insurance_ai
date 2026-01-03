import json
import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType


MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"
DATASET_PATH = "insurance_lora.json"
OUTPUT_DIR = "insurance-lora-output"


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


# ---------- Tokenizer ----------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

def tokenize(example):
    return tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=512
    )

tokenized_dataset = dataset.map(tokenize, remove_columns=dataset.column_names)


# ---------- Model ----------
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    torch_dtype=torch.float16
)

# ---------- LoRA config ----------
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
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
    fp16=True,
    logging_steps=1,
    save_strategy="epoch",
    report_to="none"
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

trainer.train()

model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("LoRA training finished.")
