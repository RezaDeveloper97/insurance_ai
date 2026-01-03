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


# ---------- Config ----------
# Options for 8GB RAM:
#   - "Qwen/Qwen2-1.5B-Instruct" (recommended, best quality)
#   - "TinyLlama/TinyLlama-1.1B-Chat-v1.0" (smaller, faster)
#   - "microsoft/phi-2" (good quality, 2.7B)

MODEL_NAME = "Qwen/Qwen2-1.5B-Instruct"
DATASET_PATH = "insurance_lora.json"
OUTPUT_DIR = "insurance-lora-output"


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


# ---------- Tokenizer ----------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
if tokenizer.pad_token is None:
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
    torch_dtype=torch.float32,  # MPS works better with float32
    trust_remote_code=True
)
model = model.to(device)


# ---------- LoRA config ----------
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
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
    use_mps_device=True if device.type == "mps" else False,
    fp16=False,  # MPS doesn't support fp16 training well
    optim="adamw_torch"
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
