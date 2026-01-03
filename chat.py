import os
import torch
from threading import Thread
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from peft import PeftModel

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# ---------- Config ----------
BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
LORA_PATH = "insurance-lora-output"

# ---------- Device ----------
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using Apple MPS")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using CUDA")
else:
    device = torch.device("cpu")
    print("Using CPU")

# ---------- Load Model ----------
print("Loading base model...")
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float32
)

print("Loading LoRA adapter...")
model = PeftModel.from_pretrained(model, LORA_PATH)
model = model.to(device)
model.eval()

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("\n" + "="*50)
print("Insurance AI Assistant Ready!")
print("Type 'exit' to quit")
print("="*50 + "\n")


def generate_streaming(question: str):
    prompt = f"""### Instruction:
You are a professional insurance advisor. Provide accurate, conservative advice. Never speculate or give exact numbers without policy verification.

### Question:
{question}

### Answer:
"""

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    streamer = TextIteratorStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=True
    )

    generation_kwargs = {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
        "max_new_tokens": 256,
        "temperature": 0.7,
        "top_p": 0.9,
        "do_sample": True,
        "streamer": streamer,
        "pad_token_id": tokenizer.pad_token_id
    }

    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    print("\nAssistant: ", end="", flush=True)
    for text in streamer:
        print(text, end="", flush=True)
    print("\n")

    thread.join()


# ---------- Interactive Loop ----------
while True:
    try:
        user_input = input("You: ").strip()

        if user_input.lower() in ["exit", "quit", "q"]:
            print("Goodbye!")
            break

        if not user_input:
            continue

        generate_streaming(user_input)

    except KeyboardInterrupt:
        print("\nGoodbye!")
        break
