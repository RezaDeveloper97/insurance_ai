import os
import sys
import time
import torch
from threading import Thread, Event
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# ---------- Config ----------
BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
LORA_PATH = "insurance-lora-output"

# ---------- Loading Animation ----------
class ThinkingAnimation:
    def __init__(self):
        self.stop_event = Event()
        self.thread = None

    def _animate(self):
        dots = ["   ", ".  ", ".. ", "..."]
        i = 0
        while not self.stop_event.is_set():
            sys.stdout.write(f"\r🤔 Thinking{dots[i % 4]}")
            sys.stdout.flush()
            i += 1
            time.sleep(0.3)

    def start(self):
        self.stop_event.clear()
        self.thread = Thread(target=self._animate)
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        if self.thread:
            self.thread.join()
        sys.stdout.write("\r" + " " * 20 + "\r")
        sys.stdout.flush()


# ---------- Device ----------
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("✓ Using Apple MPS")
else:
    device = torch.device("cpu")
    print("✓ Using CPU")

# ---------- Load Model ----------
print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float32,
    low_cpu_mem_usage=True
)

model = PeftModel.from_pretrained(model, LORA_PATH)
model = model.to(device)
model.eval()
print("✓ Model ready!")

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("\n" + "="*50)
print("  Insurance AI Assistant")
print("  Type 'exit' to quit")
print("="*50 + "\n")

thinking = ThinkingAnimation()


def generate(question: str):
    prompt = f"""### Instruction:
You are a professional insurance advisor. Give brief, accurate answers.

### Question:
{question}

### Answer:
"""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    thinking.start()

    with torch.inference_mode():
        outputs = model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=150,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    thinking.stop()

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract answer
    if "### Answer:" in response:
        answer = response.split("### Answer:")[-1].strip()
    else:
        answer = response

    # Print with typing effect
    print("Assistant: ", end="", flush=True)
    for char in answer:
        print(char, end="", flush=True)
        time.sleep(0.01)  # Small delay for typing effect
    print("\n")


# ---------- Interactive Loop ----------
while True:
    try:
        user_input = input("You: ").strip()

        if user_input.lower() in ["exit", "quit", "q"]:
            print("Goodbye! 👋")
            break

        if not user_input:
            continue

        generate(user_input)

    except KeyboardInterrupt:
        print("\nGoodbye! 👋")
        break
