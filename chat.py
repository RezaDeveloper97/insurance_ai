import os
import sys
import time
import torch
from threading import Thread, Event
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from peft import PeftModel

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# ---------- Config ----------
BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
LORA_PATH = "insurance-lora-output"
DEBUG = True  # Set to False to disable debug logs

def log(msg):
    if DEBUG:
        print(f"[DEBUG] {msg}")

# ---------- Loading Animation ----------
class ThinkingAnimation:
    def __init__(self):
        self.stop_event = Event()
        self.thread = None

    def _animate(self):
        dots = ["   ", ".  ", ".. ", "..."]
        i = 0
        while not self.stop_event.is_set():
            sys.stdout.write(f"\rAssistant: thinking{dots[i % 4]}")
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
        sys.stdout.write("\r" + " " * 40 + "\r")
        sys.stdout.flush()


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
print("Loading model", end="", flush=True)
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float32,
    low_cpu_mem_usage=True
)
print(".", end="", flush=True)

model = PeftModel.from_pretrained(model, LORA_PATH)
print(".", end="", flush=True)

model = model.to(device)
model.eval()
print(". Ready!")

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("\n" + "="*50)
print("Insurance AI Assistant")
print("Type 'exit' to quit")
print("="*50 + "\n")

thinking = ThinkingAnimation()


def generate_streaming(question: str):
    prompt = f"""### Instruction:
You are a professional insurance advisor. Give brief, accurate answers.

### Question:
{question}

### Answer:
"""
    log(f"Prompt length: {len(prompt)} chars")

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    log(f"Input tokens: {inputs['input_ids'].shape[1]}")

    streamer = TextIteratorStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=True
    )

    generation_kwargs = {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
        "max_new_tokens": 150,
        "do_sample": False,
        "streamer": streamer,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }

    thinking.start()

    generated_text = []
    error_occurred = None

    def generate_with_error_handling():
        nonlocal error_occurred
        try:
            with torch.inference_mode():
                model.generate(**generation_kwargs)
        except Exception as e:
            error_occurred = str(e)

    thread = Thread(target=generate_with_error_handling)
    thread.start()

    first_token = True
    token_count = 0

    try:
        for text in streamer:
            if first_token:
                thinking.stop()
                print("Assistant: ", end="", flush=True)
                first_token = False

            token_count += 1
            generated_text.append(text)
            print(text, end="", flush=True)
    except Exception as e:
        log(f"Streamer error: {e}")

    thread.join()

    if first_token:  # No tokens were generated
        thinking.stop()
        print("Assistant: ", end="")

    log(f"Generated {token_count} tokens")

    if error_occurred:
        print(f"\n[ERROR] {error_occurred}")
    elif token_count == 0:
        full_text = "".join(generated_text)
        if full_text:
            print(full_text)
        else:
            print("(No response generated)")
            log("Trying non-streaming generation...")
            # Fallback: non-streaming
            with torch.inference_mode():
                outputs = model.generate(
                    inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_new_tokens=150,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id
                )
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Extract only the answer part
            if "### Answer:" in response:
                answer = response.split("### Answer:")[-1].strip()
                print(f"Assistant: {answer}")
            else:
                print(f"Assistant: {response}")

    print("\n")


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
