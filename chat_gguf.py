"""
Fast chat using llama.cpp (GGUF model)
Requires: pip install llama-cpp-python
"""
import sys
import os

# Try to import llama-cpp-python
try:
    from llama_cpp import Llama
except ImportError:
    print("""
llama-cpp-python not installed. Run:

  pip install llama-cpp-python

For Apple Silicon (Metal acceleration):
  CMAKE_ARGS="-DLLAMA_METAL=on" pip install llama-cpp-python --force-reinstall --no-cache-dir
""")
    sys.exit(1)

# ---------- Config ----------
MODEL_PATH = "insurance-model.gguf"

# Check for quantized version first
if os.path.exists("insurance-model-q4.gguf"):
    MODEL_PATH = "insurance-model-q4.gguf"
    print("Using quantized model (faster)")

if not os.path.exists(MODEL_PATH):
    print(f"""
Model not found: {MODEL_PATH}

Run these steps first:
  1. python merge_lora.py
  2. python convert_to_gguf.py
""")
    sys.exit(1)

# ---------- Load Model ----------
print(f"Loading {MODEL_PATH}...")
llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=512,        # Context window
    n_threads=4,      # CPU threads (adjust for your CPU)
    verbose=False
)
print("✓ Ready!\n")

print("="*50)
print("  Insurance AI Assistant (llama.cpp)")
print("  Type 'exit' to quit")
print("="*50 + "\n")


def generate(question: str):
    prompt = f"""### Instruction:
You are a professional insurance advisor. Give brief, accurate answers.

### Question:
{question}

### Answer:
"""

    print("Assistant: ", end="", flush=True)

    # Stream output
    for token in llm(
        prompt,
        max_tokens=150,
        stop=["###", "\n\n"],
        stream=True
    ):
        text = token["choices"][0]["text"]
        print(text, end="", flush=True)

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

        generate(user_input)

    except KeyboardInterrupt:
        print("\nGoodbye!")
        break
