"""
Convert merged model to GGUF format for llama.cpp
Requires: pip install llama-cpp-python
"""
import subprocess
import sys
import os

MERGED_MODEL = "insurance-model-merged"
OUTPUT_GGUF = "insurance-model.gguf"

# Check if llama.cpp converter exists
LLAMA_CPP_PATH = os.path.expanduser("~/llama.cpp")

if not os.path.exists(LLAMA_CPP_PATH):
    print("""
llama.cpp not found. Installing...

Run these commands:

1. Clone llama.cpp:
   cd ~
   git clone https://github.com/ggerganov/llama.cpp
   cd llama.cpp

2. Install Python requirements:
   pip install -r requirements.txt

3. Build (optional, for faster inference):
   make

4. Then run this script again:
   python convert_to_gguf.py
""")
    sys.exit(1)

# Convert to GGUF
convert_script = os.path.join(LLAMA_CPP_PATH, "convert_hf_to_gguf.py")

if not os.path.exists(convert_script):
    print(f"Error: {convert_script} not found")
    sys.exit(1)

print("Converting to GGUF format...")
cmd = [
    sys.executable,
    convert_script,
    MERGED_MODEL,
    "--outfile", OUTPUT_GGUF,
    "--outtype", "f16"
]

subprocess.run(cmd, check=True)

print(f"""
✓ Done! GGUF model saved to: {OUTPUT_GGUF}

To quantize (make smaller & faster):
  cd ~/llama.cpp
  ./llama-quantize ../{OUTPUT_GGUF} ../insurance-model-q4.gguf q4_k_m

To chat:
  python chat_gguf.py
""")
