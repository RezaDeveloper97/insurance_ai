# CLAUDE.MD
## Insurance Domain AI – LoRA Fine-Tuning Project

### Project Goal
Build a domain-specific AI assistant for the insurance industry that behaves like a professional insurance expert:
- Conservative, non-speculative answers
- Clear explanation of conditions, exclusions, and dependencies
- Avoids giving exact numbers without policy verification
- Suitable for offline / edge deployment in later stages

Focus is on **behavior alignment**, not adding new factual knowledge.

---

## Current Status (DONE)

### Strategy
- ❌ Full fine-tuning rejected
- ✅ LoRA-based fine-tuning selected
- Target: Insurance Q&A assistant (Iran insurance context)
- Goal: Fast proof-of-result

### Hardware & Environment
- Apple Silicon MacBook (M4)
- Python 3.11
- Training: LoRA (fp16, MPS/CPU)

### Tooling
- transformers
- datasets
- peft
- accelerate
- torch (MPS supported)

### Base Model
- mistralai/Mistral-7B-Instruct-v0.2

### Dataset
File: insurance_lora.json

Format:
```json
{
  "instruction": "...",
  "input": "...",
  "output": "..."
}
```

Purpose:
- Teach professional insurance behavior
- Conservative answers
- Legal-safe tone

### Training
File: train_lora.py

LoRA config:
- r = 8
- alpha = 16
- dropout = 0.05
- target_modules = q_proj, v_proj

Output:
- insurance-lora-output/
- Contains LoRA adapters only

---

## Proven
- Clear behavioral change after LoRA
- More structured and cautious responses
- No hallucinated certainty

---

## Not Done Yet
- QLoRA
- Quantization
- CPU-only inference optimization
- Edge deployment

---

## Next Steps
1. Before/after inference comparison
2. Dataset expansion (50–200 samples)
3. QLoRA (4-bit)
4. GGUF export / llama.cpp
5. Raspberry Pi / Edge deployment

---

## Philosophy
"We are not teaching the model insurance.
We are teaching it how a professional insurance expert behaves."
