# Quick Start Guide

Get llcuda running in under 5 minutes. This guide assumes you have Ubuntu 22.04 and an NVIDIA GPU.

---

## Prerequisites Check

Before installation, verify your system meets the requirements:

```bash
# Check if you have an NVIDIA GPU
lspci | grep -i nvidia

# Check CUDA compute capability (should be 5.0+)
nvidia-smi

# Check Python version (need 3.8+)
python3 --version
```

!!! success "System Requirements Met?"
    If all commands above work, you're ready to install llcuda!

---

## Installation (30 seconds)

Install llcuda via pip:

```bash
pip install llcuda
```

That's it. No compilation, no CUDA toolkit, no configuration files.

??? question "Installation Failed?"
    See the [Installation Guide](/llcuda/installation/) for troubleshooting.

---

## First Run: Interactive Chat (2 minutes)

Launch the interactive chat interface:

```bash
python -m llcuda
```

**What happens on first run:**
1. llcuda detects your GPU
2. Downloads Gemma 2 2B model (~1.4GB) from Hugging Face
3. Loads model into GPU memory
4. Starts interactive chat

**First download takes 2-3 minutes on typical internet. Subsequent runs start instantly.**

??? info "Model Download Location"
    Models are cached in `~/.cache/llcuda/models/`. You won't need to download again.

---

## Your First Conversation

Once the model loads, you'll see:

```
llcuda v0.1.0 - LLM Inference on Legacy GPUs
Loaded: gemma-2-2b-it (Q4_K_M)
GPU: GeForce 940M (1GB VRAM)

Type your message (or 'quit' to exit):
>
```

Try asking a question:

```
> Explain quantum computing in simple terms
```

The model will generate a response at ~15 tokens/second on GeForce 940M.

**Tips:**
- Type `quit` or `exit` to close
- Press Ctrl+C to interrupt generation
- Each message maintains conversation context

---

## Using llcuda in Python (1 minute)

Create a Python file `test_llcuda.py`:

```python
from llcuda import LLM

# Initialize LLM (downloads model on first run)
llm = LLM()

# Ask a question
response = llm.chat("What is machine learning?")
print(response)
```

Run it:

```bash
python test_llcuda.py
```

**Output:**
```
Machine learning is a type of artificial intelligence where computers
learn from data without being explicitly programmed. Instead of following
rigid rules, ML systems find patterns in data and make predictions...
```

---

## JupyterLab Integration (30 seconds)

llcuda works perfectly in Jupyter notebooks:

```python
# Cell 1: Import and initialize
from llcuda import LLM
llm = LLM()

# Cell 2: Interactive chat
response = llm.chat("Explain gradient descent")
print(response)

# Cell 3: Follow-up question (maintains context)
response = llm.chat("Give me a Python example")
print(response)
```

The LLM maintains conversation context across cells, perfect for iterative development.

---

## Common Use Cases

### Simple Q&A

```python
from llcuda import LLM

llm = LLM()
answer = llm.chat("What is the capital of France?")
print(answer)
```

### Code Generation

```python
from llcuda import LLM

llm = LLM()
code = llm.chat("Write a Python function to calculate factorial")
print(code)
```

### Multi-Turn Conversation

```python
from llcuda import LLM

llm = LLM()

# Context is maintained across calls
llm.chat("My name is Alice and I'm learning Python")
llm.chat("What topics should I focus on?")
response = llm.chat("What was my name?")  # Remembers "Alice"
print(response)
```

### Data Analysis Helper

```python
import pandas as pd
from llcuda import LLM

df = pd.read_csv("sales.csv")
summary = df.describe().to_string()

llm = LLM()
analysis = llm.chat(f"Analyze this sales data:\n{summary}")
print(analysis)
```

---

## Choosing a Different Model

By default, llcuda uses Gemma 2 2B. You can specify a different model:

### Llama 3.2 1B (Faster)

```python
from llcuda import LLM

llm = LLM(model="llama-3.2-1b-instruct")
response = llm.chat("Hello!")
```

**Performance**: ~18 tok/s on GeForce 940M (faster but less capable)

### Qwen 2.5 0.5B (Ultra-Fast)

```python
from llcuda import LLM

llm = LLM(model="qwen-2.5-0.5b-instruct")
response = llm.chat("Hello!")
```

**Performance**: ~25 tok/s on GeForce 940M (fastest but basic)

### Custom Model from Hugging Face

```python
from llcuda import LLM

llm = LLM(
    model="TheBloke/Mistral-7B-Instruct-v0.2-GGUF",
    model_file="mistral-7b-instruct-v0.2.Q4_K_M.gguf"
)
```

!!! warning "VRAM Limitations"
    For 1GB VRAM GPUs, stick to ≤2B parameter models with Q4_K_M quantization.

---

## Performance Expectations

On **GeForce 940M (1GB VRAM)**:

| Model | Speed | Quality | Use Case |
|-------|-------|---------|----------|
| Gemma 2 2B | ~15 tok/s | High | General chat, code |
| Llama 3.2 1B | ~18 tok/s | Medium | Fast responses |
| Qwen 2.5 0.5B | ~25 tok/s | Basic | Simple tasks |

**What does ~15 tok/s feel like?**
- A 100-word response takes ~8 seconds
- Interactive enough for real-time chat
- Much faster than human reading speed

[View detailed benchmarks &rarr;](/llcuda/performance/)

---

## Configuration Options

llcuda works with zero configuration, but you can customize:

### Basic Options

```python
from llcuda import LLM

llm = LLM(
    model="gemma-2-2b-it",          # Model name
    max_tokens=512,                  # Max response length
    temperature=0.7,                 # Randomness (0-1)
    context_length=2048,             # Context window size
    gpu_layers=32,                   # Layers on GPU (auto-detected)
)
```

### Advanced Options

```python
from llcuda import LLM

llm = LLM(
    model="gemma-2-2b-it",
    max_tokens=1024,
    temperature=0.8,
    top_p=0.9,                       # Nucleus sampling
    top_k=40,                        # Top-K sampling
    repeat_penalty=1.1,              # Penalize repetition
    verbose=True,                    # Show generation stats
)
```

---

## Verifying Your Setup

Run this verification script to confirm everything works:

```python
from llcuda import LLM
import time

print("llcuda Verification Script")
print("-" * 40)

# Initialize
print("1. Initializing LLM...")
start = time.time()
llm = LLM(model="gemma-2-2b-it")
print(f"   ✓ Loaded in {time.time() - start:.1f}s")

# Test inference
print("\n2. Testing inference...")
start = time.time()
response = llm.chat("Say 'Hello World' and nothing else")
elapsed = time.time() - start
tokens = len(response.split())
print(f"   ✓ Generated {tokens} tokens in {elapsed:.1f}s")
print(f"   ✓ Speed: ~{tokens/elapsed:.1f} tok/s")

# Test context
print("\n3. Testing context retention...")
llm.chat("Remember this number: 42")
response = llm.chat("What number did I tell you to remember?")
if "42" in response:
    print("   ✓ Context maintained")
else:
    print("   ✗ Context not maintained")

print("\n" + "=" * 40)
print("Verification complete! llcuda is working.")
```

**Expected output:**
```
llcuda Verification Script
----------------------------------------
1. Initializing LLM...
   ✓ Loaded in 2.3s

2. Testing inference...
   ✓ Generated 4 tokens in 0.3s
   ✓ Speed: ~13.3 tok/s

3. Testing context retention...
   ✓ Context maintained

========================================
Verification complete! llcuda is working.
```

---

## Troubleshooting Quick Fixes

### "CUDA out of memory"

Your GPU ran out of VRAM. Solutions:

```python
# Use a smaller model
llm = LLM(model="llama-3.2-1b-instruct")

# Or reduce context length
llm = LLM(context_length=1024)

# Or offload some layers to CPU (slower)
llm = LLM(gpu_layers=16)  # Default is auto
```

### "Model download failed"

Check your internet connection and retry:

```python
from llcuda import LLM

# Force re-download
llm = LLM(model="gemma-2-2b-it", force_download=True)
```

### "Slow generation speed"

If you're getting <5 tok/s, the model might be using CPU instead of GPU:

```bash
# Verify GPU is detected
nvidia-smi

# Check llcuda is using GPU
python -c "from llcuda import LLM; llm = LLM(verbose=True)"
# Should show "Using GPU: GeForce XXX"
```

For more troubleshooting: [Installation Guide](/llcuda/installation/#troubleshooting)

---

## Next Steps

Now that llcuda is working, explore:

1. **[Installation Guide](/llcuda/installation/)** - Deep dive into setup and troubleshooting
2. **[Performance Guide](/llcuda/performance/)** - Optimize for your GPU
3. **[Examples](/llcuda/examples/)** - Production-ready code samples
4. **[Full Documentation](/llcuda/)** - Complete API reference

---

## Quick Reference

### Installation
```bash
pip install llcuda
```

### Interactive Chat
```bash
python -m llcuda
```

### Basic Python Usage
```python
from llcuda import LLM
llm = LLM()
response = llm.chat("Your question here")
```

### Different Models
```python
llm = LLM(model="llama-3.2-1b-instruct")  # Faster
llm = LLM(model="qwen-2.5-0.5b-instruct") # Ultra-fast
llm = LLM(model="gemma-2-2b-it")          # Default, best quality
```

### Common Options
```python
llm = LLM(
    model="gemma-2-2b-it",
    max_tokens=512,
    temperature=0.7,
    verbose=True
)
```

---

**You're all set! Start building with llcuda.**
