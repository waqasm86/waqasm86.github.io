# Quick Start Guide

Get llcuda v1.1.0 running in under 10 minutes.

---

## Installation (5-10 minutes for first-time setup)

```bash
# Install or upgrade to latest version
pip install --upgrade llcuda

# Or install specific version
pip install llcuda==1.1.0
```

**First-time Setup**: When you first import llcuda, it will automatically download:
- **Binaries** (~700 MB): Optimized for your GPU from GitHub Releases
- **Model** (~770 MB): Default Gemma 3 1B from Hugging Face
- **Total**: ~1.5 GB one-time download (3-5 minutes)
- **Subsequent uses**: Instant (files cached locally)

**Why the download?** llcuda v1.1.0 uses a hybrid bootstrap architecture:
- PyPI package: Only **51 KB** (Python code only)
- Supports **8 GPU architectures** (SM 5.0-8.9)
- Works on **Colab, Kaggle, and local GPUs**

---

## Basic Usage (2 minutes)

```python
import llcuda
# 🎯 First import triggers automatic setup (one-time):
#    - Detecting GPU: GeForce 940M (Compute 5.0)
#    - Downloading binaries from GitHub...
#    - Downloading model from Hugging Face...
#    - ✅ Setup Complete!

# Create inference engine (auto-detects GPU)
engine = llcuda.InferenceEngine()

# Load model (already downloaded during first import)
engine.load_model("gemma-3-1b-Q4_K_M")

# Run inference
result = engine.infer("What is artificial intelligence?", max_tokens=100)

# Display results
print(result.text)
print(f"Speed: {result.tokens_per_sec:.1f} tok/s")
```

**Output**:
```
Artificial intelligence (AI) is the simulation of human intelligence
in machines that are programmed to think and learn like humans...

Speed: 15.2 tok/s
```

---

## List Available Models

llcuda v1.1.0 includes 11 curated models in the registry:

```python
from llcuda.models import list_registry_models

models = list_registry_models()

for name, info in models.items():
    print(f"{name}: {info['description']}")
    print(f"  Size: {info['size_mb']} MB, Min VRAM: {info['min_vram_gb']} GB\n")
```

**Output**:
```
tinyllama-1.1b-Q5_K_M: TinyLlama 1.1B Chat (fastest option)
  Size: 800 MB, Min VRAM: 1 GB

gemma-3-1b-Q4_K_M: Google Gemma 3 1B (recommended for 1GB VRAM)
  Size: 700 MB, Min VRAM: 1 GB

llama-3.2-1b-Q4_K_M: Meta Llama 3.2 1B Instruct
  Size: 750 MB, Min VRAM: 1 GB

...
```

---

## Check System Info

```python
import llcuda

# Print comprehensive system information
llcuda.print_system_info()
```

**Output**:
```
=== llcuda System Information ===
llcuda version: 1.0.1
Python version: 3.11.0

=== CUDA Information ===
CUDA Available: Yes
CUDA Version: 12.8

GPU 0: GeForce 940M
  Memory: 1024 MB
  Driver: 535.183.01
  Compute Capability: 5.0

=== llama-server ===
Path: /home/user/.local/lib/python3.11/site-packages/llcuda/bin/llama-server
Status: Auto-configured (bundled)
```

---

## Interactive Conversation

```python
import llcuda

engine = llcuda.InferenceEngine()
engine.load_model("gemma-3-1b-Q4_K_M")

# Multi-turn conversation
prompts = [
    "What is machine learning?",
    "How does it differ from traditional programming?",
    "Give me a practical example"
]

for prompt in prompts:
    result = engine.infer(prompt, max_tokens=80)
    print(f"Q: {prompt}")
    print(f"A: {result.text}\n")
```

---

## Batch Inference

Process multiple prompts efficiently:

```python
import llcuda

engine = llcuda.InferenceEngine()
engine.load_model("gemma-3-1b-Q4_K_M")

prompts = [
    "Explain neural networks",
    "What is deep learning?",
    "Describe natural language processing"
]

results = engine.batch_infer(prompts, max_tokens=50)

for prompt, result in zip(prompts, results):
    print(f"Q: {prompt}")
    print(f"A: {result.text}")
    print(f"Speed: {result.tokens_per_sec:.1f} tok/s\n")
```

---

## Performance Metrics

Get detailed P50/P95/P99 latency statistics:

```python
import llcuda

engine = llcuda.InferenceEngine()
engine.load_model("gemma-3-1b-Q4_K_M")

# Run some inferences
for i in range(10):
    engine.infer("Hello, how are you?", max_tokens=20)

# Get metrics
metrics = engine.get_metrics()

print("Latency Statistics:")
print(f"  Mean: {metrics['latency']['mean_ms']:.2f} ms")
print(f"  p50:  {metrics['latency']['p50_ms']:.2f} ms")
print(f"  p95:  {metrics['latency']['p95_ms']:.2f} ms")
print(f"  p99:  {metrics['latency']['p99_ms']:.2f} ms")

print("\nThroughput:")
print(f"  Total Tokens: {metrics['throughput']['total_tokens']}")
print(f"  Tokens/sec: {metrics['throughput']['tokens_per_sec']:.2f}")
```

---

## Using Local GGUF Files

You can also use local GGUF model files:

```python
import llcuda

engine = llcuda.InferenceEngine()

# Find local GGUF models
models = llcuda.find_gguf_models()

if models:
    print(f"Found {len(models)} local GGUF models")
    # Use first model found
    engine.load_model(str(models[0]))
else:
    # Fall back to registry
    engine.load_model("gemma-3-1b-Q4_K_M")
```

---

## Context Manager Usage

Use llcuda with Python context managers for automatic cleanup:

```python
import llcuda

# Context manager handles cleanup automatically
with llcuda.InferenceEngine() as engine:
    engine.load_model("gemma-3-1b-Q4_K_M")

    result = engine.infer("Explain quantum computing", max_tokens=80)
    print(result.text)

# Engine automatically cleaned up after context exit
print("Resources cleaned up")
```

---

## Temperature Comparison

Compare outputs with different temperature settings:

```python
import llcuda

engine = llcuda.InferenceEngine()
engine.load_model("gemma-3-1b-Q4_K_M")

prompt = "Write a creative opening for a science fiction story"
temperatures = [0.3, 0.7, 1.2]

for temp in temperatures:
    result = engine.infer(prompt, temperature=temp, max_tokens=60)
    print(f"\nTemperature {temp}:")
    print(result.text)
```

---

## JupyterLab Integration

llcuda works seamlessly in Jupyter notebooks:

```python
import llcuda
import pandas as pd

# Create engine
engine = llcuda.InferenceEngine()
engine.load_model("gemma-3-1b-Q4_K_M")

# Analyze data with LLM
df = pd.read_csv("data.csv")
summary = df.describe().to_string()

analysis = engine.infer(f"Analyze this data:\n{summary}", max_tokens=150)
print(analysis.text)
```

See the complete [JupyterLab example notebook](https://github.com/waqasm86/llcuda/blob/main/examples/quickstart_jupyterlab.ipynb).

---

## Next Steps

- **[Installation Guide](/llcuda/installation/)** - Detailed setup and troubleshooting
- **[Performance Guide](/llcuda/performance/)** - Benchmarks and optimization tips
- **[Examples](/llcuda/examples/)** - Production-ready code samples
- **[GitHub](https://github.com/waqasm86/llcuda)** - Source code and issues

---

## Common Questions

### Which model should I use?

For 1GB VRAM: `gemma-3-1b-Q4_K_M` (recommended) or `tinyllama-1.1b-Q5_K_M` (faster)

For 2GB+ VRAM: `phi-3-mini-Q4_K_M` (best for code) or `llama-3.2-3b-Q4_K_M`

### How do I change GPU layers?

llcuda auto-configures based on your VRAM, but you can override:

```python
engine.load_model("gemma-3-1b-Q4_K_M", gpu_layers=30)  # More GPU offloading
```

### Can I use my own GGUF models?

Yes, either use local files:

```python
engine.load_model("/path/to/model.gguf")
```

Or HuggingFace models:

```python
engine.load_model("author/repo-name", model_filename="model.gguf")
```

### How do I unload a model?

```python
engine.unload_model()  # Stops server and frees resources
```
