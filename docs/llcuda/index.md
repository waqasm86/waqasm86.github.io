# llcuda v1.0.0: PyTorch-Style CUDA LLM Inference

**Zero-configuration CUDA-accelerated LLM inference for Python with bundled binaries, smart model loading, and hardware auto-configuration.**

[![PyPI](https://img.shields.io/pypi/v/llcuda)](https://pypi.org/project/llcuda/)
[![Python](https://img.shields.io/pypi/pyversions/llcuda)](https://pypi.org/project/llcuda/)
[![License](https://img.shields.io/github/license/waqasm86/llcuda)](https://github.com/waqasm86/llcuda)

---

## What is llcuda v1.0.0?

A **PyTorch-style Python package** that makes LLM inference on legacy NVIDIA GPUs as easy as:

```bash
pip install llcuda
```

```python
import llcuda

engine = llcuda.InferenceEngine()
engine.load_model("gemma-3-1b-Q4_K_M")  # Auto-downloads from HuggingFace
result = engine.infer("What is AI?")
print(result.text)
```

**That's it.** No manual binary downloads, no LLAMA_SERVER_PATH, no configuration files.

---

## Key Features - v1.0.0

### 1. Zero Configuration

**Truly zero setup**: Import automatically configures all paths and libraries.

```python
import llcuda  # ← This line configures everything

engine = llcuda.InferenceEngine()
engine.load_model("gemma-3-1b-Q4_K_M")
```

No environment variables. No manual path setup. No LLAMA_SERVER_PATH. It just works.

### 2. Bundled CUDA Binaries

**47 MB wheel with everything included**:
- llama-server binary (CUDA 12.8)
- All shared libraries
- CUDA runtime
- No external dependencies

```bash
pip install llcuda  # Installs binaries + libraries
python -c "import llcuda; print('Ready!')"  # Works immediately
```

### 3. Smart Model Loading

**11 curated models in the registry**, optimized for different VRAM tiers:

```python
# Auto-downloads from HuggingFace with user confirmation
engine.load_model("gemma-3-1b-Q4_K_M")       # 700 MB, 1GB VRAM
engine.load_model("tinyllama-1.1b-Q5_K_M")  # 800 MB, 1GB VRAM
engine.load_model("phi-3-mini-Q4_K_M")      # 2.2 GB, 2GB+ VRAM
```

Model registry handles:
- HuggingFace repository IDs
- Automatic downloading with progress bars
- Size and VRAM recommendations
- User confirmation before downloads

### 4. Hardware Auto-Configuration

**Detects GPU VRAM and optimizes settings automatically**:

```python
engine = llcuda.InferenceEngine()  # Detects: GeForce 940M, 1GB VRAM
engine.load_model("gemma-3-1b-Q4_K_M")

# Automatically sets:
# - gpu_layers=20 (based on VRAM)
# - ctx_size=512 (optimal for 1GB)
# - batch_size=256
```

No manual tuning required.

### 5. Performance Metrics

**P50/P95/P99 latency tracking built-in**:

```python
metrics = engine.get_metrics()

print(f"p50: {metrics['latency']['p50_ms']:.2f} ms")
print(f"p95: {metrics['latency']['p95_ms']:.2f} ms")
print(f"p99: {metrics['latency']['p99_ms']:.2f} ms")
```

Track performance in production with proper percentile metrics.

### 6. Production Ready

**Published to PyPI, works like PyTorch**:

```python
# Install
pip install llcuda

# Import and use
import llcuda
engine = llcuda.InferenceEngine()
```

Not a GitHub experiment - a maintained package with semantic versioning.

---

## Performance Benchmarks

All benchmarks on **GeForce 940M (1GB VRAM, 384 CUDA cores, Maxwell architecture)**, Ubuntu 22.04, llcuda v1.0.0.

### Gemma 3 1B Q4_K_M (Recommended)

```
Model: google/gemma-3-1b-it (Q4_K_M)
Hardware: GeForce 940M (1GB VRAM)
Performance: ~15 tokens/second
GPU Layers: 20 (auto-configured)
Context: 512 tokens
Memory Usage: ~800MB VRAM
```

**Fast enough for** interactive chat, code generation, data analysis.

### Available Models (11 total)

| Model | Size | Min VRAM | Performance (940M) | Use Case |
|-------|------|----------|-------------------|----------|
| **tinyllama-1.1b-Q5_K_M** | 800 MB | 1 GB | ~18 tok/s | Fastest option |
| **gemma-3-1b-Q4_K_M** | 700 MB | 1 GB | ~15 tok/s | Recommended |
| **llama-3.2-1b-Q4_K_M** | 750 MB | 1 GB | ~16 tok/s | Best quality |
| **phi-3-mini-Q4_K_M** | 2.2 GB | 2 GB | ~12 tok/s | Code-focused |
| **mistral-7b-Q4_K_M** | 4.1 GB | 4 GB | ~8 tok/s | Highest quality |

[View full model registry and benchmarks →](/llcuda/performance/)

---

## Quick Start

### Installation (30 seconds)

```bash
pip install llcuda
```

That's it. All binaries and libraries included.

### Basic Usage (2 minutes)

```python
import llcuda

# Create engine (auto-detects GPU)
engine = llcuda.InferenceEngine()

# Load model (auto-downloads)
engine.load_model("gemma-3-1b-Q4_K_M")

# Run inference
result = engine.infer("What is quantum computing?", max_tokens=100)
print(result.text)
print(f"Speed: {result.tokens_per_sec:.1f} tok/s")
```

### JupyterLab Integration

```python
import llcuda

engine = llcuda.InferenceEngine()
engine.load_model("gemma-3-1b-Q4_K_M")

# Interactive exploration
prompts = [
    "Explain neural networks",
    "Compare supervised vs unsupervised learning",
    "What is transfer learning?"
]

for prompt in prompts:
    result = engine.infer(prompt, max_tokens=80)
    print(f"Q: {prompt}")
    print(f"A: {result.text}\n")
```

[View detailed quick start guide →](/llcuda/quickstart/)

---

## What's New in v1.0.0

### Breaking Changes from v0.3.0

llcuda v1.0.0 is a **major rewrite** with significant improvements:

**Before (v0.3.0)**:
```python
import os
os.environ['LLAMA_SERVER_PATH'] = '/path/to/llama-server'

engine = llcuda.InferenceEngine()
engine.load_model("/path/to/model.gguf", auto_start=True, gpu_layers=20)
```

**After (v1.0.0)**:
```python
import llcuda  # Auto-configures everything

engine = llcuda.InferenceEngine()
engine.load_model("gemma-3-1b-Q4_K_M")  # Auto-downloads and configures
```

### Major Features

1. **Bundled Binaries** - 47 MB wheel with all CUDA binaries and libraries
2. **Auto-Configuration** - No manual path setup on import
3. **Model Registry** - 11 curated models with auto-download
4. **Hardware Auto-Config** - VRAM detection and optimal settings
5. **Performance Metrics** - P50/P95/P99 latency tracking

[View full changelog →](https://github.com/waqasm86/llcuda/blob/main/CHANGELOG.md)

---

## Use Cases

### Interactive Development

```python
import llcuda

engine = llcuda.InferenceEngine()
engine.load_model("gemma-3-1b-Q4_K_M")

# Generate code
code = engine.infer("Write a Python function for quicksort").text
print(code)

# Review code
review = engine.infer(f"Review this code:\n{code}").text
print(review)
```

### Data Science Workflows

```python
import pandas as pd
import llcuda

engine = llcuda.InferenceEngine()
engine.load_model("gemma-3-1b-Q4_K_M")

# Analyze data with LLM
df = pd.read_csv("sales.csv")
summary = df.describe().to_string()

analysis = engine.infer(f"Analyze this data:\n{summary}").text
print(analysis)
```

### Learning & Experimentation

```python
import llcuda

engine = llcuda.InferenceEngine()
engine.load_model("gemma-3-1b-Q4_K_M")

# Test different temperatures
for temp in [0.3, 0.7, 1.2]:
    result = engine.infer(
        "Write a haiku about AI",
        temperature=temp,
        max_tokens=50
    )
    print(f"Temperature {temp}:\n{result.text}\n")
```

[View more examples →](/llcuda/examples/)

---

## Supported Hardware

### Tested Platforms

**Primary**: GeForce 940M (1GB VRAM, Maxwell architecture), Ubuntu 22.04

**Should work on**:
- GeForce 900 series (940M, 950M, 960M, 970M, 980M)
- GeForce 800 series (840M, 850M, 860M)
- GeForce GTX 750/750 Ti and newer
- Any NVIDIA GPU with compute capability 5.0+

### Requirements

- **OS**: Ubuntu 22.04 LTS (tested), likely works on other Linux distros
- **GPU**: NVIDIA with compute capability 5.0+ (Maxwell or later)
- **VRAM**: 1GB minimum (for 1B-2B models with Q4_K_M quantization)
- **Python**: 3.11+
- **CUDA**: Not required (bundled in package)

---

## Technical Architecture

```
┌─────────────────────────────────────────┐
│      Your Python Code / Jupyter        │
└─────────────────┬───────────────────────┘
                  │
┌─────────────────▼───────────────────────┐
│     llcuda.InferenceEngine (Python)     │
│  - Auto-configuration on import         │
│  - Model registry management            │
│  - Hardware detection                   │
│  - Performance metrics                  │
└─────────────────┬───────────────────────┘
                  │
┌─────────────────▼───────────────────────┐
│    Bundled llama-server (CUDA 12.8)     │
│  - Pre-built binary (build 733c851f)    │
│  - Shared libraries included            │
│  - Automatic startup/shutdown           │
└─────────────────┬───────────────────────┘
                  │
┌─────────────────▼───────────────────────┐
│         NVIDIA GPU (CUDA)               │
│  - GeForce 940M tested                  │
│  - 1GB VRAM minimum                     │
│  - Compute capability 5.0+              │
└─────────────────────────────────────────┘
```

---

## Philosophy

### 1. Product-Minded Engineering

Build for real users on real hardware, not ideal scenarios.

### 2. Zero Configuration

If it requires manual setup, we haven't finished the job.

### 3. Empirical Testing

Every claim backed by measurements on GeForce 940M (1GB VRAM).

### 4. Production Quality

Published to PyPI, semantic versioning, comprehensive documentation.

### 5. PyTorch-Style API

Familiar interface for ML engineers. Import and use like any ML library.

---

## Comparison with Alternatives

| Feature | llcuda v1.0.0 | llama.cpp | llama-cpp-python | Ollama |
|---------|--------------|-----------|------------------|--------|
| **Installation** | `pip install` | Manual build | `pip install` + compile | Manual install |
| **CUDA Setup** | Bundled | Manual | Manual | Manual |
| **Model Loading** | Auto-download | Manual | Manual | Auto-download |
| **Legacy GPU** | Excellent | Good | Good | Limited |
| **Python API** | PyTorch-style | CLI only | Pythonic | CLI + API |
| **Config Required** | Zero | Manual | Manual | Manual |

**llcuda's advantage**: Only solution with truly zero configuration AND excellent legacy GPU support.

---

## Getting Started

Ready to run LLMs on your old GPU?

1. **[Quick Start Guide](/llcuda/quickstart/)** - Get running in 5 minutes
2. **[Installation Guide](/llcuda/installation/)** - Detailed setup instructions
3. **[Performance Data](/llcuda/performance/)** - Real benchmarks on real hardware
4. **[Examples](/llcuda/examples/)** - Production-ready code samples

---

## Links

- **PyPI**: [pypi.org/project/llcuda](https://pypi.org/project/llcuda/)
- **GitHub**: [github.com/waqasm86/llcuda](https://github.com/waqasm86/llcuda)
- **GitHub Release**: [v1.0.0](https://github.com/waqasm86/llcuda/releases/tag/v1.0.0)
- **Documentation**: [waqasm86.github.io/llcuda](https://waqasm86.github.io/llcuda/)

---

## Support

- **Issues**: [GitHub Issues](https://github.com/waqasm86/llcuda/issues)
- **Discussions**: [GitHub Discussions](https://github.com/waqasm86/llcuda/discussions)
- **Email**: [waqasm86@gmail.com](mailto:waqasm86@gmail.com)

---

## License

MIT License - see [LICENSE](https://github.com/waqasm86/llcuda/blob/main/LICENSE) for details.
