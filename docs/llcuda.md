# llcuda - CUDA-Accelerated LLM Inference for Python

High-performance Python package for running LLM inference with CUDA acceleration and automatic server management.

[:fontawesome-brands-github: View on GitHub](https://github.com/waqasm86/llcuda){ .md-button }
[:fontawesome-brands-python: View on PyPI](https://pypi.org/project/llcuda/){ .md-button }

---

## Overview

**llcuda** is a production-ready Python package that brings CUDA-accelerated LLM inference to Python with **zero-configuration setup**. It automatically manages the llama-server lifecycle, discovers models and executables, and provides a clean Pythonic API for inference.

Built for ease of use in JupyterLab, notebooks, and production environments.

### Key Features

- 🚀 **CUDA-Accelerated** - Native CUDA support for maximum performance on NVIDIA GPUs
- 🤖 **Auto-Start** - Automatically manages llama-server lifecycle (no manual setup!)
- 🐍 **Pythonic API** - Clean, intuitive interface with context manager support
- 📊 **Performance Metrics** - Built-in latency and throughput tracking
- 🔄 **Streaming Support** - Real-time token generation with callbacks
- 📦 **Batch Processing** - Efficient multi-prompt inference
- 🎯 **Smart Discovery** - Automatically finds models and executables
- 💻 **JupyterLab Ready** - Perfect for interactive workflows
- ⚡ **Production-Grade** - Comprehensive error handling and monitoring

---

## Installation

### Quick Install

```bash
# Install from PyPI
pip install llcuda

# Download pre-built llama-server (Ubuntu 22.04)
wget https://github.com/waqasm86/Ubuntu-Cuda-Llama.cpp-Executable/releases/download/v0.1.0/llama.cpp-733c851f-bin-ubuntu-cuda-x64.tar.xz
tar -xf llama.cpp-733c851f-bin-ubuntu-cuda-x64.tar.xz

# Set environment variable
export LLAMA_SERVER_PATH=$PWD/bin/llama-server
```

---

## Quick Start

### Ultra-Simple Usage

```python
import llcuda

# Create engine and load model with auto-start
engine = llcuda.InferenceEngine()
engine.load_model(
    "/path/to/model.gguf",
    auto_start=True,  # Automatically starts llama-server
    gpu_layers=99     # Offload all layers to GPU
)

# Run inference
result = engine.infer("What is artificial intelligence?", max_tokens=100)
print(result.text)
print(f"Speed: {result.tokens_per_sec:.1f} tokens/sec")
```

### JupyterLab Workflow

```python
import llcuda

# Check system setup
llcuda.print_system_info()

# Find available models
models = llcuda.find_gguf_models()
print(f"Found {len(models)} models")

# Use auto-start with context manager
with llcuda.InferenceEngine() as engine:
    engine.load_model(models[0], auto_start=True)
    result = engine.infer("Explain quantum computing")
    print(result.text)
# Server automatically stopped when exiting context
```

---

## Features in Detail

### Automatic Server Management

llcuda handles all the complexity of managing llama-server:

```python
# Before llcuda: Manual server management
# Terminal 1: llama-server -m model.gguf --port 8090 -ngl 99
# Terminal 2: python script.py

# With llcuda: Automatic management
engine = llcuda.InferenceEngine()
engine.load_model("model.gguf", auto_start=True)
result = engine.infer("Hello!")  # Server managed automatically
```

### Smart Discovery

Automatically finds llama-server and GGUF models:

```python
import llcuda

# Find llama-server executable
server_path = llcuda.ServerManager().find_llama_server()
print(f"Found server: {server_path}")

# Find GGUF models in common locations
models = llcuda.find_gguf_models()
for model in models:
    print(f"Found: {model}")
```

### Performance Monitoring

Built-in metrics for latency and throughput:

```python
# Run multiple inferences
for _ in range(10):
    engine.infer("Test prompt", max_tokens=50)

# Get detailed metrics
metrics = engine.get_metrics()
print(f"Mean latency: {metrics['latency']['mean_ms']:.2f}ms")
print(f"p95 latency: {metrics['latency']['p95_ms']:.2f}ms")
print(f"Throughput: {metrics['throughput']['tokens_per_sec']:.2f} tok/s")
```

### Streaming Inference

Real-time token generation with callbacks:

```python
def on_chunk(text):
    print(text, end='', flush=True)

result = engine.infer_stream(
    prompt="Write a story about AI",
    callback=on_chunk,
    max_tokens=200
)
print(f"\n\nGenerated {result.tokens_generated} tokens")
```

### Batch Processing

Process multiple prompts efficiently:

```python
prompts = [
    "What is AI?",
    "What is ML?",
    "What is DL?"
]

results = engine.batch_infer(prompts, max_tokens=50)

for prompt, result in zip(prompts, results):
    print(f"Q: {prompt}")
    print(f"A: {result.text}\n")
```

---

## Performance

Benchmarks on NVIDIA GeForce 940M (1GB VRAM):

| Model | Quantization | GPU Layers | Throughput | Latency (P95) |
|-------|--------------|------------|------------|---------------|
| Gemma 3 1B | Q4_K_M | 20 | ~15 tok/s | ~200ms |
| Gemma 2B | Q4_K_M | 10 | ~12 tok/s | ~250ms |
| Qwen 0.5B | Q4_K_M | 16 | ~28 tok/s | ~150ms |

Higher-end GPUs (T4, P100, V100, A100) deliver significantly better performance.

---

## System Requirements

### Hardware
- **GPU**: NVIDIA GPU with CUDA support (Compute Capability 5.0+)
- **VRAM**: 1GB+ (depends on model size)
- **RAM**: 4GB+ recommended

### Software
- **Python**: 3.11+
- **CUDA**: 11.7+ or 12.0+
- **OS**: Linux (Ubuntu 20.04+), tested on Ubuntu 22.04

### Dependencies
- `numpy>=1.20.0`
- `requests>=2.20.0`

---

## API Reference

### InferenceEngine

Main interface for LLM inference.

**Key Methods:**
```python
load_model(model_path, gpu_layers=99, auto_start=False, ...)
infer(prompt, max_tokens=128, temperature=0.7, ...)
infer_stream(prompt, callback, ...)
batch_infer(prompts, ...)
get_metrics()
check_server()
unload_model()
```

### ServerManager

Low-level server lifecycle management.

**Key Methods:**
```python
start_server(model_path, port=8090, gpu_layers=99, ...)
stop_server()
find_llama_server()
check_server_health()
```

### InferResult

Result object with performance metrics.

**Properties:**
```python
result.success          # bool
result.text            # str
result.tokens_generated # int
result.latency_ms      # float
result.tokens_per_sec  # float
result.error_message   # str (if failed)
```

---

## Use Cases

### Research & Development
- Interactive model experimentation in JupyterLab
- Quick prototyping with auto-start functionality
- Performance benchmarking with built-in metrics

### Production Deployment
- Embedded LLM inference in Python applications
- API servers with CUDA acceleration
- Batch processing pipelines

### Education
- Teaching LLM inference concepts
- Demonstrating GPU acceleration benefits
- Hands-on workshops and tutorials

---

## Related Projects

- **[cuda-nvidia-systems-engg](cuda-nvidia-systems-engg.md)** - Production-grade C++/CUDA distributed inference system
- **[Ubuntu-Cuda-Llama.cpp-Executable](https://github.com/waqasm86/Ubuntu-Cuda-Llama.cpp-Executable)** - Pre-built llama.cpp binaries
- **[llama.cpp](https://github.com/ggerganov/llama.cpp)** - Upstream GGML/GGUF inference engine

---

## Links

- **GitHub Repository**: [https://github.com/waqasm86/llcuda](https://github.com/waqasm86/llcuda)
- **PyPI Package**: [https://pypi.org/project/llcuda/](https://pypi.org/project/llcuda/)
- **Issue Tracker**: [https://github.com/waqasm86/llcuda/issues](https://github.com/waqasm86/llcuda/issues)
- **Changelog**: [CHANGELOG.md](https://github.com/waqasm86/llcuda/blob/main/CHANGELOG.md)

---

## License

MIT License - See [LICENSE](https://github.com/waqasm86/llcuda/blob/main/LICENSE) for details.

---

**Built for production-grade on-device AI inference** 🚀
