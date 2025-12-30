# On-Device AI for Legacy NVIDIA GPUs

**Product-minded engineer building production-ready AI tools for Ubuntu 22.04 + old NVIDIA GPUs**

---

## The llcuda Ecosystem

Making large language models accessible on legacy hardware through empirical engineering and zero-configuration design.

### Featured Project: llcuda v1.1.0

**[llcuda on PyPI](https://pypi.org/project/llcuda/)** is a PyTorch-style Python package that brings LLM inference to **all modern NVIDIA GPUs** with zero configuration. Works on GeForce 940M, Google Colab, and Kaggle with hybrid bootstrap architecture.

```bash
# Install or upgrade to latest version (only 51 KB!)
pip install --upgrade llcuda
```

```python
import llcuda  # Auto-downloads binaries and models on first run

engine = llcuda.InferenceEngine()
engine.load_model("gemma-3-1b-Q4_K_M")
result = engine.infer("What is AI?")
print(result.text)
```

**Key Features:**

- **Hybrid Bootstrap Architecture**: 51 KB PyPI package, auto-downloads binaries (~700 MB) and models (~770 MB) on first import
- **Universal GPU Support**: Works on all NVIDIA GPUs (compute 5.0-8.9) - Maxwell to Ada Lovelace
- **Cloud Platform Ready**: Google Colab (T4, P100, V100, A100), Kaggle (T4), local GPUs
- **Zero Configuration**: No manual path setup, everything auto-configured
- **Smart Model Loading**: Auto-download from Hugging Face registry
- **Hardware Auto-Config**: Detects VRAM and optimizes settings automatically
- **11 Curated Models**: Ready to use out of the box
- **Performance Metrics**: P50/P95/P99 latency tracking built-in
- **Professional Distribution**: Matches PyTorch/TensorFlow pattern
- **Empirical Performance**: ~15 tokens/second on GeForce 940M and Tesla T4

---

## Philosophy: Product-Minded Engineering

I build tools that **actually work** on real hardware people own. No assumptions about cutting-edge GPUs. No theoretical benchmarks.

**My Approach:**

1. **Target Real Hardware**: GeForce 940M (1GB VRAM) as the baseline
2. **Empirical Testing**: Every claim backed by measurements on actual hardware
3. **Zero-Configuration**: Installation should be `pip install` and done
4. **Production Quality**: Published to PyPI, not just GitHub repos
5. **Documentation First**: If users can't use it, it doesn't exist

---

## Performance Data

All benchmarks run on real hardware with llcuda v1.1.0.

### Gemma 3 1B Q4_K_M Performance

| GPU | VRAM | Speed | GPU Layers |
|-----|------|-------|------------|
| **GeForce 940M** (Local) | 1 GB | ~15 tok/s | 20 |
| **Tesla T4** (Colab/Kaggle) | 15 GB | ~15 tok/s | 26 (all) |
| **Tesla P100** (Colab) | 16 GB | ~18 tok/s | 26 (all) |
| **Tesla V100** (Colab Pro) | 16 GB | ~20 tok/s | 26 (all) |

**Auto-Configuration:**
- GPU and platform detected automatically
- VRAM analyzed
- Optimal settings calculated (gpu_layers, ctx_size, batch_size)
- No manual tuning required

### Available Models (11 total)

llcuda includes a curated registry of models tested on GeForce 940M:

- **gemma-3-1b-Q4_K_M** (700 MB) - Recommended for 1GB VRAM
- **tinyllama-1.1b-Q5_K_M** (800 MB) - Smallest option
- **phi-3-mini-Q4_K_M** (2.2 GB) - For 2GB+ VRAM
- **mistral-7b-Q4_K_M** (4.1 GB) - For 4GB+ VRAM
- ... and 7 more models

[View full model registry →](/llcuda/performance/)

### Real-World Use Cases

- **Interactive Chat**: Responsive enough for real-time conversation
- **Jupyter Notebooks**: Perfect for exploratory data analysis and prototyping
- **Local Development**: Test LLM integrations without cloud APIs
- **Learning**: Understand LLM behavior without expensive hardware
- **Production**: P50/P95/P99 latency tracking for monitoring

---

## Quick Start

Get up and running in under 10 minutes (includes one-time setup):

```bash
# Install llcuda (51 KB package)
pip install --upgrade llcuda
```

**First-time setup**: On first `import llcuda`, binaries (~700 MB) and models (~770 MB) will auto-download (3-5 minutes). Subsequent runs are instant.

```python
# Basic usage - auto-downloads model with confirmation
import llcuda

engine = llcuda.InferenceEngine()
engine.load_model("gemma-3-1b-Q4_K_M")  # Auto-downloads from HuggingFace
result = engine.infer("Explain quantum computing in simple terms.")
print(result.text)
print(f"Performance: {result.tokens_per_sec:.1f} tok/s")
```

For JupyterLab integration:

```python
import llcuda

engine = llcuda.InferenceEngine()

# Load model with auto-configuration for your GPU
engine.load_model("gemma-3-1b-Q4_K_M")

# Interactive chat with performance tracking
conversation = [
    "What is machine learning?",
    "How does it differ from traditional programming?",
    "Give me a practical example"
]

for message in conversation:
    result = engine.infer(message, max_tokens=100)
    print(f"User: {message}")
    print(f"AI: {result.text}")
    print(f"Speed: {result.tokens_per_sec:.1f} tok/s\n")
```

[View detailed quick start guide &rarr;](/llcuda/quickstart/)

---

## Why llcuda?

### The Problem

Most LLM tools assume you have:
- Modern NVIDIA GPUs (RTX series)
- 8GB+ VRAM
- Willingness to compile complex C++ projects
- Latest CUDA toolkit installed

**Reality**: Millions of users have older GPUs collecting dust.

### The Solution

llcuda is designed for the hardware people actually own:

- **GeForce 900 series**: 940M, 950M, 960M
- **GeForce 800 series**: 840M, 850M
- **Maxwell/Kepler architectures**: Still capable, just ignored
- **1-2GB VRAM**: More than enough for quantized models

### The Approach

1. **Pre-built binaries**: No compilation needed
2. **Quantized models**: Q4_K_M quantization for memory efficiency
3. **Empirical optimization**: Tested on actual hardware, not simulators
4. **Python-first**: Native integration with data science workflows

---

## Project: llcuda

**[PyPI Package](https://pypi.org/project/llcuda/)** | **[GitHub](https://github.com/waqasm86/llcuda)** | **[v1.1.0 Release](https://github.com/waqasm86/llcuda/releases/tag/v1.1.0)**

PyTorch-style Python package for LLM inference on all modern NVIDIA GPUs. Hybrid bootstrap architecture with 51 KB PyPI package, auto-download binaries and models, universal GPU support (SM 5.0-8.9), Google Colab and Kaggle compatibility. Empirically tested on GeForce 940M and Tesla T4.

**What's New in v1.1.0:**
- **Hybrid Bootstrap Architecture**: 51 KB PyPI package (was 327 MB)
- **Universal GPU Support**: SM 5.0, 6.0, 6.1, 7.0, 7.5, 8.0, 8.6, 8.9 (8 architectures)
- **Cloud Platforms**: Google Colab (T4, P100, V100, A100), Kaggle (T4)
- **Auto-Download**: Binaries and models downloaded on first import
- **Professional Distribution**: Matches PyTorch/TensorFlow pattern
- **Fully Backward Compatible**: All v1.0.x code works unchanged

[Explore llcuda documentation &rarr;](/llcuda/)

---

## Technical Stack

**Languages & Tools:**
- Python (packaging, PyPI distribution)
- CUDA (GPU acceleration)
- C++ (llama.cpp integration)
- CMake (build systems)

**Expertise:**
- PyPI package publishing and versioning
- CUDA programming for legacy GPUs (compute capability 5.0)
- Empirical performance testing and optimization
- Production-quality Python library design
- Technical documentation and developer experience

**Hardware Testing:**
- GeForce 940M (1GB VRAM, Maxwell architecture)
- Ubuntu 22.04 LTS
- CUDA 12.8 (build 7489)

---

## Get Started

Ready to run LLMs on your old GPU?

1. **[Quick Start Guide](/llcuda/quickstart/)** - Get running in 5 minutes
2. **[Installation Guide](/llcuda/installation/)** - Comprehensive setup instructions
3. **[Performance Data](/llcuda/performance/)** - Real benchmarks on real hardware
4. **[Examples](/llcuda/examples/)** - Production-ready code samples

---

## About

I'm a product-minded engineer focused on making AI tools accessible on hardware people actually own. I believe in empirical testing, zero-configuration design, and building tools that solve real problems.

**Published Work:**
- [llcuda on PyPI](https://pypi.org/project/llcuda/) - Python package for LLM inference on legacy GPUs

**Philosophy:**
- Build for real hardware, not ideal hardware
- Every claim backed by measurements
- Installation should be trivial
- Documentation is a feature, not an afterthought

[Read more about my background &rarr;](/about/)

---

## Contact

**Email**: [waqasm86@gmail.com](mailto:waqasm86@gmail.com)
**GitHub**: [github.com/waqasm86](https://github.com/waqasm86)
**PyPI**: [pypi.org/project/llcuda](https://pypi.org/project/llcuda/)

[Get in touch &rarr;](/contact/)
