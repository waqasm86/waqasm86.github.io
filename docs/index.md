# On-Device AI for Legacy NVIDIA GPUs

**Product-minded engineer building production-ready AI tools for Ubuntu 22.04 + old NVIDIA GPUs**

---

## The llcuda Ecosystem

Making large language models accessible on legacy hardware through empirical engineering and zero-configuration design.

### Featured Project: llcuda

**[llcuda on PyPI](https://pypi.org/project/llcuda/)** is a Python package that brings LLM inference to old NVIDIA GPUs with minimal setup. Built specifically for Ubuntu 22.04 and tested extensively on GeForce 940M (1GB VRAM).

```bash
pip install llcuda
python -m llcuda
```

**Key Features:**

- **Zero Configuration**: Works out of the box on Ubuntu 22.04
- **Legacy GPU Support**: Optimized for GPUs with 1GB VRAM (GeForce 940M tested)
- **Production Ready**: Published to PyPI with comprehensive testing
- **JupyterLab Integration**: First-class support for notebook workflows
- **Empirical Performance**: ~15 tokens/second with Gemma 2 2B Q4_K_M on GeForce 940M

### Infrastructure: Ubuntu-Cuda-Llama.cpp-Executable

The foundation of the llcuda ecosystem is a pre-built, statically-linked llama.cpp binary compiled for Ubuntu 22.04 with CUDA 12.6 support. This eliminates the need for users to compile llama.cpp themselves.

**Why This Matters:**

- No compilation required
- No CUDA toolkit installation needed
- No dependency hell
- Just download and run

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

All benchmarks run on **GeForce 940M (1GB VRAM, 384 CUDA cores, Maxwell architecture)** on Ubuntu 22.04.

### Gemma 2 2B Q4_K_M

```
Model: google/gemma-2-2b-it (Q4_K_M quantization)
Hardware: GeForce 940M (1GB VRAM)
Performance: ~15 tokens/second
Context: 2048 tokens
Memory Usage: ~950MB VRAM
```

### Real-World Use Cases

- **Interactive Chat**: Responsive enough for real-time conversation
- **Jupyter Notebooks**: Perfect for exploratory data analysis and prototyping
- **Local Development**: Test LLM integrations without cloud APIs
- **Learning**: Understand LLM behavior without expensive hardware

---

## Quick Start

Get up and running in under 5 minutes:

```bash
# Install llcuda
pip install llcuda

# Run interactive chat (downloads model automatically)
python -m llcuda

# Or use in Python
from llcuda import LLM

llm = LLM()
response = llm.chat("Explain quantum computing in simple terms.")
print(response)
```

For JupyterLab integration:

```python
from llcuda import LLM

llm = LLM(model="gemma-2-2b-it")

# Interactive chat with context
conversation = [
    "What is machine learning?",
    "How does it differ from traditional programming?",
    "Give me a practical example"
]

for message in conversation:
    response = llm.chat(message)
    print(f"User: {message}")
    print(f"AI: {response}\n")
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

## Projects

### Core Infrastructure

#### llcuda
**[PyPI Package](https://pypi.org/project/llcuda/)** | **[GitHub](https://github.com/waqasm86/llcuda)**

Python package for LLM inference on legacy NVIDIA GPUs. Zero-configuration installation, JupyterLab integration, empirically tested on GeForce 940M.

[Explore llcuda documentation &rarr;](/llcuda/)

#### Ubuntu-Cuda-Llama.cpp-Executable
**[GitHub](https://github.com/waqasm86/Ubuntu-Cuda-Llama.cpp-Executable)**

Pre-built llama.cpp binary for Ubuntu 22.04 with CUDA 12.6 support. The foundation that makes llcuda possible.

[View documentation &rarr;](/ubuntu-cuda-executable/)

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
- CUDA 12.6

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
