# llcuda: LLM Inference for Legacy NVIDIA GPUs

**A Python package that brings large language model inference to old NVIDIA GPUs with zero configuration.**

[![PyPI](https://img.shields.io/pypi/v/llcuda)](https://pypi.org/project/llcuda/)
[![Python](https://img.shields.io/pypi/pyversions/llcuda)](https://pypi.org/project/llcuda/)
[![License](https://img.shields.io/github/license/waqasm86/llcuda)](https://github.com/waqasm86/llcuda)

---

## Overview

llcuda is designed to make LLM inference accessible on hardware you already own. No expensive GPU upgrades. No complex compilation. No CUDA toolkit installation. Just `pip install llcuda` and start running models.

### The Challenge

Running large language models typically requires:
- Modern NVIDIA RTX GPUs
- 8GB+ VRAM
- Complex compilation of llama.cpp with CUDA support
- CUDA toolkit installation and configuration
- Deep understanding of model quantization

**This creates a huge barrier for:**
- Students learning AI/ML
- Developers with older laptops
- Researchers in resource-constrained environments
- Anyone with legacy NVIDIA GPUs (GeForce 900/800 series)

### The Solution

llcuda removes all these barriers:

```bash
pip install llcuda
python -m llcuda
```

That's it. No compilation. No CUDA toolkit. No configuration. It just works.

**How?**
- Pre-built llama.cpp binaries with CUDA 12.6 support
- Automatic model downloading from Hugging Face
- Optimized for low-VRAM GPUs (tested on 1GB)
- Intelligent quantization selection (Q4_K_M by default)
- JupyterLab-first design for data science workflows

---

## Key Features

### Zero Configuration
Install via pip and run immediately. No manual steps, no dependencies, no compilation.

```bash
pip install llcuda
python -m llcuda  # Interactive chat
```

### Legacy GPU Support
Tested extensively on GeForce 940M (1GB VRAM, Maxwell architecture). If you have a CUDA-capable NVIDIA GPU from 2014 or later, llcuda will work.

**Supported GPUs:**
- GeForce 900 series (940M, 950M, 960M, 970M, 980M)
- GeForce 800 series (840M, 850M, 860M)
- GeForce GTX 750/750 Ti and newer
- Any GPU with compute capability 5.0+ (Maxwell architecture and later)

### Production Ready
Published to PyPI with semantic versioning. Not a GitHub experiment, but a maintained package you can depend on.

- **PyPI**: [pypi.org/project/llcuda](https://pypi.org/project/llcuda/)
- **GitHub**: [github.com/waqasm86/llcuda](https://github.com/waqasm86/llcuda)
- **Version**: 0.1.x (actively maintained)

### JupyterLab Integration
First-class support for Jupyter notebooks. Perfect for:
- Exploratory data analysis with LLM assistance
- Prototyping LLM-powered features
- Interactive learning and experimentation
- Documentation generation

```python
from llcuda import LLM

llm = LLM()
response = llm.chat("Explain gradient descent")
print(response)
```

### Empirical Performance
Every performance claim is backed by real measurements on real hardware.

**GeForce 940M (1GB VRAM) - Gemma 2 2B Q4_K_M:**
- **Speed**: ~15 tokens/second
- **Memory**: ~950MB VRAM usage
- **Context**: 2048 tokens
- **Quality**: Coherent, context-aware responses

[View detailed benchmarks &rarr;](/llcuda/performance/)

---

## How It Works

### Architecture

```
┌─────────────────────────────────────────┐
│         Your Python Code / Jupyter      │
└─────────────────┬───────────────────────┘
                  │
┌─────────────────▼───────────────────────┐
│            llcuda (Python)              │
│  - Model management                     │
│  - API interface                        │
│  - Context handling                     │
└─────────────────┬───────────────────────┘
                  │
┌─────────────────▼───────────────────────┐
│  Ubuntu-Cuda-Llama.cpp-Executable       │
│  - Pre-built binary                     │
│  - CUDA 12.6 support                    │
│  - Optimized for legacy GPUs            │
└─────────────────┬───────────────────────┘
                  │
┌─────────────────▼───────────────────────┐
│         NVIDIA GPU (CUDA)               │
│  - GeForce 940M tested                  │
│  - 1GB VRAM minimum                     │
│  - Compute capability 5.0+              │
└─────────────────────────────────────────┘
```

### Component Breakdown

**1. llcuda (Python Package)**
- High-level Python API for LLM inference
- Automatic model downloading from Hugging Face
- Context window management
- Conversation history tracking
- Error handling and recovery

**2. Ubuntu-Cuda-Llama.cpp-Executable**
- Pre-compiled llama.cpp with CUDA support
- Statically linked (no external dependencies)
- Compiled with CUDA 12.6 for Ubuntu 22.04
- Optimized build flags for legacy GPUs

**3. Model Management**
- Automatic GGUF model downloading
- Smart quantization selection (Q4_K_M default)
- Model caching to avoid re-downloads
- Support for custom models via Hugging Face IDs

---

## Supported Models

llcuda works with any GGUF-format model from Hugging Face. Default recommendations for 1GB VRAM:

### Recommended Models

**Gemma 2 2B (Default)**
```python
llm = LLM(model="gemma-2-2b-it")
```
- **Size**: ~1.4GB (Q4_K_M quantization)
- **Performance**: ~15 tok/s on GeForce 940M
- **Use Case**: General chat, Q&A, code assistance
- **Context**: 2048 tokens

**Llama 3.2 1B**
```python
llm = LLM(model="llama-3.2-1b-instruct")
```
- **Size**: ~900MB (Q4_K_M quantization)
- **Performance**: ~18 tok/s on GeForce 940M
- **Use Case**: Fast responses, simple tasks
- **Context**: 2048 tokens

**Qwen 2.5 0.5B**
```python
llm = LLM(model="qwen-2.5-0.5b-instruct")
```
- **Size**: ~400MB (Q4_K_M quantization)
- **Performance**: ~25 tok/s on GeForce 940M
- **Use Case**: Ultra-fast, basic chat
- **Context**: 2048 tokens

### Custom Models

Use any GGUF model from Hugging Face:

```python
llm = LLM(
    model="TheBloke/Mistral-7B-Instruct-v0.2-GGUF",
    model_file="mistral-7b-instruct-v0.2.Q4_K_M.gguf"
)
```

!!! note "VRAM Considerations"
    For 1GB VRAM GPUs, stick to models ≤2B parameters with Q4_K_M quantization. Larger models require more VRAM or CPU offloading (which is much slower).

---

## Use Cases

### Interactive Development
Run LLMs locally during development without cloud API costs:

```python
from llcuda import LLM

llm = LLM()

# Generate code
code = llm.chat("Write a Python function to calculate Fibonacci numbers")
print(code)

# Review code
review = llm.chat(f"Review this code for improvements: {code}")
print(review)
```

### Jupyter Notebooks
Perfect for data science workflows:

```python
import pandas as pd
from llcuda import LLM

llm = LLM()

# Analyze data with LLM assistance
df = pd.read_csv("sales_data.csv")
summary = df.describe().to_string()

analysis = llm.chat(f"Analyze this sales data:\n{summary}")
print(analysis)
```

### Learning & Experimentation
Understand LLM behavior without cloud costs:

```python
from llcuda import LLM

llm = LLM()

# Test different prompting strategies
prompts = [
    "What is photosynthesis?",
    "Explain photosynthesis to a 5-year-old",
    "You are a biology professor. Explain photosynthesis.",
]

for prompt in prompts:
    response = llm.chat(prompt)
    print(f"Prompt: {prompt}")
    print(f"Response: {response}\n")
```

### Offline Applications
Build applications that work without internet:

```python
from llcuda import LLM

class OfflineAssistant:
    def __init__(self):
        self.llm = LLM(model="gemma-2-2b-it")

    def answer_question(self, question):
        return self.llm.chat(question)

    def summarize_text(self, text):
        prompt = f"Summarize this text:\n\n{text}"
        return self.llm.chat(prompt)

assistant = OfflineAssistant()
answer = assistant.answer_question("What is quantum computing?")
```

---

## Installation

Quick installation:

```bash
pip install llcuda
```

For detailed installation instructions, including system requirements and troubleshooting:

[View Installation Guide &rarr;](/llcuda/installation/)

---

## Quick Start

Get up and running in under 5 minutes:

```bash
# Install
pip install llcuda

# Run interactive chat
python -m llcuda
```

For detailed tutorials and examples:

[View Quick Start Guide &rarr;](/llcuda/quickstart/)

---

## Performance

Real-world benchmarks on actual hardware (GeForce 940M, 1GB VRAM):

| Model | Quantization | Speed | VRAM | Context |
|-------|--------------|-------|------|---------|
| Gemma 2 2B | Q4_K_M | ~15 tok/s | 950MB | 2048 |
| Llama 3.2 1B | Q4_K_M | ~18 tok/s | 750MB | 2048 |
| Qwen 2.5 0.5B | Q4_K_M | ~25 tok/s | 450MB | 2048 |

For comprehensive benchmarks and optimization tips:

[View Performance Guide &rarr;](/llcuda/performance/)

---

## Examples

Production-ready code samples for common use cases:

- **Basic Chat**: Simple question-answering
- **Context Management**: Multi-turn conversations
- **Custom Models**: Loading specific GGUF files
- **JupyterLab Integration**: Notebook workflows
- **Batch Processing**: Processing multiple inputs
- **Error Handling**: Robust production code

[View Examples &rarr;](/llcuda/examples/)

---

## Philosophy

llcuda is built on these principles:

### 1. Zero Configuration
Installation should be `pip install` and done. No manual steps, no compilation, no configuration files.

### 2. Real Hardware Testing
Every performance claim is tested on GeForce 940M (1GB VRAM). No theoretical benchmarks.

### 3. Production Quality
Published to PyPI, not just GitHub. Semantic versioning, proper packaging, comprehensive documentation.

### 4. Python-First
Designed for Python developers and data scientists. Native Jupyter support, Pythonic API.

### 5. Empirical Optimization
Optimize based on measurements, not assumptions. Profile on real hardware, fix real bottlenecks.

---

## Technical Details

### Requirements
- **OS**: Ubuntu 22.04 LTS (tested), likely works on other Linux distros
- **GPU**: NVIDIA GPU with compute capability 5.0+ (Maxwell architecture or later)
- **VRAM**: 1GB minimum (for 2B models with Q4_K_M quantization)
- **Python**: 3.8+
- **CUDA**: Not required (pre-built binaries include CUDA runtime)

### Dependencies
- Minimal Python dependencies (requests, tqdm)
- Pre-built llama.cpp binary (no compilation required)
- No CUDA toolkit installation needed

### Package Structure
```
llcuda/
├── __init__.py          # Public API
├── llm.py              # Main LLM class
├── model_manager.py    # Model downloading/caching
├── llama_wrapper.py    # llama.cpp interface
└── binaries/           # Pre-built executables
    └── llama-cli       # CUDA-enabled llama.cpp
```

---

## Comparison with Alternatives

| Tool | Compilation | CUDA Required | PyPI | Legacy GPU Support |
|------|-------------|---------------|------|-------------------|
| **llcuda** | No | No | Yes | Excellent |
| llama.cpp | Yes | Yes | No | Good |
| llama-cpp-python | Yes | Yes | Yes | Good |
| Ollama | Yes | Yes | No | Limited |
| vLLM | No | Yes | Yes | None |

**llcuda's Advantage**: Only solution with zero-configuration setup AND excellent legacy GPU support.

---

## Roadmap

**Current**: v0.1.x - Core functionality, PyPI publication

**Planned Features**:
- Support for more quantization formats (Q2_K, Q5_K_M, Q6_K)
- Multi-GPU support for older multi-GPU setups
- Windows support (pre-built binaries for Windows + CUDA)
- Model compression utilities
- Fine-tuning helpers for custom models
- Advanced context window management (sliding window, RAG integration)

---

## Contributing

llcuda is open source and welcomes contributions:

- **Bug Reports**: [GitHub Issues](https://github.com/waqasm86/llcuda/issues)
- **Feature Requests**: [GitHub Discussions](https://github.com/waqasm86/llcuda/discussions)
- **Pull Requests**: [GitHub PRs](https://github.com/waqasm86/llcuda/pulls)

**Areas for Contribution**:
- Testing on different GPU models
- Windows/MacOS support
- Additional model integrations
- Documentation improvements
- Performance optimization

---

## Support

**Documentation**: [waqasm86.github.io](https://waqasm86.github.io)
**GitHub**: [github.com/waqasm86/llcuda](https://github.com/waqasm86/llcuda)
**PyPI**: [pypi.org/project/llcuda](https://pypi.org/project/llcuda/)
**Email**: [waqasm86@gmail.com](mailto:waqasm86@gmail.com)

---

## License

MIT License - see [LICENSE](https://github.com/waqasm86/llcuda/blob/main/LICENSE) for details.

---

## Next Steps

Ready to get started?

1. **[Quick Start](/llcuda/quickstart/)** - Get running in 5 minutes
2. **[Installation](/llcuda/installation/)** - Detailed setup guide
3. **[Performance](/llcuda/performance/)** - Benchmarks and optimization
4. **[Examples](/llcuda/examples/)** - Production code samples
