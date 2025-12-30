# Installation Guide

Complete installation instructions for llcuda v1.1.0.

---

## Quick Install

```bash
# Install or upgrade to latest version
pip install --upgrade llcuda

# Or install specific version
pip install llcuda==1.1.0
```

**First-time setup:** On first import, llcuda will automatically download optimized binaries (~700 MB) and a default model (~770 MB) based on your GPU. This is a one-time download that takes 3-5 minutes depending on your internet connection.

```python
import llcuda  # First import triggers automatic setup
# 🎯 llcuda First-Time Setup
# 🎮 GPU Detected: [Your GPU] (Compute X.X)
# 📥 Downloading binaries...
# 📥 Downloading model...
# ✅ Setup Complete!
```

---

## System Requirements

### Operating System
- **Supported**: Ubuntu 22.04 LTS (tested)
- **Likely works**: Ubuntu 20.04+, Debian 11+, other Linux distros

### Hardware
- **GPU**: NVIDIA with compute capability 5.0+ (Maxwell architecture or later)
- **VRAM**: 1GB minimum (for 1B-2B models)
- **CPU**: Any modern x86_64 processor
- **RAM**: 4GB+ recommended

### Software
- **Python**: 3.11 or later
- **CUDA**: Not required (bundled in package)
- **GPU Drivers**: NVIDIA drivers 535+ recommended

---

## Supported GPUs

### Tested
- **GeForce 940M** (1GB VRAM) - Primary test platform

### Should Work (compute capability 5.0+)
- **GeForce 900 series**: 940M, 950M, 960M, 970M, 980M
- **GeForce 800 series**: 840M, 850M, 860M
- **GeForce GTX series**: 750, 750 Ti, 950, 960, 970, 980 and newer
- **GeForce RTX series**: All models
- **Quadro/Tesla**: Maxwell generation and later

---

## Installation Steps

### 1. Check GPU

```bash
nvidia-smi
```

You should see your GPU listed. If not, install NVIDIA drivers first.

### 2. Install llcuda

```bash
# Install latest version
pip install --upgrade llcuda
```

### 3. Verify Installation

```python
import llcuda

llcuda.print_system_info()
```

You should see your GPU detected and llama-server auto-configured.

---

## Troubleshooting

### GPU Not Detected

**Problem**: `llcuda.check_cuda_available()` returns `False`

**Solutions**:
1. Check NVIDIA drivers: `nvidia-smi`
2. Reinstall drivers if needed
3. Verify compute capability ≥ 5.0

### Import Error

**Problem**: `ModuleNotFoundError: No module named 'llcuda'`

**Solution**:
```bash
pip install --upgrade llcuda
```

### Model Download Fails

**Problem**: HuggingFace download timeout or fails

**Solutions**:
1. Check internet connection
2. Try again (HuggingFace may be temporarily down)
3. Use local GGUF file instead:
   ```python
   engine.load_model("/path/to/model.gguf")
   ```

### Out of Memory

**Problem**: CUDA out of memory error

**Solutions**:
1. Use smaller model (gemma-3-1b-Q4_K_M for 1GB VRAM)
2. Reduce GPU layers:
   ```python
   engine.load_model("gemma-3-1b-Q4_K_M", gpu_layers=10)
   ```
3. Close other GPU applications

---

## Verification

Test your installation:

```python
import llcuda

# System check
llcuda.print_system_info()

# Quick inference test
engine = llcuda.InferenceEngine()
engine.load_model("gemma-3-1b-Q4_K_M")
result = engine.infer("Hello!", max_tokens=20)
print(result.text)
print(f"✓ llcuda working! Speed: {result.tokens_per_sec:.1f} tok/s")
```

---

## Next Steps

- **[Quick Start](/llcuda/quickstart/)** - Basic usage examples
- **[Performance](/llcuda/performance/)** - Benchmarks and optimization
- **[Examples](/llcuda/examples/)** - Production code samples
