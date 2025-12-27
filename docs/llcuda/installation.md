# Installation Guide

Comprehensive installation instructions for llcuda, including system requirements, installation methods, verification, and troubleshooting.

---

## System Requirements

### Operating System

**Supported:**
- Ubuntu 22.04 LTS (tested and recommended)
- Ubuntu 20.04 LTS (should work)
- Debian 11+ (should work)
- Other Linux distributions (may work, not tested)

**Not Currently Supported:**
- Windows (planned for future release)
- macOS (not planned - limited CUDA support)

### Hardware Requirements

**GPU Requirements:**
- NVIDIA GPU with CUDA support
- Compute capability 5.0 or higher (Maxwell architecture or later)
- Minimum 1GB VRAM (for 2B parameter models with Q4_K_M quantization)

**Tested GPUs:**
- GeForce 940M (1GB VRAM) - Primary test platform
- GeForce GTX 1050 (2GB VRAM)
- GeForce GTX 1650 (4GB VRAM)

**Supported GPU Families:**
- GeForce 900 series (940M, 950M, 960M, 970M, 980M, GTX 950, GTX 960, GTX 970, GTX 980, GTX 980 Ti)
- GeForce 800 series (840M, 850M, 860M, 870M, 880M)
- GeForce GTX 750/750 Ti and all newer models
- Quadro K-series (K620, K1200, K2200, etc.) and newer
- Tesla K-series and newer

**Check Your GPU:**
```bash
# List NVIDIA GPUs
lspci | grep -i nvidia

# Check compute capability
nvidia-smi --query-gpu=compute_cap --format=csv
```

### Software Requirements

**Python:**
- Python 3.8 or newer
- pip (Python package manager)

**NVIDIA Drivers:**
- NVIDIA driver version 450.80.02 or newer
- No CUDA toolkit installation required (llcuda includes pre-built binaries)

**Check Your Python Version:**
```bash
python3 --version
pip3 --version
```

**Check Your NVIDIA Driver:**
```bash
nvidia-smi
```

### Disk Space

- **Package**: ~50MB
- **Models**: 400MB - 2GB per model (cached in `~/.cache/llcuda/`)
- **Recommended free space**: 5GB minimum

---

## Installation Methods

### Method 1: pip (Recommended)

The easiest and recommended way to install llcuda:

```bash
# Install from PyPI
pip install llcuda

# Or with pip3 explicitly
pip3 install llcuda

# Install for current user only (no sudo needed)
pip install --user llcuda
```

**Verify Installation:**
```bash
python -c "import llcuda; print(llcuda.__version__)"
```

### Method 2: pip with Virtual Environment

For isolated installation (recommended for development):

```bash
# Create virtual environment
python3 -m venv llcuda-env

# Activate virtual environment
source llcuda-env/bin/activate

# Install llcuda
pip install llcuda

# Verify
python -c "import llcuda; print(llcuda.__version__)"
```

**To use llcuda later:**
```bash
source llcuda-env/bin/activate
python -m llcuda
```

### Method 3: Install from GitHub (Development)

For the latest development version:

```bash
# Clone repository
git clone https://github.com/waqasm86/llcuda.git
cd llcuda

# Install in development mode
pip install -e .

# Verify
python -c "import llcuda; print(llcuda.__version__)"
```

### Method 4: Jupyter/Colab Installation

For use in Jupyter notebooks:

```python
# In a Jupyter notebook cell
!pip install llcuda

# Verify
import llcuda
print(llcuda.__version__)
```

---

## Post-Installation Setup

### Verify GPU Detection

Check that llcuda can detect your GPU:

```python
from llcuda import LLM

llm = LLM(verbose=True)
# Should print: "Using GPU: [Your GPU Name]"
```

### Download Default Model

llcuda downloads models automatically on first use, but you can pre-download:

```python
from llcuda import LLM

# This will download Gemma 2 2B (~1.4GB)
llm = LLM(model="gemma-2-2b-it")
print("Model downloaded and cached")
```

**Model cache location**: `~/.cache/llcuda/models/`

### Verify Installation

Run the comprehensive verification script:

```python
from llcuda import LLM
import time

print("llcuda Installation Verification")
print("=" * 50)

# 1. Check version
import llcuda
print(f"✓ llcuda version: {llcuda.__version__}")

# 2. Check GPU
import subprocess
result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'],
                       capture_output=True, text=True)
print(f"✓ GPU: {result.stdout.strip()}")

# 3. Initialize LLM
print("\nInitializing LLM...")
start = time.time()
llm = LLM(model="gemma-2-2b-it", verbose=False)
print(f"✓ Model loaded in {time.time() - start:.1f}s")

# 4. Test inference
print("\nTesting inference...")
start = time.time()
response = llm.chat("Say 'Hello World'")
elapsed = time.time() - start
tokens = len(response.split())
speed = tokens / elapsed
print(f"✓ Generated response in {elapsed:.1f}s")
print(f"✓ Speed: {speed:.1f} tokens/second")

# 5. Test context
print("\nTesting context retention...")
llm.chat("My favorite number is 42")
response = llm.chat("What is my favorite number?")
if "42" in response:
    print("✓ Context retention working")
else:
    print("✗ Context retention issue")

print("\n" + "=" * 50)
print("Installation verified successfully!")
print("\nNext steps:")
print("1. Run interactive chat: python -m llcuda")
print("2. View examples: https://waqasm86.github.io/llcuda/examples/")
```

---

## Configuration

llcuda works with zero configuration, but you can customize behavior:

### Environment Variables

```bash
# Set custom model cache directory
export LLCUDA_MODEL_DIR=/path/to/models

# Disable GPU (CPU only, very slow)
export LLCUDA_USE_GPU=0

# Set default model
export LLCUDA_DEFAULT_MODEL=llama-3.2-1b-instruct
```

### Config File (Optional)

Create `~/.llcuda/config.json`:

```json
{
  "default_model": "gemma-2-2b-it",
  "model_dir": "~/.cache/llcuda/models",
  "max_tokens": 512,
  "temperature": 0.7,
  "context_length": 2048,
  "gpu_layers": -1
}
```

---

## Upgrading

### Upgrade to Latest Version

```bash
pip install --upgrade llcuda
```

### Upgrade to Specific Version

```bash
pip install llcuda==0.1.5
```

### Check Current Version

```bash
pip show llcuda
```

Or in Python:
```python
import llcuda
print(llcuda.__version__)
```

---

## Uninstalling

### Remove Package

```bash
pip uninstall llcuda
```

### Remove Cached Models

```bash
rm -rf ~/.cache/llcuda
```

### Remove Configuration

```bash
rm -rf ~/.llcuda
```

---

## Troubleshooting

### Issue: "No module named 'llcuda'"

**Cause**: llcuda not installed or wrong Python environment

**Solution:**
```bash
# Reinstall
pip install llcuda

# Or check which Python you're using
which python
which pip

# Use explicit pip3/python3
pip3 install llcuda
python3 -c "import llcuda"
```

### Issue: "CUDA out of memory"

**Cause**: Model too large for your GPU's VRAM

**Solutions:**

1. **Use a smaller model:**
```python
# Instead of Gemma 2 2B
llm = LLM(model="gemma-2-2b-it")

# Try Llama 3.2 1B
llm = LLM(model="llama-3.2-1b-instruct")

# Or Qwen 2.5 0.5B
llm = LLM(model="qwen-2.5-0.5b-instruct")
```

2. **Reduce context length:**
```python
llm = LLM(context_length=1024)  # Default is 2048
```

3. **Offload layers to CPU (slower):**
```python
llm = LLM(gpu_layers=16)  # Default is -1 (all on GPU)
```

### Issue: "NVIDIA driver not found"

**Cause**: NVIDIA driver not installed or too old

**Solution:**
```bash
# Check if driver is installed
nvidia-smi

# If not installed, install NVIDIA driver
sudo ubuntu-drivers autoinstall

# Or install specific version
sudo apt install nvidia-driver-535

# Reboot
sudo reboot
```

### Issue: "Model download failed"

**Cause**: Network issue or Hugging Face API limit

**Solutions:**

1. **Retry with force download:**
```python
llm = LLM(model="gemma-2-2b-it", force_download=True)
```

2. **Manual download:**
```bash
# Download model manually
cd ~/.cache/llcuda/models
wget https://huggingface.co/google/gemma-2-2b-it-GGUF/resolve/main/gemma-2-2b-it.Q4_K_M.gguf
```

3. **Check internet connection and Hugging Face status:**
```bash
curl -I https://huggingface.co
```

### Issue: "Slow generation (< 5 tokens/second)"

**Cause**: Model running on CPU instead of GPU

**Diagnosis:**
```python
from llcuda import LLM

llm = LLM(verbose=True)
# Should show "Using GPU: [Your GPU Name]"
# If shows "Using CPU", GPU not detected
```

**Solutions:**

1. **Verify NVIDIA driver:**
```bash
nvidia-smi
```

2. **Check GPU compute capability:**
```bash
nvidia-smi --query-gpu=compute_cap --format=csv
# Should be >= 5.0
```

3. **Reinstall llcuda:**
```bash
pip uninstall llcuda
pip install --no-cache-dir llcuda
```

### Issue: "Permission denied"

**Cause**: No write permission for model cache directory

**Solution:**
```bash
# Fix permissions
mkdir -p ~/.cache/llcuda
chmod -R u+w ~/.cache/llcuda

# Or install to custom directory
export LLCUDA_MODEL_DIR=/tmp/llcuda_models
```

### Issue: "Segmentation fault"

**Cause**: Incompatible CUDA version or corrupted binary

**Solutions:**

1. **Check CUDA version:**
```bash
nvidia-smi | grep "CUDA Version"
# Should be 12.0 or higher
```

2. **Update NVIDIA driver:**
```bash
sudo ubuntu-drivers autoinstall
sudo reboot
```

3. **Reinstall llcuda:**
```bash
pip uninstall llcuda
pip install --no-cache-dir llcuda
```

### Issue: "ImportError: cannot import name 'LLM'"

**Cause**: Old version of llcuda or conflicting package

**Solution:**
```bash
# Uninstall all versions
pip uninstall llcuda -y

# Clear pip cache
pip cache purge

# Reinstall latest
pip install llcuda

# Verify
python -c "from llcuda import LLM; print('Success')"
```

### Issue: "Response is gibberish or nonsensical"

**Cause**: Corrupted model file or wrong quantization

**Solution:**
```bash
# Remove corrupted model cache
rm -rf ~/.cache/llcuda/models/*

# Re-download
python -c "from llcuda import LLM; llm = LLM(force_download=True)"
```

---

## Platform-Specific Notes

### Ubuntu 22.04 (Recommended)

Fully supported and tested. No special configuration needed.

```bash
# Install dependencies (if needed)
sudo apt update
sudo apt install python3 python3-pip

# Install llcuda
pip3 install llcuda
```

### Ubuntu 20.04

Should work with default settings. If issues:

```bash
# Update Python to 3.8+
sudo apt update
sudo apt install python3.8 python3-pip
```

### Other Linux Distributions

May work but not officially tested. Requirements:
- NVIDIA driver 450.80.02+
- Python 3.8+
- GLIBC 2.31+ (check with `ldd --version`)

### WSL2 (Windows Subsystem for Linux)

Not officially supported. CUDA support in WSL2 is experimental.

### Docker

Not yet officially supported. Coming in future release.

---

## Developer Installation

For contributing to llcuda:

```bash
# Clone repository
git clone https://github.com/waqasm86/llcuda.git
cd llcuda

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/

# Run linting
flake8 llcuda/
black llcuda/
```

---

## Getting Help

If you're still having issues:

1. **Check existing issues**: [GitHub Issues](https://github.com/waqasm86/llcuda/issues)
2. **Search documentation**: [waqasm86.github.io](https://waqasm86.github.io)
3. **Ask the community**: [GitHub Discussions](https://github.com/waqasm86/llcuda/discussions)
4. **Contact directly**: [waqasm86@gmail.com](mailto:waqasm86@gmail.com)

**When reporting issues, include:**
- llcuda version (`python -c "import llcuda; print(llcuda.__version__)"`)
- Python version (`python --version`)
- GPU model (`nvidia-smi`)
- Operating system (`cat /etc/os-release`)
- Complete error message

---

## Next Steps

Installation complete! Now:

1. **[Quick Start Guide](/llcuda/quickstart/)** - Get running in 5 minutes
2. **[Performance Guide](/llcuda/performance/)** - Optimize for your GPU
3. **[Examples](/llcuda/examples/)** - Production code samples

---

## Quick Reference

### Installation
```bash
pip install llcuda
```

### Verification
```bash
python -c "import llcuda; print(llcuda.__version__)"
```

### First Run
```bash
python -m llcuda
```

### Upgrade
```bash
pip install --upgrade llcuda
```

### Uninstall
```bash
pip uninstall llcuda
rm -rf ~/.cache/llcuda
```
