# Ubuntu-Cuda-Llama.cpp-Executable

Pre-built llama.cpp binary with CUDA 12.8 support for Ubuntu 22.04. The essential infrastructure that powers llcuda. **Updated December 28, 2025** - Build 7489.

---

## Overview

**Ubuntu-Cuda-Llama.cpp-Executable** is a pre-compiled llama.cpp binary optimized for Ubuntu 22.04 with CUDA 12.8 support (build 7489). It eliminates the need for users to compile llama.cpp themselves, removing a major barrier to entry for running LLMs on legacy GPUs.

**Latest Update (Dec 28, 2025)**: Updated from build 6093 to build 7489 with critical CUDA bug fixes, performance improvements, and new features. Now uses shared libraries for better maintainability.

**Repository**: [github.com/waqasm86/Ubuntu-Cuda-Llama.cpp-Executable](https://github.com/waqasm86/Ubuntu-Cuda-Llama.cpp-Executable)

---

## Why This Exists

### The Problem

Running llama.cpp with CUDA support traditionally requires:

1. **Install CUDA Toolkit** (~3GB download, complex setup)
2. **Install build dependencies** (CMake, g++, CUDA compilers)
3. **Clone llama.cpp repository**
4. **Configure CMake** with correct CUDA flags
5. **Compile** (10-20 minutes, can fail with cryptic errors)
6. **Troubleshoot** compilation issues

**This is intimidating for most users** and creates a massive barrier to entry.

### The Solution

A pre-built binary that:
- Requires **zero compilation**
- Needs **no CUDA toolkit installation**
- Works **out of the box** on Ubuntu 22.04
- Is **statically linked** (no external dependencies)
- Is **optimized** for legacy GPUs (compute capability 5.0+)

**Just download and run.**

---

## Features

### Zero Dependencies

The distribution includes shared libraries for flexibility:
- CUDA runtime (12.8)
- cuBLAS library
- All llama.cpp dependencies
- Shared libraries (libggml-cuda.so, libllama.so, etc.)

**Wrapper script handles library paths automatically.**

### Optimized for Legacy GPUs

Compiled with:
- Compute capability 5.0 (Maxwell architecture)
- Optimizations for low-VRAM GPUs
- Efficient memory management
- Support for quantized models (GGUF format)

### Tested Hardware

Verified working on:
- **GeForce 940M** (1GB VRAM) - Primary test platform
- **GeForce GTX 1050** (2GB VRAM)
- **GeForce GTX 1650** (4GB VRAM)
- **GeForce 900/800 series**
- **Any Maxwell+ GPU** (compute capability 5.0+)

---

## Technical Details

### Compilation Flags

```bash
cmake -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA=ON \
  -DGGML_CUDA_FORCE_CUBLAS=ON \
  -DGGML_CUDA_FA=ON \
  -DGGML_CUDA_GRAPHS=ON \
  -DGGML_NATIVE=ON \
  -DGGML_OPENMP=ON

cmake --build build --config Release -j$(nproc)
```

**Key flags:**
- `GGML_CUDA=ON`: Enable CUDA support
- `GGML_CUDA_FORCE_CUBLAS=ON`: Use cuBLAS for matrix operations
- `GGML_CUDA_FA=ON`: Enable FlashAttention CUDA kernels
- `GGML_CUDA_GRAPHS=ON`: Use CUDA graphs for efficiency
- `GGML_NATIVE=ON`: Native CPU optimizations
- `GGML_OPENMP=ON`: OpenMP multi-threading

### Build Environment

- **OS**: Ubuntu 22.04 LTS
- **CUDA**: 12.8.61
- **GCC**: 11.4.0
- **CMake**: 3.24+
- **llama.cpp**: Build 7489 (commit 10b4f82d4, Dec 20, 2025)

### Binary Details

- **llama-server**: 6.5MB (uses shared libraries)
- **llama-cli**: 4.1MB (uses shared libraries)
- **Format**: ELF 64-bit LSB executable
- **Architecture**: x86-64
- **CUDA Library**: libggml-cuda.so.0.9.4 (24MB)
- **Libraries**: Shared libraries in lib/ directory

**Check library dependencies:**
```bash
ldd bin/llama-server | grep -E "cuda|ggml"
# Shows: libggml-cuda.so.0, libcudart.so.12, libcublas.so.12
```

---

## Usage

### Direct Usage

Download and run the binary directly:

```bash
# Download binary
wget https://github.com/waqasm86/Ubuntu-Cuda-Llama.cpp-Executable/releases/latest/download/llama-cli

# Make executable
chmod +x llama-cli

# Run inference
./llama-cli -m model.gguf -p "Hello, how are you?"
```

### Common Options

```bash
# Interactive chat mode
./llama-cli -m model.gguf -cnv

# Set context length
./llama-cli -m model.gguf -c 2048 -p "Your prompt"

# Control GPU layers (for VRAM management)
./llama-cli -m model.gguf -ngl 32 -p "Your prompt"

# Set generation parameters
./llama-cli -m model.gguf \
  -p "Your prompt" \
  --temp 0.7 \
  --top-p 0.9 \
  --top-k 40 \
  --repeat-penalty 1.1
```

### Integration with llcuda

llcuda automatically uses this binary under the hood:

```python
from llcuda import LLM

# llcuda downloads and uses the pre-built binary
llm = LLM()
response = llm.chat("Hello!")
```

**Users never need to interact with the binary directly** when using llcuda.

---

## Installation

### Method 1: Via llcuda (Recommended)

The easiest way is to install llcuda, which includes the binary:

```bash
pip install llcuda
```

llcuda automatically downloads and manages the binary.

### Method 2: Direct Download

Download the binary directly from GitHub:

```bash
# Download latest release
wget https://github.com/waqasm86/Ubuntu-Cuda-Llama.cpp-Executable/releases/latest/download/llama-cli

# Make executable
chmod +x llama-cli

# Move to PATH (optional)
sudo mv llama-cli /usr/local/bin/
```

### Method 3: Build from Source

If you want to build yourself:

```bash
# Install dependencies
sudo apt update
sudo apt install cmake build-essential

# Install CUDA 12.6
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
sudo mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/12.6.0/local_installers/cuda-repo-ubuntu2204-12-6-local_12.6.0-560.28.03-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2204-12-6-local_12.6.0-560.28.03-1_amd64.deb
sudo cp /var/cuda-repo-ubuntu2204-12-6-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt update
sudo apt install cuda-toolkit-12-6

# Clone llama.cpp
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp

# Build with CUDA
cmake -B build \
  -DLLAMA_CUDA=ON \
  -DLLAMA_CUDA_F16=ON \
  -DCMAKE_CUDA_ARCHITECTURES=50 \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_SHARED_LIBS=OFF

cmake --build build --config Release -j$(nproc)

# Binary at: build/bin/llama-cli
```

---

## Compatibility

### Supported Operating Systems

**Tested and Supported:**
- Ubuntu 22.04 LTS (primary target)

**Likely Compatible:**
- Ubuntu 20.04 LTS
- Ubuntu 24.04 LTS
- Debian 11+
- Linux Mint 21+

**Not Supported:**
- Windows (different build required)
- macOS (limited CUDA support)
- WSL2 (experimental CUDA support)

### Supported GPUs

**Minimum Requirements:**
- NVIDIA GPU with compute capability 5.0+
- Maxwell architecture or newer
- 1GB VRAM minimum (for small models)

**Tested GPUs:**
- GeForce 900 series (940M, 950M, 960M, etc.)
- GeForce 800 series (840M, 850M, etc.)
- GeForce GTX 750/750 Ti and newer
- Quadro K-series and newer

**Check your GPU:**
```bash
nvidia-smi --query-gpu=compute_cap --format=csv
# Should output 5.0 or higher
```

---

## Performance

Performance on **GeForce 940M (1GB VRAM)**:

| Model | Quantization | Speed | VRAM |
|-------|--------------|-------|------|
| Gemma 2 2B | Q4_K_M | ~15 tok/s | 950MB |
| Llama 3.2 1B | Q4_K_M | ~18 tok/s | 750MB |
| Qwen 2.5 0.5B | Q4_K_M | ~25 tok/s | 450MB |

**Faster GPUs see proportionally better performance:**
- GTX 1050 (2GB): ~2x faster
- GTX 1650 (4GB): ~3x faster

---

## Advantages Over Standard llama.cpp

### 1. No Compilation Required

**Standard llama.cpp:**
```bash
# ~20 minutes of compilation
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp
cmake -B build -DLLAMA_CUDA=ON
cmake --build build --config Release
```

**Ubuntu-Cuda-Llama.cpp-Executable:**
```bash
# Instant
wget <binary-url>
chmod +x llama-cli
```

### 2. No CUDA Toolkit Needed

**Standard llama.cpp:**
- Requires CUDA toolkit installation (~3GB)
- Complex environment setup
- Version compatibility issues

**Ubuntu-Cuda-Llama.cpp-Executable:**
- Only needs NVIDIA driver
- CUDA runtime embedded in binary
- Works out of the box

### 3. Optimized for Legacy GPUs

**Standard llama.cpp:**
- Default builds target modern GPUs
- May not optimize for Maxwell architecture

**Ubuntu-Cuda-Llama.cpp-Executable:**
- Compiled specifically for compute 5.0
- Optimized for low-VRAM scenarios
- Tested on GeForce 940M

### 4. Guaranteed Compatibility

**Standard llama.cpp:**
- Build issues with different CUDA versions
- CMake configuration problems
- Dependency conflicts

**Ubuntu-Cuda-Llama.cpp-Executable:**
- Tested on Ubuntu 22.04
- Static linking eliminates dependency issues
- Known working configuration

---

## Limitations

### Platform-Specific

This binary is built specifically for:
- **Linux** (Ubuntu 22.04)
- **x86-64 architecture**
- **NVIDIA GPUs** (CUDA)

**Does not support:**
- Windows (need different build)
- macOS (limited CUDA support)
- AMD GPUs (need ROCm build)
- ARM processors (need ARM build)

### CUDA Version

Binary includes CUDA 12.8 runtime. Requires:
- **NVIDIA driver 525.60.13 or newer**

Most modern drivers satisfy this requirement.

---

## Troubleshooting

### "CUDA driver version is insufficient"

**Issue**: Driver too old for CUDA 12.6

**Solution**: Update NVIDIA driver
```bash
sudo ubuntu-drivers autoinstall
sudo reboot
```

### "Illegal instruction (core dumped)"

**Issue**: CPU doesn't support required instruction set

**Solution**: Re-compile with broader compatibility
```bash
cmake -B build -DLLAMA_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="50;60;70;75;80"
```

### "CUDA out of memory"

**Issue**: Model too large for GPU VRAM

**Solution**: Reduce GPU layers
```bash
./llama-cli -m model.gguf -ngl 20  # Use fewer GPU layers
```

---

## Contributing

Contributions welcome! Areas for improvement:

1. **Support more platforms** (Windows, macOS)
2. **Optimize for newer GPUs** (Ampere, Ada Lovelace)
3. **Reduce binary size** (current: ~45MB)
4. **Add more build configurations**

**Repository**: [github.com/waqasm86/Ubuntu-Cuda-Llama.cpp-Executable](https://github.com/waqasm86/Ubuntu-Cuda-Llama.cpp-Executable)

---

## Related Projects

### llcuda
**[PyPI Package](https://pypi.org/project/llcuda/)** | **[Docs](/llcuda/)**

Python wrapper around this binary. Provides high-level API, automatic model downloading, and JupyterLab integration.

**Use llcuda if you want:**
- Python API
- Automatic model management
- JupyterLab integration
- Zero configuration

**Use the binary directly if you want:**
- Command-line interface
- Shell scripting
- Custom integration
- Maximum control

---

## License

MIT License - see [LICENSE](https://github.com/waqasm86/Ubuntu-Cuda-Llama.cpp-Executable/blob/main/LICENSE)

Based on [llama.cpp](https://github.com/ggerganov/llama.cpp) by Georgi Gerganov.

---

## Support

**Documentation**: [waqasm86.github.io](https://waqasm86.github.io)
**GitHub Issues**: [github.com/waqasm86/Ubuntu-Cuda-Llama.cpp-Executable/issues](https://github.com/waqasm86/Ubuntu-Cuda-Llama.cpp-Executable/issues)
**Email**: [waqasm86@gmail.com](mailto:waqasm86@gmail.com)

---

## Summary

Ubuntu-Cuda-Llama.cpp-Executable makes llama.cpp accessible to everyone by:
- Eliminating compilation complexity
- Removing CUDA toolkit requirement
- Optimizing for legacy GPUs
- Providing guaranteed compatibility

**It's the foundation that makes llcuda possible.**

[Explore llcuda &rarr;](/llcuda/)
