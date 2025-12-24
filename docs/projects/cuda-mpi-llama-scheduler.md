# cuda-mpi-llama-scheduler

A production-style GPU inference pipeline for running large language models on constrained hardware with CUDA acceleration and multi-rank scheduling.

[:fontawesome-brands-github: View on GitHub](https://github.com/waqasm86/cuda-mpi-llama-scheduler){ .md-button }

---

## Overview

This project demonstrates a complete GPU inference pipeline that combines CUDA acceleration with llama.cpp and quantized models to run large language models efficiently on resource-constrained hardware. It features OpenAI-compatible API endpoints and multi-rank scheduling inspired by MPI-style load distribution.

**Key Achievement:**

Successfully running a 1B parameter LLM (Gemma-3-1B-IT) on a GPU with only 1GB VRAM while maintaining acceptable performance and providing production-ready HTTP endpoints.

---

## Features

- **llama.cpp HTTP Server**: OpenAI-compatible `/v1/chat/completions` endpoint
- **CUDA Acceleration**: GPU-accelerated inference on low-VRAM devices
- **Quantized Models**: GGUF Q4_K_M format for extreme memory efficiency
- **Multi-rank Scheduling**: MPI-inspired load distribution across concurrent requests
- **Performance Metrics**: Detailed latency percentiles (p50, p95, p99) and throughput tracking
- **Prompt Caching**: Optimized handling of repeated requests
- **Zero External Dependencies**: No Docker, Python, or ML framework requirements (beyond build tools)

---

## Hardware & Model Configuration

### Test Environment

- **GPU**: NVIDIA GeForce 940M
  - Compute Capability 5.0
  - VRAM: 1GB
- **Model**: Gemma-3-1B-IT
  - Format: GGUF Q4_K_M quantization
  - Size: ~762 MiB
  - Context: 32k tokens

This configuration demonstrates that modern LLM inference is possible even on older, resource-constrained hardware.

---

## Architecture

The system consists of three layers working together:

### 1. CUDA-backed Inference Engine
- llama.cpp with CUDA support
- Quantized model loading (Q4_K_M)
- GPU memory management for constrained devices
- Kernel execution optimization

### 2. HTTP Server Layer
- OpenAI-compatible REST API
- JSON request/response handling
- Connection management
- Error handling and logging

### 3. Scheduler & Load Distribution
- Multi-rank request distribution
- Concurrent request handling
- Load balancing across ranks
- Performance monitoring

---

## Project Structure

```
cuda-mpi-llama-scheduler/
├── src/                    # Scheduler and client implementations
│   ├── scheduler.cpp       # Multi-rank scheduling logic
│   └── client.cpp          # HTTP client for testing
├── include/mls/            # Header files
├── scripts/                # Build and execution scripts
│   ├── build.sh           # Compilation script
│   └── run_2ranks.sh      # 2-rank load test
├── docs/                   # Architecture and analysis guides
│   ├── architecture.md    # System design
│   ├── setup.md          # Installation guide
│   └── analysis.md       # Performance analysis
└── README.md
```

---

## Dependencies

- **llama.cpp**: CUDA-enabled build
- **CMake**: Build system (3.15+)
- **Ninja**: Fast build tool
- **nlohmann/json**: JSON parsing library
- **CUDA Toolkit**: 12.x
- **C++17 compiler**: GCC 11+ or Clang 14+

---

## Build

```bash
# Run the build script
./scripts/build.sh
```

The build script:
1. Configures CMake with CUDA support
2. Compiles llama.cpp with GPU acceleration
3. Builds the scheduler and client components
4. Links against required libraries

### Prerequisites

```bash
# Install build dependencies (Ubuntu/Debian)
sudo apt-get install cmake ninja-build libcurl4-openssl-dev

# Install nlohmann/json
sudo apt-get install nlohmann-json3-dev
```

---

## Run

### 1. Start the llama.cpp Server

```bash
./llama-server \
    -m /path/to/gemma-3-1b-it-Q4_K_M.gguf \
    --port 8090 \
    --mlock
```

**Parameters:**
- `-m`: Path to GGUF model file
- `--port`: HTTP server port (default: 8090)
- `--mlock`: Lock model in RAM to prevent swapping

### 2. Run Load Tests

```bash
# 2-rank concurrent test
./scripts/run_2ranks.sh

# Custom test with N ranks
./build/scheduler --ranks N --endpoint http://localhost:8090
```

---

## Performance

### Observed Metrics

On NVIDIA GeForce 940M (1GB VRAM):

- **Mean Latency**: ~3342ms
- **Throughput**: 16 tokens/sec
- **VRAM Usage**: 850-900 MiB
- **GPU Utilization**: Stable during inference

### Latency Distribution

- **p50**: ~3200ms
- **p95**: ~4500ms
- **p99**: ~5200ms

### Memory Efficiency

The Q4_K_M quantization enables:
- 4-bit weight quantization
- ~762 MiB model footprint
- ~100 MiB headroom for KV cache and activations

---

## API Usage

### OpenAI-Compatible Endpoint

```bash
curl -X POST http://localhost:8090/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "Explain CUDA in one sentence."}
    ],
    "max_tokens": 50,
    "temperature": 0.7
  }'
```

### Response Format

```json
{
  "id": "chatcmpl-123",
  "object": "chat.completion",
  "created": 1234567890,
  "model": "gemma-3-1b-it",
  "choices": [{
    "index": 0,
    "message": {
      "role": "assistant",
      "content": "CUDA is NVIDIA's parallel computing platform..."
    },
    "finish_reason": "stop"
  }],
  "usage": {
    "prompt_tokens": 12,
    "completion_tokens": 45,
    "total_tokens": 57
  }
}
```

---

## Multi-Rank Scheduling

The scheduler distributes requests across multiple "ranks" (concurrent workers), inspired by MPI programming models:

### Scheduling Strategy

1. **Round-Robin Distribution**: Requests assigned to ranks in rotation
2. **Concurrent Execution**: Each rank processes requests independently
3. **Aggregated Metrics**: Combined latency and throughput statistics

### Example: 2-Rank Test

```bash
# Terminal 1: Start server
./llama-server -m model.gguf --port 8090

# Terminal 2: Run 2-rank scheduler
./scripts/run_2ranks.sh
```

Output shows per-rank and aggregated performance metrics.

---

## Optimization Techniques

### 1. Prompt Caching
- Cache repeated prompt prefixes
- Reduce redundant computation
- Improve throughput for similar requests

### 2. Memory Management
- Model locking with `--mlock`
- Careful KV cache sizing
- Dynamic batch sizing based on VRAM

### 3. Quantization
- Q4_K_M: 4-bit weights with optimized kernels
- Minimal accuracy loss
- 4x memory reduction vs FP16

---

## Benchmarking

### Running Benchmarks

```bash
# Standard benchmark (10 requests)
./build/client --endpoint http://localhost:8090 --requests 10

# High-load test (100 requests, 4 ranks)
./build/scheduler --ranks 4 --requests 100
```

### Metrics Collected

- Request latency (p50, p95, p99, max)
- Throughput (tokens/sec, requests/sec)
- GPU utilization
- Memory usage

---

## Use Cases

- **Edge Inference**: Running LLMs on resource-constrained devices
- **Research**: Studying performance characteristics of quantized models
- **Education**: Learning CUDA-accelerated inference pipelines
- **Development**: Prototyping LLM applications without cloud dependencies

---

## Troubleshooting

### Out of Memory Errors

Reduce context size or use more aggressive quantization:
```bash
./llama-server -m model.gguf --ctx-size 2048
```

### Slow Performance

Enable GPU layers:
```bash
./llama-server -m model.gguf --n-gpu-layers 99
```

### Connection Refused

Check server is running:
```bash
curl http://localhost:8090/health
```

---

## Future Enhancements

- Support for multi-GPU systems
- Advanced scheduling algorithms (priority queues, backpressure)
- Integration with NCCL for distributed inference
- Dynamic model swapping
- Quantization-aware fine-tuning support

---

## Notes

This project showcases that sophisticated LLM inference is achievable on modest hardware through careful optimization, quantization, and efficient resource management. The combination of CUDA acceleration, quantized models, and smart scheduling enables production-quality inference on GPUs that would typically be considered insufficient for modern LLM workloads.

**Key Insight**: The bottleneck often isn't the GPU compute capability, but rather memory bandwidth and capacity. Quantization and efficient memory management unlock LLM inference on previously unsuitable hardware.

**Author**: Mohammad Waqas
