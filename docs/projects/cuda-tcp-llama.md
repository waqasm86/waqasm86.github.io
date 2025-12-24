# cuda-tcp-llama.cpp

A high-performance TCP-based inference server for CUDA-accelerated LLM inference using llama.cpp as the backend.

[:fontawesome-brands-github: View on GitHub](https://github.com/waqasm86/cuda-tcp-llama.cpp){ .md-button }

---

## Overview

This project implements a custom TCP-based inference data plane designed to serve CUDA-accelerated large language model (LLM) inference. It explores low-level networking, concurrency, and GPU integration challenges under tight hardware constraints like limited VRAM (≤1 GB).

**Key Focus Areas:**

- Custom binary TCP protocol optimized for minimal latency
- Non-blocking I/O with epoll-based concurrency
- Direct CUDA integration with llama.cpp
- Performance instrumentation (latency percentiles, throughput)
- Explicit control over data movement for predictable latency

---

## Architecture

The system is organized into three distinct layers:

### 1. Control Plane
- Process lifecycle management
- Server configuration
- Backend selection

### 2. Data Plane
- Custom TCP protocol with binary framing
- epoll-driven I/O for efficient multi-client handling
- Backpressure control mechanisms
- Minimal abstraction overhead

### 3. Inference Runtime
- CUDA-backed llama.cpp engine
- GGUF quantized models
- Optimized for GPUs with ≤1 GB VRAM

This clean separation enables future extensions to RDMA or GPU-aware transports without core modifications.

---

## Features

- **Custom TCP Protocol**: Binary framing designed for minimal latency and overhead
- **epoll-Based Concurrency**: Non-blocking socket I/O handling multiple clients efficiently
- **GPU-Aware Inference**: Direct CUDA integration optimized for constrained GPUs
- **Performance Metrics**: Latency tracking (p50/p95/p99) and throughput measurement
- **Low Abstraction**: Emphasis on predictable latency and explicit control

---

## Technology Stack

- **Language**: C++17/C++20
- **GPU**: CUDA 12.x
- **Networking**: epoll and POSIX sockets
- **Inference Backend**: llama.cpp (GGUF models)
- **Build System**: CMake
- **Platform**: Linux

---

## Build

```bash
# Configure the build
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release

# Build the project
cmake --build build -j
```

### Prerequisites

- CUDA Toolkit 12.x
- C++17/C++20 compatible compiler
- CMake 3.15+
- llama.cpp dependencies

---

## Run

### Starting the Server

```bash
./build/bin/llama_tcp_server \
    --model /path/to/model.gguf \
    --listen 0.0.0.0:8080
```

### Configuration Options

- `--model`: Path to GGUF quantized model file
- `--listen`: Address and port to bind the server (default: 0.0.0.0:8080)

---

## Performance

The system prioritizes:

- **Minimal Latency**: Direct GPU memory management and non-blocking I/O
- **Efficient Concurrency**: epoll-based event loop for multiple concurrent clients
- **Memory Efficiency**: Optimized for GPUs with limited VRAM (≤1 GB)

Performance metrics include:
- p50/p95/p99 latency percentiles
- Tokens per second throughput
- GPU memory utilization

---

## Design Philosophy

> "Explicit control over data movement"

The project emphasizes:

1. **Clean Separation**: Networking logic independent from inference runtime
2. **Low-Level Control**: Direct management of sockets, memory, and GPU operations
3. **Predictable Performance**: Minimal abstractions for consistent latency
4. **Extensibility**: Architecture ready for RDMA or GPU-direct transports

---

## Use Cases

- Research into low-latency LLM serving
- GPU-constrained environments (edge devices, older GPUs)
- Custom networking protocols for inference workloads
- Educational exploration of systems programming with CUDA

---

## Notes

This project demonstrates production-grade systems thinking for LLM infrastructure, focusing on the intersection of high-performance networking and GPU computing. It's particularly valuable for understanding how to build efficient inference services on resource-constrained hardware.

**Author**: Mohammad Waqas
