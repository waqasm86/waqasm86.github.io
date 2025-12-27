# CUDA Projects

This section contains documentation, design notes, build steps, and results for my CUDA repositories.

## Featured Projects

### [cuda-nvidia-systems-engg](cuda-nvidia-systems-engg.md)
**Production-grade distributed LLM inference system** - Unified C++20/CUDA platform combining TCP networking (epoll), MPI scheduling (work-stealing), content-addressed storage (SeaweedFS), and empirical performance research (p50/p95/p99 latencies). 3200+ LOC demonstrating systems engineering expertise for on-device AI.

**Technologies**: C++20, CUDA 17, OpenMPI, Epoll, SeaweedFS, CMake

---

## Individual Components

The following projects were later unified into cuda-nvidia-systems-engg:

### [local-llama-cuda](local-llama-cuda.md)
Custom CUDA implementation with MPI-based distributed inference

### [cuda-tcp-llama.cpp](cuda-tcp-llama.md)
High-performance TCP inference gateway with epoll async I/O

### [cuda-openmpi](cuda-openmpi.md)
CUDA-aware OpenMPI integration and testing

### [cuda-mpi-llama-scheduler](cuda-mpi-llama-scheduler.md)
Distributed scheduler with work-stealing and percentile analysis

### [cuda-llm-storage-pipeline](cuda-llm-storage-pipeline.md)
Content-addressed model distribution with SHA256 verification
