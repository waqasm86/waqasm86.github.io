# cuda-openmpi

A comprehensive testing framework validating NVIDIA CUDA and OpenMPI integration for distributed GPU computing environments.

[:fontawesome-brands-github: View on GitHub](https://github.com/waqasm86/cuda-openmpi){ .md-button }

---

## Overview

**cuda-openmpi** provides an integration test suite that verifies proper configuration of GPU computing environments for parallel computing applications. It specifically tests CUDA-aware MPI capabilities, enabling direct GPU-to-GPU communication without intermediate host memory transfers—critical for high-performance distributed GPU applications.

**Purpose**: Validate that CUDA Toolkit and OpenMPI installations are correctly configured and can work together for distributed GPU computing workloads before running production applications.

**Key Testing Areas**:

- **Basic CUDA Operations**: GPU detection, memory allocation, kernel execution, data transfers
- **Standard MPI Communication**: Process initialization, host-to-host transfers, data integrity
- **CUDA-Aware MPI**: Direct GPU-to-GPU communication detection and testing
- **Environment Validation**: Automated verification of build configuration and runtime capabilities
- **Educational Patterns**: Demonstrating fundamental CUDA-MPI integration techniques

This framework is essential for validating HPC cluster configurations, CI/CD pipelines for GPU applications, and educational environments teaching distributed GPU computing.

---

## System Architecture

### Test Suite Design

```
┌────────────────────────────────────────────────────────────────┐
│                    TEST LAYER 1: BASIC CUDA                    │
│  ┌──────────────────────────────────────────────────────┐     │
│  │  GPU Device Detection                                │     │
│  │  • Enumerate available CUDA devices                  │     │
│  │  • Query device properties (CC, VRAM, cores)         │     │
│  │  • Verify CUDA runtime initialization                │     │
│  └──────────────────────────────────────────────────────┘     │
│  ┌──────────────────────────────────────────────────────┐     │
│  │  Memory Operations                                    │     │
│  │  • Host allocation (malloc)                           │     │
│  │  • Device allocation (cudaMalloc)                     │     │
│  │  • Host-to-device transfer (cudaMemcpy H2D)          │     │
│  │  • Device-to-host transfer (cudaMemcpy D2H)          │     │
│  └──────────────────────────────────────────────────────┘     │
│  ┌──────────────────────────────────────────────────────┐     │
│  │  Kernel Execution                                     │     │
│  │  • Vector addition kernel (test computation)         │     │
│  │  • Result validation                                  │     │
│  │  • Error handling                                     │     │
│  └──────────────────────────────────────────────────────┘     │
└────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────┐
│                 TEST LAYER 2: STANDARD MPI                     │
│  ┌──────────────────────────────────────────────────────┐     │
│  │  Process Initialization                              │     │
│  │  • MPI_Init                                           │     │
│  │  • MPI_Comm_rank (get process ID)                    │     │
│  │  • MPI_Comm_size (get process count)                 │     │
│  └──────────────────────────────────────────────────────┘     │
│  ┌──────────────────────────────────────────────────────┐     │
│  │  Host-to-Host Communication                          │     │
│  │  • MPI_Send (rank 0 → rank 1)                        │     │
│  │  • MPI_Recv (rank 1 ← rank 0)                        │     │
│  │  • Data integrity verification                        │     │
│  └──────────────────────────────────────────────────────┘     │
└────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────┐
│                TEST LAYER 3: CUDA-AWARE MPI                    │
│  ┌──────────────────────────────────────────────────────┐     │
│  │  GPU-to-GPU Communication Detection                  │     │
│  │  • Query OpenMPI for CUDA support                    │     │
│  │  • Attempt direct device pointer MPI_Send/Recv       │     │
│  │  • Fallback to host-staged transfers if not supported│     │
│  └──────────────────────────────────────────────────────┘     │
│  ┌──────────────────────────────────────────────────────┐     │
│  │  Performance Comparison (if supported)               │     │
│  │  • Direct GPU-to-GPU latency                         │     │
│  │  • Host-staged latency                               │     │
│  │  • Bandwidth measurement                             │     │
│  └──────────────────────────────────────────────────────┘     │
└────────────────────────────────────────────────────────────────┘
```

### Communication Patterns

**Standard MPI** (most common):
```
Process A                     Process B
    │                             │
    ├─ GPU Memory                 ├─ GPU Memory
    │  [data]                     │  [empty]
    ▼                             ▼
cudaMemcpy(D2H)             (waiting)
    │                             │
    ├─ Host Memory                ├─ Host Memory
    │  [data]                     │  [empty]
    ▼                             ▼
MPI_Send(host_ptr) ───────► MPI_Recv(host_ptr)
    │                             │
    │                             ▼
    │                       cudaMemcpy(H2D)
    │                             │
    │                             ├─ GPU Memory
    │                             │  [data]
```

**CUDA-Aware MPI** (requires special build):
```
Process A                     Process B
    │                             │
    ├─ GPU Memory                 ├─ GPU Memory
    │  [data]                     │  [empty]
    ▼                             ▼
MPI_Send(device_ptr) ──────► MPI_Recv(device_ptr)
    │                             │
    │                             ├─ GPU Memory
    │                             │  [data]
(No host staging required)
```

**Benefits of CUDA-Aware MPI**:
1. **Lower Latency**: Eliminates D2H and H2D copies
2. **Higher Bandwidth**: Direct GPU memory access
3. **Simpler Code**: No manual staging required
4. **Better Scalability**: Reduced memory pressure on host

---

## Test Coverage

### Test 1: Basic CUDA Operations

**What It Tests**:
```cpp
// GPU detection
int device_count;
cudaGetDeviceCount(&device_count);
// Verify: device_count > 0

// Memory allocation
float *d_a, *d_b, *d_c;
cudaMalloc(&d_a, size);
cudaMalloc(&d_b, size);
cudaMalloc(&d_c, size);
// Verify: All allocations succeed

// Data transfer
cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
// Verify: No errors

// Kernel execution
vector_add<<<grid, block>>>(d_a, d_b, d_c, N);
cudaDeviceSynchronize();
// Verify: No kernel errors

// Result retrieval
cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);
// Verify: h_c contains correct results
```

**Success Criteria**:
- ✅ At least 1 CUDA device detected
- ✅ Memory allocations succeed
- ✅ Kernel executes without errors
- ✅ Results match expected values

### Test 2: Standard MPI Communication

**What It Tests**:
```cpp
// Initialize MPI
MPI_Init(&argc, &argv);
int rank, size;
MPI_Comm_rank(MPI_COMM_WORLD, &rank);
MPI_Comm_size(MPI_COMM_WORLD, &size);
// Verify: rank and size correct

if (rank == 0) {
  // Send data
  int data = 42;
  MPI_Send(&data, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);
} else if (rank == 1) {
  // Receive data
  int data;
  MPI_Recv(&data, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  // Verify: data == 42
}
```

**Success Criteria**:
- ✅ MPI initialized successfully
- ✅ Correct rank assignment
- ✅ Data transmitted intact
- ✅ No communication errors

### Test 3: CUDA-Aware MPI

**What It Tests**:
```cpp
// Check if CUDA-aware MPI is available
#ifdef MPIX_CUDA_AWARE_SUPPORT
  bool cuda_aware = true;
#else
  bool cuda_aware = false;
#endif

if (cuda_aware) {
  // Direct GPU-to-GPU transfer
  if (rank == 0) {
    MPI_Send(d_data, count, MPI_FLOAT, 1, 0, MPI_COMM_WORLD);
  } else if (rank == 1) {
    MPI_Recv(d_data, count, MPI_FLOAT, 0, 0, MPI_COMM_WORLD,
             MPI_STATUS_IGNORE);
  }
} else {
  // Host-staged transfer
  if (rank == 0) {
    cudaMemcpy(h_data, d_data, size, cudaMemcpyDeviceToHost);
    MPI_Send(h_data, count, MPI_FLOAT, 1, 0, MPI_COMM_WORLD);
  } else if (rank == 1) {
    MPI_Recv(h_data, count, MPI_FLOAT, 0, 0, MPI_COMM_WORLD,
             MPI_STATUS_IGNORE);
    cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice);
  }
}
```

**Success Criteria**:
- ✅ CUDA awareness correctly detected
- ✅ GPU-to-GPU transfer succeeds (if supported)
- ✅ Host-staged transfer works (fallback)
- ✅ Data integrity maintained

---

## System Specifications

### Tested Environment

**Hardware**:
- GPU: NVIDIA GeForce 940M
  - Compute Capability: 5.0
  - VRAM: 1 GB
  - CUDA Cores: 640
- CPU: Intel Core i5 (4 threads)
- RAM: 8 GB

**Software**:
- OS: Xubuntu 22.04 LTS
- CUDA: 12.8
- OpenMPI: 5.0.6
- GCC: 11.4
- CMake: 3.22

**Configuration**:
- Vector size: 3 elements (minimal test)
- Memory per process: ~12 KB GPU memory
- Scalability: Tested up to 4 MPI ranks on single GPU

---

## Build & Run

### Prerequisites

**Required**:
- NVIDIA CUDA Toolkit (12.x+)
- OpenMPI (5.x+)
- CMake (3.15+)
- C++ compiler with CUDA support (nvcc)
- Linux environment

### Install Dependencies (Ubuntu/Debian)

```bash
# CUDA Toolkit (if not installed)
# Follow: https://developer.nvidia.com/cuda-downloads

# OpenMPI
sudo apt update
sudo apt install libopenmpi-dev openmpi-bin

# CMake
sudo apt install cmake

# Verify installations
nvcc --version
mpirun --version
cmake --version
```

### Build

**Using CMake** (recommended):
```bash
# Create build directory
mkdir -p build && cd build

# Configure
cmake ..

# Build
make

# Expected output
# ...
# [100%] Built target cuda_mpi_test
```

**Using Direct Makefile**:
```bash
make -f Makefile.direct
```

### Run Tests

**Single Process** (CUDA only):
```bash
./cuda_mpi_test
```

**Expected Output**:
```
=== CUDA Basic Test ===
Found 1 CUDA device(s)
Using device 0: GeForce 940M
  Compute Capability: 5.0
  Total memory: 1024 MB
CUDA vector addition test PASSED
```

**Multi-Process** (CUDA + MPI):
```bash
# 2 ranks
mpirun -np 2 ./cuda_mpi_test

# 4 ranks
mpirun -np 4 ./cuda_mpi_test
```

**Expected Output**:
```
=== CUDA Basic Test ===
Found 1 CUDA device(s)
Using device 0: GeForce 940M
CUDA vector addition test PASSED

=== MPI Communication Test ===
Rank 0/2: Host-to-host MPI test PASSED
Rank 1/2: Host-to-host MPI test PASSED

=== CUDA-aware MPI Test ===
CUDA-aware MPI: Not supported (using host staging)
GPU-to-GPU communication test PASSED (via host staging)
```

### Verify CUDA-Aware MPI Support

```bash
# Check if OpenMPI was built with CUDA support
ompi_info --parsable --all | grep mpi_built_with_cuda_support

# If output contains:
# mca:mpi:base:param:mpi_built_with_cuda_support:value:true
# → CUDA-aware MPI is available

# If false or not found:
# → Standard MPI only (host staging required)
```

---

## Performance Considerations

### Memory Usage

**Conservative Design**:
- Vector size: 3 elements (12 bytes)
- Safe for multi-process on single GPU
- Avoids OOM on 1GB VRAM

**Scalability Test**:
```bash
# Test with different vector sizes
# Edit cuda_mpi_test.cu: const int N = 1024;
# Rebuild and run

# Monitor GPU memory
watch -n 1 nvidia-smi
```

### Multi-Process Considerations

**Running Multiple Ranks on Single GPU**:
- Each rank allocates independent GPU memory
- Total memory = N_ranks × per_rank_memory
- CUDA runtime handles concurrent access
- Use CUDA streams for better parallelism (advanced)

**Example Memory Calculation**:
```
Vector size: 1024 elements
Data type: float (4 bytes)
Arrays: 3 (a, b, c)

Per rank: 1024 × 4 bytes × 3 = 12 KB
For 4 ranks: 12 KB × 4 = 48 KB
```

---

## Use Cases

### 1. HPC Cluster Validation

**Scenario**: Verify GPU nodes before production deployment

**Workflow**:
```bash
# On each compute node
srun -N 1 -n 4 ./cuda_mpi_test

# Aggregate results across cluster
srun -N 8 -n 32 ./cuda_mpi_test

# Check for failures
# All nodes should show: PASSED
```

**Benefits**:
- Early detection of misconfigured nodes
- Validates MPI fabric
- Tests GPU accessibility across nodes

### 2. CI/CD Pipeline Integration

**Scenario**: Automated testing in GPU CI/CD

**.gitlab-ci.yml Example**:
```yaml
test_cuda_mpi:
  tags: [gpu, nvidia]
  script:
    - module load cuda/12.8 openmpi/5.0.6
    - mkdir build && cd build
    - cmake ..
    - make
    - ./cuda_mpi_test
    - mpirun -np 4 ./cuda_mpi_test
  artifacts:
    reports:
      junit: test_results.xml
```

### 3. Educational

**Scenario**: Teach CUDA + MPI fundamentals

**Learning Outcomes**:
- CUDA programming basics (kernels, memory management)
- MPI communication patterns (send/recv, collectives)
- Hybrid CPU-GPU programming
- Distributed GPU computing concepts

**Hands-On Exercises**:
1. Modify kernel to perform matrix multiplication
2. Implement MPI_Bcast for parameter distribution
3. Add performance timing for different transfer methods
4. Explore CUDA streams with MPI

---

## Troubleshooting

### MPI Not Found

**Symptoms**:
```
CMake Error: Could not find MPI
```

**Solution**:
```bash
# Ubuntu/Debian
sudo apt install libopenmpi-dev openmpi-bin

# Verify installation
which mpirun
mpirun --version
```

### CUDA Not Found

**Symptoms**:
```
nvcc: command not found
```

**Solution**:
```bash
# Ensure CUDA is installed
nvidia-smi  # Should show GPU info

# Add CUDA to PATH
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Verify
nvcc --version
```

### GPU Memory Errors

**Symptoms**:
```
CUDA error: out of memory
```

**Solution**:
```bash
# Reduce vector size in cuda_mpi_test.cu
const int N = 3;  // Minimal size

# Reduce number of MPI ranks
mpirun -np 2 ./cuda_mpi_test  # Instead of 4
```

### Compute Capability Mismatch

**Symptoms**:
```
CUDA error: no kernel image available for device
```

**Solution**:
```bash
# Find your GPU's compute capability
nvidia-smi --query-gpu=compute_cap --format=csv,noheader

# Edit CMakeLists.txt or build command
set(CMAKE_CUDA_ARCHITECTURES "50")  # For CC 5.0

# Rebuild
rm -rf build && mkdir build && cd build && cmake .. && make
```

---

## Advanced Topics

### Building CUDA-Aware OpenMPI

**Why**: Enable direct GPU-to-GPU communication

**Steps**:
```bash
# Download OpenMPI source
wget https://download.open-mpi.org/release/open-mpi/v5.0/openmpi-5.0.6.tar.gz
tar -xzf openmpi-5.0.6.tar.gz
cd openmpi-5.0.6

# Configure with CUDA support
./configure --prefix=/opt/openmpi-cuda \
  --with-cuda=/usr/local/cuda \
  --enable-mpi-cxx

# Build
make -j$(nproc)

# Install
sudo make install

# Update environment
export PATH=/opt/openmpi-cuda/bin:$PATH
export LD_LIBRARY_PATH=/opt/openmpi-cuda/lib:$LD_LIBRARY_PATH

# Verify
ompi_info --parsable --all | grep cuda
# Should show: mpi_built_with_cuda_support:value:true
```

### Performance Benchmarking

**Goal**: Measure CUDA-aware vs. host-staged performance

**Benchmark Code**:
```cpp
#include <chrono>

auto start = std::chrono::high_resolution_clock::now();

// Transfer method (CUDA-aware or host-staged)
perform_transfer();

auto end = std::chrono::high_resolution_clock::now();
auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

if (rank == 0) {
  std::cout << "Transfer latency: " << duration.count() << " μs\n";
}
```

**Expected Results** (example):
```
CUDA-aware MPI:   ~50 μs
Host-staged MPI:  ~150 μs
Speedup: 3x
```

---

## Related Projects

**By the same author**:

1. **[cuda-tcp-llama.cpp](https://github.com/waqasm86/cuda-tcp-llama.cpp)**
   - Custom TCP inference data plane
   - Integration point: Multi-rank coordination

2. **[cuda-mpi-llama-scheduler](https://github.com/waqasm86/cuda-mpi-llama-scheduler)**
   - Multi-rank LLM inference
   - Integration point: MPI communication patterns

3. **[cuda-llm-storage-pipeline](https://github.com/waqasm86/cuda-llm-storage-pipeline)**
   - Distributed artifact storage
   - Integration point: Multi-node artifact distribution

---

## Future Enhancements

**Planned Features**:
- Multi-GPU system support
- Advanced CUDA-aware MPI patterns (collectives)
- Performance benchmarking suite
- Integration with NCCL
- Automated test reporting

---

## Technical Specifications

| **Aspect** | **Details** |
|------------|-------------|
| **GPU** | GeForce 940M (CC 5.0, 1GB VRAM) |
| **CUDA** | 12.8 |
| **OpenMPI** | 5.0.6 |
| **Platform** | Linux (Ubuntu 22.04) |
| **Test Vector Size** | 3 elements (minimal) |
| **Memory per Rank** | ~12 KB GPU |
| **Max Ranks Tested** | 4 on single GPU |

---

## Quick Reference

```bash
# Build
mkdir build && cd build && cmake .. && make

# Run (single process)
./cuda_mpi_test

# Run (multi-process)
mpirun -np 2 ./cuda_mpi_test

# Check CUDA-aware MPI
ompi_info --parsable --all | grep mpi_built_with_cuda_support
```

---

## Contact

- **Author**: Mohammad Waqas
- **GitHub**: [waqasm86](https://github.com/waqasm86)
- **Repository**: [cuda-openmpi](https://github.com/waqasm86/cuda-openmpi)

---

**This test suite is essential for validating distributed GPU computing environments, ensuring CUDA and OpenMPI work together correctly before deploying production workloads. While most OpenMPI installations don't include CUDA-aware support by default, the suite gracefully handles both scenarios and provides clear feedback on available capabilities.**
