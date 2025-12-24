# cuda-openmpi

A comprehensive testing framework to validate NVIDIA CUDA and OpenMPI integration on Linux systems.

[:fontawesome-brands-github: View on GitHub](https://github.com/waqasm86/cuda-openmpi){ .md-button }

---

## Overview

This repository provides an integration test suite that verifies the proper configuration of GPU computing environments for parallel computing applications. It specifically tests CUDA-aware MPI capabilities, which enable direct GPU-to-GPU communication without intermediate host memory transfers.

**Purpose:**

Validate that your CUDA toolkit and OpenMPI installation are correctly configured and can work together for distributed GPU computing workloads.

---

## Test Coverage

The suite evaluates three core areas:

### 1. Basic CUDA Operations
- GPU device detection and enumeration
- Memory allocation and deallocation on device
- Kernel execution (vector addition test)
- Host-to-device and device-to-host data transfers
- Result validation

### 2. Standard MPI Communication
- Process initialization and rank assignment
- Host-to-host data transfers between MPI ranks
- Data integrity verification across multiple processes
- Inter-process communication patterns

### 3. CUDA-aware MPI
- Direct GPU-to-GPU communication capabilities
- Automatic detection of CUDA support in OpenMPI
- Performance comparison with host-staged transfers

---

## System Specifications

The test suite has been validated on:

- **CUDA Version**: 12.8
- **OpenMPI Version**: 5.0.6
- **Test GPU**: NVIDIA GeForce 940M
  - 1GB VRAM
  - Compute Capability 5.0
- **Operating System**: Xubuntu 22.04

---

## Project Structure

```
cuda-openmpi/
├── cuda_mpi_test.cu        # Main CUDA/MPI test program
├── CMakeLists.txt          # CMake build configuration
├── Makefile.direct         # Alternative direct build
├── build_and_run.sh        # Interactive build/test script
├── quick_test.sh           # Automated quick test
├── logs/                   # Test execution logs
└── README.md               # Documentation
```

### Key Files

- **cuda_mpi_test.cu**: Comprehensive test program covering CUDA operations, MPI communication, and CUDA-aware MPI
- **build_and_run.sh**: Interactive script for building and testing with multiple configurations
- **quick_test.sh**: Automated testing for CI/CD pipelines

---

## Dependencies

- **NVIDIA CUDA Toolkit** (12.8+)
- **OpenMPI** (5.0.6+)
- **CMake** (3.15+)
- **C++ compiler** with CUDA support (nvcc)
- **Linux environment**

---

## Build

### Using CMake (Recommended)

```bash
# Create build directory
mkdir -p build && cd build

# Configure the project
cmake ..

# Compile
make
```

Expected build output confirms:
- MPI detection and configuration
- CUDA toolkit version (12.8)
- Target GPU architecture (Compute Capability 5.0)

### Using Direct Makefile

```bash
# Alternative build method
make -f Makefile.direct
```

---

## Run

### Test CUDA Only (Single Process)

```bash
./cuda_mpi_test
```

This runs basic CUDA operations without MPI communication.

### Test CUDA + MPI (Multiple Processes)

```bash
# Run with 2 MPI ranks
mpirun -np 2 ./cuda_mpi_test

# Run with 4 MPI ranks
mpirun -np 4 ./cuda_mpi_test
```

### Check CUDA-aware MPI Support

```bash
# Verify if OpenMPI was built with CUDA support
ompi_info --parsable --all | grep mpi_built_with_cuda_support
```

If the output shows `false`, your OpenMPI installation uses the standard data flow:
```
GPU → Host Memory → MPI Transfer → Host Memory → GPU
```

---

## Test Output

Successful execution displays:

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
GPU-to-GPU communication test PASSED
```

---

## Architecture Notes

### Standard vs CUDA-aware MPI

**Standard MPI** (most common):
```
Process A GPU → Host Memory → MPI Send → Host Memory → Process B GPU
```

**CUDA-aware MPI** (requires special OpenMPI build):
```
Process A GPU ────────── MPI Send ──────────→ Process B GPU
```

CUDA-aware MPI eliminates intermediate host copies, reducing latency and improving throughput for GPU-intensive parallel applications.

### Memory Considerations

With 1GB VRAM on the test GPU:
- Safe allocation per process: <400MB
- Test suite memory usage: ~12KB per process
- Suitable for resource-constrained hardware testing

---

## Performance

The test suite uses minimal resources to ensure compatibility with constrained hardware:

- **Vector Size**: 3 elements (minimal test)
- **Memory per Process**: ~12KB GPU memory
- **Scalability**: Tested up to 4 MPI ranks on single GPU

This conservative approach ensures the tests run on older or low-VRAM GPUs like the GeForce 940M.

---

## Troubleshooting

### MPI Not Found

```bash
# Install OpenMPI on Ubuntu/Debian
sudo apt-get install libopenmpi-dev openmpi-bin
```

### CUDA Not Found

Ensure CUDA toolkit is installed and `nvcc` is in your PATH:
```bash
nvcc --version
```

### GPU Memory Errors

Reduce the test vector size in `cuda_mpi_test.cu` if running on GPUs with <1GB VRAM.

---

## Use Cases

- **Environment Validation**: Verify CUDA + MPI setup before running production workloads
- **CI/CD Testing**: Automated verification of GPU cluster configurations
- **Educational**: Learn CUDA-MPI integration patterns
- **Benchmarking**: Baseline performance testing for distributed GPU applications

---

## Future Enhancements

- Support for multi-GPU systems
- Performance benchmarking suite
- Advanced CUDA-aware MPI patterns
- Integration with NCCL for multi-GPU communication

---

## Notes

This test suite is essential for validating distributed GPU computing environments. While most OpenMPI installations don't include CUDA-aware support by default, the test suite gracefully handles both scenarios and provides clear feedback on available capabilities.

**Author**: Mohammad Waqas
