# Waqas Muhammad — CUDA Projects Portfolio

[![GitHub Pages](https://img.shields.io/badge/GitHub%20Pages-Live-brightgreen)](https://waqasm86.github.io)
[![MkDocs Material](https://img.shields.io/badge/MkDocs-Material-blue)](https://squidfunk.github.io/mkdocs-material/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Personal engineering portfolio showcasing production-grade CUDA + C++ + LLM inference projects**

🌐 **Live Site**: [https://waqasm86.github.io](https://waqasm86.github.io)

---

## Overview

This repository hosts my personal engineering portfolio website, built with **MkDocs Material**. It features comprehensive documentation, design notes, build steps, and performance benchmarks for my CUDA/C++ systems engineering projects focused on on-device LLM inference.

### Key Focus Areas

- **Production-grade distributed systems** - TCP networking, MPI scheduling, content-addressed storage
- **Empirical performance research** - Percentile latencies (p50/p95/p99), ablation studies, throughput benchmarks
- **On-device AI optimization** - Works on constrained hardware (1GB VRAM GPUs)
- **Systems programming** - Modern C++20, CUDA 17, epoll async I/O, GPU kernel development

---

## Featured Projects

### 🚀 [cuda-nvidia-systems-engg](https://waqasm86.github.io/projects/cuda-nvidia-systems-engg/)

**Production-grade distributed LLM inference system** combining four specialized subsystems:

1. **TCP Gateway** - Epoll-based async I/O, binary protocol, streaming responses
2. **MPI Scheduler** - Work-stealing algorithm, multi-rank coordination, percentile analysis
3. **Content-Addressed Storage** - SHA256 deduplication, SeaweedFS integration, LRU caching
4. **CUDA Post-Processing** - GPU kernels, memory pooling, stream management

**Technologies**: C++20, CUDA 17, OpenMPI, Epoll, SeaweedFS, CMake
**Lines of Code**: 3200+ LOC demonstrating systems engineering expertise

**Performance** (GeForce 940M, 1GB VRAM):
- Gemma 2B Q4_K_M: 42 tok/s, P95 latency 1.2s
- Multi-rank scaling: 89% efficiency at 8 ranks

### 📦 Component Projects

The following projects were later unified into cuda-nvidia-systems-engg:

- **[local-llama-cuda](https://waqasm86.github.io/projects/local-llama-cuda/)** - Custom CUDA implementation with MPI-based distributed inference
- **[cuda-tcp-llama.cpp](https://waqasm86.github.io/projects/cuda-tcp-llama/)** - High-performance TCP inference gateway with epoll async I/O
- **[cuda-openmpi](https://waqasm86.github.io/projects/cuda-openmpi/)** - CUDA-aware OpenMPI integration and testing
- **[cuda-mpi-llama-scheduler](https://waqasm86.github.io/projects/cuda-mpi-llama-scheduler/)** - Distributed scheduler with work-stealing and percentile analysis
- **[cuda-llm-storage-pipeline](https://waqasm86.github.io/projects/cuda-llm-storage-pipeline/)** - Content-addressed model distribution with SHA256 verification

---

## Repository Structure

```
waqasm86.github.io/
├── docs/                          # Documentation source files
│   ├── index.md                   # Homepage
│   ├── about.md                   # About page
│   └── projects/                  # Project documentation
│       ├── index.md               # Projects overview
│       ├── cuda-nvidia-systems-engg.md    # Unified systems project (25KB)
│       ├── local-llama-cuda.md            # Custom CUDA implementation
│       ├── cuda-tcp-llama.md              # TCP gateway
│       ├── cuda-openmpi.md                # OpenMPI integration
│       ├── cuda-mpi-llama-scheduler.md    # MPI scheduler
│       └── cuda-llm-storage-pipeline.md   # Storage pipeline
├── .github/
│   └── workflows/
│       └── gh-pages.yml           # GitHub Actions deployment workflow
├── mkdocs.yml                     # MkDocs configuration
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

---

## Building Locally

### Prerequisites

```bash
# Python 3.11+
python3 --version

# pip package manager
pip --version
```

### Installation

```bash
# Clone repository
git clone https://github.com/waqasm86/waqasm86.github.io.git
cd waqasm86.github.io

# Install dependencies
pip install -r requirements.txt
```

### Local Development Server

```bash
# Start live-reloading dev server
mkdocs serve

# Open browser to http://127.0.0.1:8000
```

The dev server will automatically reload when you edit documentation files.

### Build Static Site

```bash
# Build static HTML site
mkdocs build

# Output in ./site/ directory
ls site/
```

---

## Deployment

This site is automatically deployed to **GitHub Pages** via GitHub Actions on every push to the `main` branch.

### Deployment Workflow

1. **Push changes** to `main` branch
2. **GitHub Actions** triggers `.github/workflows/gh-pages.yml`
3. **MkDocs builds** static site from `docs/` directory
4. **GitHub Pages** publishes to `https://waqasm86.github.io`
5. **Live in 2-3 minutes** after push

### Manual Deployment

```bash
# Build and deploy to gh-pages branch
mkdocs gh-deploy

# Force rebuild
mkdocs gh-deploy --force
```

---

## Documentation Guidelines

### Adding a New Project

1. **Create documentation file**:
   ```bash
   touch docs/projects/new-project.md
   ```

2. **Write documentation** using Markdown with Material theme extensions:
   ```markdown
   # Project Name

   Short description

   [:fontawesome-brands-github: View on GitHub](https://github.com/user/repo){ .md-button }

   ## Overview

   Detailed project overview...
   ```

3. **Add to projects index** in `docs/projects/index.md`:
   ```markdown
   ### [new-project](new-project.md)
   Brief description of the project
   ```

4. **Commit and push**:
   ```bash
   git add docs/projects/new-project.md docs/projects/index.md
   git commit -m "Add new-project documentation"
   git push origin main
   ```

### Markdown Features

MkDocs Material supports:

- **Admonitions**: `!!! note`, `!!! warning`, `!!! tip`
- **Code blocks**: Triple backticks with syntax highlighting
- **Tables**: GitHub-flavored markdown tables
- **Diagrams**: Mermaid diagrams (if enabled)
- **Icons**: FontAwesome, Material Design icons
- **Buttons**: `{ .md-button }` modifier
- **Task lists**: `- [x] Completed task`

---

## MkDocs Configuration

### Current Theme Settings

```yaml
theme:
  name: material
  features:
    - navigation.tabs       # Top-level navigation tabs
    - navigation.sections   # Expandable sections
    - toc.integrate         # Table of contents in sidebar
    - search.suggest        # Search suggestions
    - search.highlight      # Highlight search results
```

### Navigation Structure

```yaml
nav:
  - Home: index.md
  - CUDA Projects: projects/index.md
  - About: about.md
```

Projects are listed on the `projects/index.md` page, keeping the main navigation clean and focused.

---

## Technologies Used

### Documentation

- **[MkDocs](https://www.mkdocs.org/)** - Static site generator for project documentation
- **[Material for MkDocs](https://squidfunk.github.io/mkdocs-material/)** - Modern, responsive theme
- **[Markdown](https://www.markdownguide.org/)** - Lightweight markup language

### Deployment

- **[GitHub Pages](https://pages.github.com/)** - Free static site hosting
- **[GitHub Actions](https://github.com/features/actions)** - CI/CD automation
- **[Python 3.11+](https://www.python.org/)** - Runtime environment

### Content

- **C++20** - Modern systems programming
- **CUDA 17** - GPU acceleration and kernel development
- **OpenMPI** - Message passing for distributed computing
- **Epoll** - Scalable I/O event notification
- **SeaweedFS** - Distributed file system

---

## Statistics

- **Total documentation**: 240KB (9 files)
- **Projects documented**: 6 CUDA/C++ projects
- **Code examples**: 50+ usage examples, build instructions, benchmarks
- **Performance data**: Comprehensive benchmarks on GeForce 940M (1GB VRAM)
- **Build time**: ~30 seconds for full site generation
- **Deploy time**: 2-3 minutes from push to live

---

## Why This Portfolio?

This portfolio demonstrates **exactly** the skillset sought by companies building on-device AI infrastructure:

### ✅ Empirical Research Mindset
- Percentile analysis (p50/p95/p99) in all benchmarks
- Ablation studies on inflight queue depth, batch size, GPU layers
- Visualization-ready data export (CSV format)

### ✅ Production Systems Engineering
- 3200+ LOC of production C++20/CUDA code
- Advanced algorithms (work-stealing, epoll, content-addressing)
- Modern patterns (RAII, optional, string_view, concepts)

### ✅ On-Device AI Optimization
- Works on 1GB VRAM (GeForce 940M from 2014!)
- Quantization support (Q4_K_M, Q8_0)
- Layer offloading for memory-constrained GPUs

### ✅ Scientific Method
- Systematic ablation studies
- Hypothesis-driven experimentation
- Tail latency measurement (not just means!)

---

## Contributing

This is a personal portfolio repository. However, if you notice any issues or have suggestions:

1. **Open an issue** describing the problem or suggestion
2. **Submit a pull request** with documentation improvements
3. **Share feedback** on project architecture or approach

---

## License

This portfolio website is licensed under the **MIT License**. See [LICENSE](LICENSE) for details.

Individual projects may have their own licenses - check their respective repositories:

- [cuda-nvidia-systems-engg](https://github.com/waqasm86/cuda-nvidia-systems-engg) - MIT License
- Other projects - See individual repositories

---

## Contact

**Waqas Muhammad**

- **GitHub**: [@waqasm86](https://github.com/waqasm86)
- **Email**: waqasm86@gmail.com
- **Portfolio**: [waqasm86.github.io](https://waqasm86.github.io)

---

## Acknowledgments

- **[MkDocs Material](https://squidfunk.github.io/mkdocs-material/)** - Excellent documentation framework
- **[GitHub Pages](https://pages.github.com/)** - Free, reliable hosting
- **[NVIDIA CUDA](https://developer.nvidia.com/cuda-toolkit)** - GPU acceleration framework
- **[llama.cpp](https://github.com/ggerganov/llama.cpp)** - Efficient LLM inference engine

---

**Built with empirical rigor and production discipline for on-device AI research.**

*Last updated: December 2024*
