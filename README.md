# Waqas Muhammad — Product Engineer

[![GitHub Pages](https://img.shields.io/badge/GitHub%20Pages-Live-brightgreen)](https://waqasm86.github.io)
[![MkDocs Material](https://img.shields.io/badge/MkDocs-Material-blue)](https://squidfunk.github.io/mkdocs-material/)
[![PyPI - llcuda](https://img.shields.io/badge/PyPI-llcuda-blue)](https://pypi.org/project/llcuda/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Product-minded engineer building on-device AI tools for hardware people actually own.**

🌐 **Live Site**: [https://waqasm86.github.io](https://waqasm86.github.io)

---

## Overview

This repository hosts my personal engineering portfolio website, built with **MkDocs Material**. It showcases my work on making LLM inference accessible on legacy NVIDIA GPUs through zero-configuration tools and empirical engineering.

### Key Focus

**llcuda Ecosystem** - A unified solution for running modern LLMs on old GPUs:
- **llcuda**: Python package published to PyPI (zero-configuration LLM inference)
- **Ubuntu-Cuda-Llama.cpp-Executable**: Pre-built binaries eliminating compilation complexity
- Tested on GeForce 940M (1GB VRAM) - actual legacy hardware, not simulations
- Empirical performance data: ~15 tokens/second on 2014-era GPU

### Philosophy

- **Build for Real Hardware**: GeForce 940M baseline, not RTX 4090
- **Zero Configuration**: `pip install llcuda` and done
- **Empirical Testing**: All benchmarks on actual hardware
- **Production Quality**: Published to PyPI, not just GitHub repos
- **Documentation First**: Comprehensive guides and examples

---

## Featured Project

### 🚀 [llcuda](https://waqasm86.github.io/llcuda/)

**Python package for LLM inference on legacy NVIDIA GPUs**

- **PyPI**: [pypi.org/project/llcuda](https://pypi.org/project/llcuda/)
- **GitHub**: [github.com/waqasm86/llcuda](https://github.com/waqasm86/llcuda)
- **Documentation**: [waqasm86.github.io/llcuda](https://waqasm86.github.io/llcuda/)

**Key Features:**
- Zero-configuration installation (no CUDA toolkit, no compilation)
- Legacy GPU support (GeForce 940M tested)
- JupyterLab-first design for data science workflows
- ~15 tokens/second on 1GB VRAM GPU
- Published to PyPI with semantic versioning

**Quick Start:**
```bash
pip install llcuda
python -m llcuda
```

### 📦 [Ubuntu-Cuda-Llama.cpp-Executable](https://waqasm86.github.io/ubuntu-cuda-executable/)

**Pre-built llama.cpp binary with CUDA 12.6 support**

- **GitHub**: [github.com/waqasm86/Ubuntu-Cuda-Llama.cpp-Executable](https://github.com/waqasm86/Ubuntu-Cuda-Llama.cpp-Executable)
- **Documentation**: [waqasm86.github.io/ubuntu-cuda-executable](https://waqasm86.github.io/ubuntu-cuda-executable/)

**Key Features:**
- Statically-linked binary (no external dependencies)
- Optimized for Maxwell architecture (compute 5.0)
- Eliminates need for CUDA toolkit installation
- Foundation that powers llcuda

---

## Repository Structure

```
waqasm86.github.io/
├── docs/                               # Documentation source files
│   ├── index.md                        # Homepage
│   ├── about.md                        # About page
│   ├── contact.md                      # Contact information
│   ├── llcuda/                         # llcuda documentation
│   │   ├── index.md                    # Overview
│   │   ├── quickstart.md               # 5-minute setup guide
│   │   ├── installation.md             # Comprehensive installation
│   │   ├── performance.md              # Empirical benchmarks
│   │   └── examples.md                 # Production code samples
│   ├── ubuntu-cuda-executable/         # Binary documentation
│   │   └── index.md                    # Pre-built binary guide
│   └── resume/                         # Resume files
│       └── README.md                   # Resume placeholder
├── .github/
│   └── workflows/
│       └── ci.yml                      # GitHub Actions deployment
├── mkdocs.yml                          # MkDocs configuration
├── requirements.txt                    # Python dependencies
└── README.md                           # This file
```

---

## Building Locally

### Prerequisites

```bash
# Python 3.8+
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

The dev server automatically reloads when you edit documentation files.

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
2. **GitHub Actions** triggers `.github/workflows/ci.yml`
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

### Adding New Content

1. **Create documentation file**:
   ```bash
   touch docs/new-section/new-page.md
   ```

2. **Write documentation** using Markdown with Material theme extensions

3. **Update navigation** in `mkdocs.yml`:
   ```yaml
   nav:
     - Home: index.md
     - New Section:
         - Page: new-section/new-page.md
   ```

4. **Commit and push**:
   ```bash
   git add docs/new-section/new-page.md mkdocs.yml
   git commit -m "Add new documentation page"
   git push origin main
   ```

### Markdown Features

MkDocs Material supports:

- **Admonitions**: `!!! note`, `!!! warning`, `!!! tip`, `!!! success`
- **Code blocks**: Triple backticks with syntax highlighting
- **Tables**: GitHub-flavored markdown tables
- **Icons**: FontAwesome, Material Design icons
- **Buttons**: `[Text](link){ .md-button }`
- **Task lists**: `- [x] Completed task`
- **Collapsible sections**: `??? question "Title"`

---

## MkDocs Configuration

### Current Theme Settings

```yaml
theme:
  name: material
  palette:
    - scheme: default          # Light mode
      primary: indigo
      accent: indigo
      toggle:
        icon: material/brightness-7
        name: Switch to dark mode
    - scheme: slate            # Dark mode
      primary: indigo
      accent: indigo
      toggle:
        icon: material/brightness-4
        name: Switch to light mode
  features:
    - navigation.tabs          # Top-level navigation tabs
    - navigation.sections      # Expandable sections
    - navigation.expand        # Auto-expand sections
    - navigation.top           # Back to top button
    - navigation.tracking      # Anchor tracking
    - toc.integrate            # TOC in sidebar
    - search.suggest           # Search suggestions
    - search.highlight         # Highlight search results
    - search.share             # Share search results
    - content.code.copy        # Copy code button
    - content.code.annotate    # Code annotations
```

### Navigation Structure

```yaml
nav:
  - Home: index.md
  - llcuda:
      - Overview: llcuda/index.md
      - Quick Start: llcuda/quickstart.md
      - Installation: llcuda/installation.md
      - Performance: llcuda/performance.md
      - Examples: llcuda/examples.md
  - Ubuntu CUDA Executable: ubuntu-cuda-executable/index.md
  - About:
      - About Me: about.md
      - Resume: resume/Muhammad_Waqas_Resume_2025.pdf
      - Contact: contact.md
```

---

## Technologies Used

### Documentation

- **[MkDocs](https://www.mkdocs.org/)** - Static site generator for project documentation
- **[Material for MkDocs](https://squidfunk.github.io/mkdocs-material/)** - Modern, responsive theme
- **[Markdown](https://www.markdownguide.org/)** - Lightweight markup language
- **[PyMdown Extensions](https://facelessuser.github.io/pymdown-extensions/)** - Enhanced Markdown

### Deployment

- **[GitHub Pages](https://pages.github.com/)** - Free static site hosting
- **[GitHub Actions](https://github.com/features/actions)** - CI/CD automation
- **[Python 3.8+](https://www.python.org/)** - Runtime environment

### Content

- **Python** - PyPI packaging, library design
- **CUDA** - GPU acceleration for legacy hardware
- **C++** - llama.cpp integration and optimization
- **CMake** - Build systems and static linking

---

## Statistics

- **Total documentation**: ~100KB across 10 files
- **Projects documented**: llcuda ecosystem (2 projects)
- **Code examples**: 50+ production-ready examples
- **Performance data**: Comprehensive benchmarks on GeForce 940M
- **Build time**: ~10 seconds for full site generation
- **Deploy time**: 2-3 minutes from push to live

---

## Why This Portfolio?

This portfolio demonstrates a **product-minded engineering approach**:

### ✅ Real Hardware Testing
- All benchmarks on GeForce 940M (1GB VRAM from 2014)
- No theoretical performance claims
- Honest about limitations and trade-offs

### ✅ Production Quality
- Published to PyPI with semantic versioning
- Comprehensive documentation (quick start, installation, performance, examples)
- Zero-configuration design (no manual compilation)

### ✅ Empirical Methodology
- Measured performance: ~15 tokens/second
- Real-world use cases: JupyterLab, data analysis, code generation
- Reproducible benchmarks with provided scripts

### ✅ User-Centric Design
- 5-minute quick start guide
- Detailed troubleshooting section
- Production-ready code examples
- Active maintenance and support

---

## Adding Your Resume

To complete the site, add your resume PDF:

```bash
# Copy your resume to the resume directory
cp /path/to/your/resume.pdf docs/resume/Muhammad_Waqas_Resume_2025.pdf

# Commit and push
git add docs/resume/Muhammad_Waqas_Resume_2025.pdf
git commit -m "Add resume PDF"
git push origin main
```

The navigation is already configured to link to `resume/Muhammad_Waqas_Resume_2025.pdf`.

---

## Contributing

This is a personal portfolio repository. However, if you notice any issues or have suggestions:

1. **Open an issue** describing the problem or suggestion
2. **Submit a pull request** with documentation improvements
3. **Share feedback** on project approach or documentation

---

## License

This portfolio website is licensed under the **MIT License**. See [LICENSE](LICENSE) for details.

Individual projects have their own licenses:
- **llcuda**: MIT License - [github.com/waqasm86/llcuda](https://github.com/waqasm86/llcuda)
- **Ubuntu-Cuda-Llama.cpp-Executable**: MIT License - [github.com/waqasm86/Ubuntu-Cuda-Llama.cpp-Executable](https://github.com/waqasm86/Ubuntu-Cuda-Llama.cpp-Executable)

---

## Contact

**Waqas Muhammad**

- **Email**: [waqasm86@gmail.com](mailto:waqasm86@gmail.com)
- **GitHub**: [@waqasm86](https://github.com/waqasm86)
- **PyPI**: [pypi.org/project/llcuda](https://pypi.org/project/llcuda/)
- **Portfolio**: [waqasm86.github.io](https://waqasm86.github.io)

---

## Acknowledgments

- **[MkDocs Material](https://squidfunk.github.io/mkdocs-material/)** - Excellent documentation framework
- **[GitHub Pages](https://pages.github.com/)** - Free, reliable hosting
- **[llama.cpp](https://github.com/ggerganov/llama.cpp)** - Efficient LLM inference by Georgi Gerganov
- **[NVIDIA CUDA](https://developer.nvidia.com/cuda-toolkit)** - GPU acceleration framework
- **[Dan McCreary](https://dmccreary.medium.com/)** - Inspiration for clean documentation design

---

## Quick Links

**Live Site**: [waqasm86.github.io](https://waqasm86.github.io)
**llcuda Documentation**: [waqasm86.github.io/llcuda](https://waqasm86.github.io/llcuda/)
**llcuda on PyPI**: [pypi.org/project/llcuda](https://pypi.org/project/llcuda/)
**GitHub Profile**: [github.com/waqasm86](https://github.com/waqasm86)

---

**Built with empirical rigor and product discipline for accessible on-device AI.**

*Last updated: December 2024*
