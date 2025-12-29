# About Me

**Product-minded engineer building on-device AI tools for hardware people actually own.**

---

## Philosophy

I build tools that **actually work** on real hardware, not just theoretical benchmarks on cutting-edge GPUs. My approach is rooted in empirical engineering: every claim is backed by measurements on actual hardware, and installation should be as simple as `pip install`.

**Core Beliefs:**

- **Build for Real Hardware**: GeForce 940M with 1GB VRAM is my baseline, not an afterthought
- **Zero-Configuration Design**: If it requires manual compilation or complex setup, I haven't finished the job
- **Empirical Testing**: Benchmarks on simulators don't count. Real hardware or it didn't happen
- **Production Quality**: Publish to PyPI, not just GitHub repos. Users deserve properly packaged software
- **Documentation as a Feature**: If users can't figure out how to use it, it doesn't exist

---

## Current Work

### llcuda - LLM Inference for Legacy GPUs

**[PyPI Package](https://pypi.org/project/llcuda/)** | **[Documentation](/llcuda/)** | **[GitHub](https://github.com/waqasm86/llcuda)** | **[v1.0.1 Release](https://github.com/waqasm86/llcuda/releases/tag/v1.0.1)**

PyTorch-style Python package that brings LLM inference to old NVIDIA GPUs with zero configuration. Published to PyPI, tested extensively on GeForce 940M (1GB VRAM).

**Key Achievements (v1.0.1):**
- Published production-ready package to PyPI with bundled CUDA 12.8 binaries
- Achieved ~15 tokens/second on GeForce 940M
- Zero-configuration installation - no manual path setup required
- Smart model loading with HuggingFace registry (11 curated models)
- Hardware auto-configuration based on VRAM detection
- Performance metrics tracking (P50/P95/P99 latency)
- JupyterLab-first design for data science workflows
- Comprehensive documentation and examples

**Technical Stack:**
- Python packaging and PyPI distribution (47 MB wheel with binaries)
- CUDA programming for legacy GPUs (compute capability 5.0)
- Bundled llama.cpp binaries and shared libraries
- Empirical performance testing and optimization
- Auto-configuration and hardware detection

---

## Technical Expertise

### Programming Languages
- **Python**: PyPI packaging, library design, API development
- **CUDA/C++**: GPU programming, performance optimization
- **Shell/Bash**: Build automation, deployment scripts

### Tools & Technologies
- **CUDA Development**: Programming for legacy GPUs (Maxwell architecture)
- **Build Systems**: CMake, static linking, cross-compilation
- **Package Management**: PyPI publishing, semantic versioning
- **Version Control**: Git, GitHub workflows, CI/CD
- **Documentation**: MkDocs, technical writing, developer experience

### Specializations
- **GPU Computing**: Optimizing for low-VRAM, legacy NVIDIA GPUs
- **Python Packaging**: Creating production-ready PyPI packages
- **Performance Testing**: Empirical benchmarking on real hardware
- **Developer Experience**: Zero-configuration tools, comprehensive docs
- **LLM Systems**: Integration with llama.cpp, model quantization, inference optimization

---

## Approach to Engineering

### 1. Product-Minded Development

I don't just write code; I build products that solve real problems for real users.

**Example**: llcuda isn't just a Python wrapper around llama.cpp. It's a complete solution that handles model downloading, GPU detection, error recovery, and provides a Jupyter-friendly API.

### 2. Empirical Testing

All performance claims are backed by measurements on actual hardware.

**Example**: Every benchmark in the llcuda documentation was run on a GeForce 940M. No simulations, no theoretical calculations.

### 3. Documentation First

Documentation isn't an afterthought—it's a core feature.

**Example**: llcuda has a complete quick start guide, installation guide, performance guide, and production-ready examples. Users can get running in under 5 minutes.

### 4. Zero-Configuration Design

Installation complexity is a bug, not a feature.

**Example**: llcuda installs with `pip install llcuda`. No CUDA toolkit, no compilation, no configuration files. It just works.

### 5. Production Quality

If it's not on PyPI with proper versioning, it's not production-ready.

**Example**: llcuda is published to PyPI with semantic versioning, not just a GitHub repo with a README.

---

## Background

### Education

**Computer Science Background** with focus on:
- Algorithms and Data Structures
- Systems Programming
- GPU Computing and Parallel Processing
- Machine Learning and AI

### Professional Experience

**Product-Minded Software Engineer** specializing in:
- Python package development and PyPI publishing
- CUDA programming and GPU optimization
- Building tools for machine learning workflows
- Technical documentation and developer experience

**Key Projects:**
- **llcuda v1.0.1**: PyPI package for LLM inference on legacy GPUs with bundled CUDA binaries
- **CUDA Systems Research**: Empirical testing on Maxwell-era GPUs

---

## Why Legacy GPUs?

### The Problem

The AI/ML community often assumes everyone has:
- Modern RTX GPUs (3000/4000 series)
- 8GB+ VRAM
- Willingness to spend $500-$2000 on hardware

**Reality**: Millions of people have perfectly capable older GPUs collecting dust because the tools don't support them.

### The Opportunity

Legacy GPUs like the GeForce 940M can run modern LLMs with proper optimization:
- 1GB VRAM is enough for 2B parameter models
- Maxwell architecture (2014) still has hundreds of CUDA cores
- Quantization (Q4_K_M) makes models fit in limited memory
- Performance is acceptable for interactive use (~15 tok/s)

### The Mission

Make AI tools accessible on hardware people already own. No expensive upgrades needed.

---

## Projects

### Active Project

**llcuda v1.0.1** - PyTorch-style LLM inference for legacy NVIDIA GPUs
- **Status**: Published to PyPI, actively maintained
- **Version**: 1.0.1 (December 2025)
- **Focus**: Zero-configuration installation, smart model loading, hardware auto-config
- **Features**: Bundled CUDA 12.8 binaries, HuggingFace registry, performance metrics
- **Links**: [PyPI](https://pypi.org/project/llcuda/) | [Docs](/llcuda/) | [GitHub](https://github.com/waqasm86/llcuda) | [v1.0.1 Release](https://github.com/waqasm86/llcuda/releases/tag/v1.0.1)

### Future Directions

**Planned Work:**
- **Windows Support**: Pre-built binaries for Windows + CUDA
- **Model Optimization**: Custom quantization for legacy GPUs
- **Advanced Features**: Speculative decoding, FlashAttention integration
- **Broader Hardware Support**: AMD GPUs (ROCm), Intel GPUs (oneAPI)

---

## Values

### Accessibility
AI tools should be accessible to everyone, not just those with expensive hardware.

### Empiricism
Claims should be backed by real measurements, not marketing promises.

### Transparency
Open source code, public documentation, honest benchmarks.

### Quality
Production-ready tools, not just research prototypes.

### User-Centric
Design for the user's experience, not the developer's convenience.

---

## Technical Writing

I believe in **documentation as a core feature** of software. Good documentation:
- Gets users productive in minutes, not hours
- Provides realistic benchmarks and expectations
- Includes production-ready code examples
- Anticipates and answers common questions
- Is maintained alongside the code

**Example**: The llcuda documentation includes:
- 5-minute quick start guide
- Comprehensive installation guide with troubleshooting
- Real benchmarks on actual hardware
- Production-ready code examples
- Clear explanations of design decisions

---

## Open Source

All my projects are open source:

**llcuda v1.0.1**
- **License**: MIT
- **Repository**: [github.com/waqasm86/llcuda](https://github.com/waqasm86/llcuda)
- **PyPI**: [pypi.org/project/llcuda](https://pypi.org/project/llcuda/)
- **Contributions**: Bug reports, feature requests, model testing, and PRs welcome

---

## Contact

I'm always interested in:
- Feedback on llcuda and related projects
- Collaboration on making AI more accessible
- Discussions about GPU optimization and LLM systems
- Opportunities to build production-quality AI tools

**Email**: [waqasm86@gmail.com](mailto:waqasm86@gmail.com)
**GitHub**: [github.com/waqasm86](https://github.com/waqasm86)
**PyPI**: [pypi.org/project/llcuda](https://pypi.org/project/llcuda/)

[Get in touch &rarr;](/contact/)

---

## Resume

For a detailed resume including professional experience, education, and technical skills:

**[Download Resume (PDF)](/resume/Muhammad_Waqas_Resume_2025.pdf)**

---

## Testimonials

### From the Community

> "Finally, a tool that actually works on my old laptop GPU! llcuda made LLM development accessible without buying new hardware."
> **— Data Science Student**

> "The documentation is excellent. Got up and running in under 5 minutes, exactly as promised."
> **— Python Developer**

> "Impressive performance on legacy hardware. The empirical benchmarks gave me realistic expectations."
> **— ML Engineer**

---

## Inspiration

My work is inspired by engineers who build practical, accessible tools:

- **Dan McCreary**: Excellent technical documentation and knowledge graphs
- **Georgi Gerganov**: Creator of llama.cpp, making LLMs accessible
- **Yann LeCun**: Advocate for open, accessible AI research
- **Linus Torvalds**: Focus on practical engineering over hype

---

## Fun Facts

- **Favorite GPU**: GeForce 940M (1GB VRAM) - my primary testing platform
- **Favorite Language**: Python for APIs, C++ for performance
- **Favorite Tool**: JupyterLab for interactive development
- **Favorite Metric**: Tokens per second (measured on real hardware)
- **Favorite Documentation Style**: MkDocs Material (this site!)

---

## What's Next?

I'm continually working to make AI more accessible on legacy hardware:

1. **Expand llcuda**: Windows support, more models, better performance
2. **Explore New Architectures**: AMD GPUs (ROCm), Intel GPUs (oneAPI)
3. **Build Community**: Help others run LLMs on their existing hardware
4. **Document Everything**: Share knowledge through comprehensive guides

**Follow my work:**
- **GitHub**: [github.com/waqasm86](https://github.com/waqasm86)
- **PyPI**: [pypi.org/project/llcuda](https://pypi.org/project/llcuda/)
- **This Site**: Regular updates on projects and learnings

---

## Let's Build Together

Interested in collaborating on making AI tools more accessible? Have an old GPU and want to contribute to testing? Want to improve the documentation?

**I'd love to hear from you.**

[Contact Me &rarr;](/contact/)
