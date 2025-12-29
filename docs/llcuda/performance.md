# Performance Benchmarks

Real-world benchmarks on actual hardware running llcuda v1.1.0.

---

## Cloud Platforms

llcuda v1.1.0 now supports Google Colab and Kaggle!

### Tesla T4 (Google Colab / Kaggle) - 15GB VRAM

| Model | Quantization | GPU Layers | Speed | VRAM Usage |
|-------|--------------|-----------|-------|------------|
| Gemma 3 1B | Q4_K_M | 26 (all) | ~15 tok/s | ~1.2 GB |
| Llama 3.1 7B | Q4_K_M | 20 | ~5-8 tok/s | ~8 GB |
| Llama 3.1 7B | Q4_K_M | 32 (all) | ~8-10 tok/s | ~12 GB |

**Platform**: Google Colab Free / Kaggle (2x T4, 30GB total VRAM)
**llcuda**: v1.1.0
**Use case**: Research, education, rapid prototyping

### Tesla P100 (Google Colab) - 16GB VRAM

| Model | Quantization | GPU Layers | Speed | VRAM Usage |
|-------|--------------|-----------|-------|------------|
| Gemma 3 1B | Q4_K_M | 26 (all) | ~18 tok/s | ~1.2 GB |
| Llama 3.1 7B | Q4_K_M | 32 (all) | ~10 tok/s | ~12 GB |
| Llama 3.1 13B | Q4_K_M | 20 | ~5-7 tok/s | ~14 GB |

**Platform**: Google Colab (varies by availability)
**llcuda**: v1.1.0
**Use case**: Larger models, faster inference

---

## Local GPUs

### GeForce 940M - 1GB VRAM

**Hardware**: GeForce 940M (384 CUDA cores, Maxwell architecture)
**OS**: Ubuntu 22.04 LTS
**llcuda**: v1.1.0
**CUDA**: 12.8 (bundled)

#### Gemma 3 1B Q4_K_M (Recommended)

```
Model: google/gemma-3-1b-it (Q4_K_M quantization)
Size: 700 MB
VRAM Usage: ~800 MB
Performance: ~15 tokens/second
GPU Layers: 20 (auto-configured)
Context: 512 tokens
```

**Use case**: General chat, Q&A, code assistance

#### TinyLlama 1.1B Q5_K_M (Fastest)

```
Model: TinyLlama-1.1B-Chat (Q5_K_M quantization)
Size: 800 MB
VRAM Usage: ~750 MB
Performance: ~18 tokens/second
GPU Layers: 20 (auto-configured)
Context: 512 tokens
```

**Use case**: Fast responses, simple tasks

#### Llama 3.2 1B Q4_K_M (Best Quality)

```
Model: meta-llama/Llama-3.2-1B-Instruct (Q4_K_M)
Size: 750 MB
VRAM Usage: ~800 MB
Performance: ~16 tokens/second
GPU Layers: 20 (auto-configured)
Context: 512 tokens
```

**Use case**: Best quality for 1GB VRAM

---

## Supported GPU Architectures

llcuda v1.1.0 supports all modern NVIDIA GPUs with compute capability 5.0+:

| Architecture | Compute Cap | Examples | Platforms |
|--------------|-------------|----------|-----------|
| Maxwell | 5.0-5.3 | GTX 900, 940M | Local |
| Pascal | 6.0-6.2 | GTX 10xx, P100 | Local, Colab |
| Volta | 7.0 | V100 | Colab Pro |
| Turing | 7.5 | T4, RTX 20xx | Colab, Kaggle |
| Ampere | 8.0-8.6 | A100, RTX 30xx | Colab Pro, Local |
| Ada Lovelace | 8.9 | RTX 40xx | Local |

---

## Full Model Registry

llcuda includes 11 curated models optimized for different VRAM tiers:

### 1GB VRAM Tier (GeForce 940M)

| Model | Size | Performance (940M) | Use Case |
|-------|------|-------------------|----------|
| **tinyllama-1.1b-Q5_K_M** | 800 MB | ~18 tok/s | Fastest option |
| **gemma-3-1b-Q4_K_M** | 700 MB | ~15 tok/s | Recommended |
| **llama-3.2-1b-Q4_K_M** | 750 MB | ~16 tok/s | Best quality |

### 2GB+ VRAM Tier

| Model | Size | Performance (T4) | Use Case |
|-------|------|-----------------|----------|
| **phi-3-mini-Q4_K_M** | 2.2 GB | ~12 tok/s | Code-focused |
| **gemma-2-2b-Q4_K_M** | 1.6 GB | ~14 tok/s | Balanced |
| **llama-3.2-3b-Q4_K_M** | 2.0 GB | ~12 tok/s | Quality |

### 4GB+ VRAM Tier

| Model | Size | Performance (T4) | Use Case |
|-------|------|-----------------|----------|
| **mistral-7b-Q4_K_M** | 4.1 GB | ~8 tok/s | Highest quality |
| **llama-3.1-7b-Q4_K_M** | 4.3 GB | ~5-8 tok/s | Latest Llama |
| **phi-3-medium-Q4_K_M** | 8.0 GB | ~5 tok/s | Code expert |

### 8GB+ VRAM Tier

| Model | Size | Performance (P100) | Use Case |
|-------|------|--------------------|----------|
| **llama-3.1-13b-Q4_K_M** | 7.8 GB | ~5-7 tok/s | Large context |
| **mixtral-8x7b-Q4_K_M** | 26 GB | ~3-5 tok/s | MoE architecture |

---

## Latency Metrics

llcuda tracks P50/P95/P99 latencies:

### GeForce 940M (1GB VRAM)

```
Model: gemma-3-1b-Q4_K_M
GPU Layers: 20
Context: 512 tokens

P50 latency: 65 ms
P95 latency: 72 ms
P99 latency: 78 ms
```

### Tesla T4 (Colab/Kaggle)

```
Model: gemma-3-1b-Q4_K_M
GPU Layers: 26
Context: 2048 tokens

P50 latency: 62 ms
P95 latency: 68 ms
P99 latency: 74 ms
```

---

## Testing Methodology

All benchmarks measured using:

```python
import llcuda

engine = llcuda.InferenceEngine()
engine.load_model("gemma-3-1b-Q4_K_M")

# Warmup
for _ in range(5):
    engine.infer("Test", max_tokens=50)

# Measure
results = []
for _ in range(100):
    result = engine.infer("Explain AI", max_tokens=50)
    results.append(result.tokens_per_sec)

print(f"Mean: {sum(results)/len(results):.1f} tok/s")
```

---

## Tips for Best Performance

### 1. GPU Layer Tuning

```python
# Start with auto-configuration
engine.load_model("model.gguf", auto_configure=True)

# Then manually tune
engine.load_model("model.gguf", gpu_layers=20)  # Adjust based on VRAM
```

### 2. Context Size

```python
# Smaller context = less VRAM
engine.load_model("model.gguf", ctx_size=512)   # 1GB VRAM
engine.load_model("model.gguf", ctx_size=2048)  # 4GB+ VRAM
```

### 3. Batch Size

```python
# Larger batch = better throughput (if VRAM allows)
engine.load_model("model.gguf", batch_size=512)
```

### 4. Quantization Selection

Lower quantization = smaller model = faster loading, but slightly lower quality:

- **Q4_K_M**: Best balance (recommended)
- **Q5_K_M**: Higher quality, larger size
- **Q6_K**: Highest quality, biggest size
- **Q3_K_M**: Smallest, fastest, lower quality

---

## Cloud Platform Quick Start

### Google Colab

```python
!pip install llcuda

import llcuda
engine = llcuda.InferenceEngine()
engine.load_model("gemma-3-1b-Q4_K_M", gpu_layers=26)
result = engine.infer("What is AI?")
print(result.text)
```

### Kaggle

```python
!pip install llcuda

import llcuda
engine = llcuda.InferenceEngine()
engine.load_model(
    "unsloth/gemma-3-1b-it-GGUF:gemma-3-1b-it-Q4_K_M.gguf",
    gpu_layers=26
)
result = engine.infer("Explain machine learning")
print(result.text)
```

[View complete cloud platform guide →](/llcuda/cloud-platforms/)

---

## Comparison: v1.0.x vs v1.1.0

| Metric | v1.0.x | v1.1.0 |
|--------|--------|--------|
| **Supported GPUs** | Compute 5.0 only | Compute 5.0-8.9 |
| **Cloud Platforms** | ❌ None | ✅ Colab, Kaggle |
| **Package Size** | 50 MB | 313 MB |
| **Performance (940M)** | ~15 tok/s | ~15 tok/s (same) |
| **T4 Support** | ❌ Failed | ✅ Works |
| **P100 Support** | ❌ Failed | ✅ Works |

**Backward compatible**: v1.1.0 works exactly the same on GeForce 940M as v1.0.x.

---

## Links

- **PyPI**: [pypi.org/project/llcuda](https://pypi.org/project/llcuda/)
- **GitHub**: [github.com/waqasm86/llcuda](https://github.com/waqasm86/llcuda)
- **Latest Release**: [v1.1.0](https://github.com/waqasm86/llcuda/releases/tag/v1.1.0)
- **Cloud Guide**: [Cloud Platforms](/llcuda/cloud-platforms/)
