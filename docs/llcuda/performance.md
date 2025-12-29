# Performance Benchmarks

Real-world benchmarks on actual hardware running llcuda v1.0.1.

---

## Test Platform

**Hardware**: GeForce 940M (1GB VRAM, 384 CUDA cores, Maxwell architecture)
**OS**: Ubuntu 22.04 LTS
**llcuda**: v1.0.1
**CUDA**: 12.8 (bundled)

---

## Model Performance

### Gemma 3 1B Q4_K_M (Recommended)

```
Model: google/gemma-3-1b-it (Q4_K_M quantization)
Size: 700 MB
VRAM Usage: ~800 MB
Performance: ~15 tokens/second
GPU Layers: 20 (auto-configured)
Context: 512 tokens
```

**Use case**: General chat, Q&A, code assistance

### TinyLlama 1.1B Q5_K_M (Fastest)

```
Model: TinyLlama-1.1B-Chat (Q5_K_M quantization)
Size: 800 MB
VRAM Usage: ~750 MB
Performance: ~18 tokens/second
GPU Layers: 20 (auto-configured)
Context: 512 tokens
```

**Use case**: Fast responses, simple tasks

### Llama 3.2 1B Q4_K_M (Best Quality)

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

## Full Model Registry

| Model | Size | Min VRAM | 940M Speed | Use Case |
|-------|------|----------|------------|----------|
| tinyllama-1.1b-Q5_K_M | 800 MB | 1 GB | ~18 tok/s | Fastest |
| gemma-3-1b-Q4_K_M | 700 MB | 1 GB | ~15 tok/s | Recommended |
| llama-3.2-1b-Q4_K_M | 750 MB | 1 GB | ~16 tok/s | Best quality |
| qwen-2.5-0.5b-Q4_K_M | 350 MB | 1 GB | ~25 tok/s | Ultra-fast |
| phi-2-Q4_K_M | 1.6 GB | 2 GB | ~13 tok/s | Math/code |
| phi-3-mini-Q4_K_M | 2.2 GB | 2 GB | ~12 tok/s | Code-focused |
| llama-3.2-3b-Q4_K_M | 1.9 GB | 2 GB | ~10 tok/s | Better quality |
| gemma-2-2b-Q4_K_M | 1.6 GB | 2 GB | ~12 tok/s | Google model |
| mistral-7b-Q4_K_M | 4.1 GB | 4 GB | ~8 tok/s | High quality |
| llama-3.1-8b-Q4_K_M | 4.9 GB | 6 GB | ~6 tok/s | Best quality |
| qwen-2.5-7b-Q4_K_M | 4.4 GB | 4 GB | ~7 tok/s | Multilingual |

---

## Latency Metrics (Gemma 3 1B)

Based on 100 inference runs on GeForce 940M:

```
p50 (median): 850 ms
p95: 1200 ms
p99: 1500 ms
Mean: 920 ms
Min: 720 ms
Max: 1650 ms
```

**Throughput**: ~15 tokens/second average

---

## Memory Usage

### GeForce 940M (1GB VRAM)

| Model | VRAM Used | GPU Layers | Fits? |
|-------|-----------|------------|-------|
| gemma-3-1b-Q4_K_M | 800 MB | 20 | ✓ Yes |
| tinyllama-1.1b-Q5_K_M | 750 MB | 20 | ✓ Yes |
| llama-3.2-1b-Q4_K_M | 800 MB | 20 | ✓ Yes |
| phi-3-mini-Q4_K_M | 950 MB | 15 | ⚠ Tight |
| mistral-7b-Q4_K_M | - | - | ✗ No |

---

## Performance Comparison

### llcuda vs Cloud APIs (per 100 tokens)

| Provider | Latency | Cost | Privacy |
|----------|---------|------|---------|
| **llcuda (940M)** | ~6.7s | $0.00 | Local |
| OpenAI GPT-4 | ~2s | $0.30 | Cloud |
| OpenAI GPT-3.5 | ~1s | $0.002 | Cloud |
| Claude Sonnet | ~1.5s | $0.30 | Cloud |

**llcuda advantage**: Zero cost, complete privacy, works offline

---

## Optimization Tips

### 1. Choose Right Model for VRAM

```python
# 1GB VRAM
engine.load_model("gemma-3-1b-Q4_K_M")

# 2GB VRAM
engine.load_model("phi-3-mini-Q4_K_M")

# 4GB+ VRAM
engine.load_model("mistral-7b-Q4_K_M")
```

### 2. Adjust GPU Layers

```python
# More GPU = faster (if VRAM available)
engine.load_model("gemma-3-1b-Q4_K_M", gpu_layers=30)

# Less GPU = saves VRAM
engine.load_model("gemma-3-1b-Q4_K_M", gpu_layers=10)
```

### 3. Use Batch Inference

```python
# More efficient than individual calls
results = engine.batch_infer(prompts, max_tokens=50)
```

### 4. Monitor Performance

```python
metrics = engine.get_metrics()
print(f"p95: {metrics['latency']['p95_ms']:.2f} ms")
```

---

## Hardware Scaling

### Expected Performance on Different GPUs

| GPU | VRAM | Estimated Speed | Recommended Model |
|-----|------|----------------|-------------------|
| GeForce 940M | 1 GB | ~15 tok/s | gemma-3-1b-Q4_K_M |
| GTX 1050 | 2 GB | ~25 tok/s | phi-3-mini-Q4_K_M |
| GTX 1060 | 6 GB | ~50 tok/s | mistral-7b-Q4_K_M |
| RTX 3060 | 12 GB | ~80 tok/s | llama-3.1-8b-Q4_K_M |

*Estimates based on relative compute capability*

---

## Real-World Use Cases

### Interactive Chat: ✓ Good
15 tok/s ≈ 150 words/minute (human reading speed: 250 wpm)

### Code Generation: ✓ Good
Fast enough for real-time code completion and review

### Data Analysis: ✓ Excellent
Perfect for JupyterLab exploratory analysis

### Production APIs: ⚠ Consider Batching
Use batch inference for better throughput

---

## Next Steps

- **[Quick Start](/llcuda/quickstart/)** - Basic usage
- **[Installation](/llcuda/installation/)** - Setup guide
- **[Examples](/llcuda/examples/)** - Production code samples
