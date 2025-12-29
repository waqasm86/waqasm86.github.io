# Code Examples

Production-ready code samples for llcuda v1.0.0. All examples tested on GeForce 940M (1GB VRAM).

---

## Basic Usage

### Hello World

```python
import llcuda

engine = llcuda.InferenceEngine()
engine.load_model("gemma-3-1b-Q4_K_M")

result = engine.infer("What is Python?")
print(result.text)
```

---

## Interactive Chat

### Multi-Turn Conversation

```python
import llcuda

engine = llcuda.InferenceEngine()
engine.load_model("gemma-3-1b-Q4_K_M")

conversation = [
    "What is machine learning?",
    "How does it differ from AI?",
    "Give me a practical example"
]

for prompt in conversation:
    result = engine.infer(prompt, max_tokens=100)
    print(f"Q: {prompt}")
    print(f"A: {result.text}\n")
```

---

## JupyterLab Integration

### Data Analysis with LLM

```python
import pandas as pd
import llcuda

# Load data
df = pd.read_csv("sales_data.csv")

# Create engine
engine = llcuda.InferenceEngine()
engine.load_model("gemma-3-1b-Q4_K_M")

# Get data summary
summary = df.describe().to_string()

# Analyze with LLM
analysis = engine.infer(
    f"Analyze this sales data and provide insights:\n{summary}",
    max_tokens=200
)

print(analysis.text)
```

---

## Batch Processing

### Process Multiple Prompts

```python
import llcuda

engine = llcuda.InferenceEngine()
engine.load_model("gemma-3-1b-Q4_K_M")

prompts = [
    "Explain neural networks",
    "What is deep learning?",
    "Describe NLP"
]

# Batch inference (more efficient)
results = engine.batch_infer(prompts, max_tokens=80)

for prompt, result in zip(prompts, results):
    print(f"Q: {prompt}")
    print(f"A: {result.text}")
    print(f"Speed: {result.tokens_per_sec:.1f} tok/s\n")
```

---

## Code Generation

### Generate and Review Code

```python
import llcuda

engine = llcuda.InferenceEngine()
engine.load_model("gemma-3-1b-Q4_K_M")

# Generate code
code_prompt = "Write a Python function to calculate Fibonacci numbers"
code_result = engine.infer(code_prompt, max_tokens=150)
print("Generated Code:")
print(code_result.text)

# Review code
review_prompt = f"Review this code for improvements:\n{code_result.text}"
review_result = engine.infer(review_prompt, max_tokens=150)
print("\nCode Review:")
print(review_result.text)
```

---

## Temperature Comparison

### Experiment with Different Temperatures

```python
import llcuda

engine = llcuda.InferenceEngine()
engine.load_model("gemma-3-1b-Q4_K_M")

prompt = "Write a haiku about AI"
temperatures = [0.3, 0.7, 1.2]

for temp in temperatures:
    result = engine.infer(prompt, temperature=temp, max_tokens=50)
    print(f"\nTemperature {temp}:")
    print(result.text)
```

---

## Performance Monitoring

### Track Latency and Throughput

```python
import llcuda

engine = llcuda.InferenceEngine()
engine.load_model("gemma-3-1b-Q4_K_M")

# Run some inferences
for i in range(20):
    engine.infer("Hello, how are you?", max_tokens=20)

# Get performance metrics
metrics = engine.get_metrics()

print("Latency Statistics:")
print(f"  Mean: {metrics['latency']['mean_ms']:.2f} ms")
print(f"  p50:  {metrics['latency']['p50_ms']:.2f} ms")
print(f"  p95:  {metrics['latency']['p95_ms']:.2f} ms")
print(f"  p99:  {metrics['latency']['p99_ms']:.2f} ms")

print("\nThroughput:")
print(f"  Total Tokens: {metrics['throughput']['total_tokens']}")
print(f"  Tokens/sec: {metrics['throughput']['tokens_per_sec']:.2f}")
```

---

## Error Handling

### Robust Production Code

```python
import llcuda

try:
    engine = llcuda.InferenceEngine()
    engine.load_model("gemma-3-1b-Q4_K_M")

    result = engine.infer("What is AI?", max_tokens=100)

    if result.success:
        print(result.text)
    else:
        print(f"Inference failed: {result.error_message}")

except Exception as e:
    print(f"Error: {e}")
finally:
    if 'engine' in locals():
        engine.unload_model()
```

---

## Context Manager Pattern

### Automatic Resource Cleanup

```python
import llcuda

# Use context manager for automatic cleanup
with llcuda.InferenceEngine() as engine:
    engine.load_model("gemma-3-1b-Q4_K_M")

    result = engine.infer("Explain quantum computing", max_tokens=100)
    print(result.text)

# Engine automatically cleaned up here
```

---

## Using Local GGUF Files

### Load Custom Models

```python
import llcuda

engine = llcuda.InferenceEngine()

# Find local GGUF models
models = llcuda.find_gguf_models()

if models:
    # Use first model found
    engine.load_model(str(models[0]))
else:
    # Fall back to registry
    engine.load_model("gemma-3-1b-Q4_K_M")

result = engine.infer("Hello!", max_tokens=20)
print(result.text)
```

---

## Production Pattern: API Wrapper

### Build a Simple API

```python
import llcuda
from typing import Dict

class LLMService:
    def __init__(self, model_name: str = "gemma-3-1b-Q4_K_M"):
        self.engine = llcuda.InferenceEngine()
        self.engine.load_model(model_name)

    def generate(self, prompt: str, max_tokens: int = 100) -> Dict:
        result = self.engine.infer(prompt, max_tokens=max_tokens)

        return {
            "text": result.text,
            "tokens": result.tokens_generated,
            "speed": result.tokens_per_sec,
            "latency_ms": result.latency_ms
        }

    def batch_generate(self, prompts: list, max_tokens: int = 100) -> list:
        results = self.engine.batch_infer(prompts, max_tokens=max_tokens)
        return [
            {
                "text": r.text,
                "tokens": r.tokens_generated,
                "speed": r.tokens_per_sec
            }
            for r in results
        ]

    def get_stats(self) -> Dict:
        return self.engine.get_metrics()

    def cleanup(self):
        self.engine.unload_model()

# Usage
service = LLMService()
response = service.generate("What is AI?")
print(response)
```

---

## Complete JupyterLab Example

See the full [JupyterLab notebook](https://github.com/waqasm86/llcuda/blob/main/examples/quickstart_jupyterlab.ipynb) with:

- System info checks
- Model registry listing
- Batch inference
- Performance visualization
- Context manager usage
- Temperature comparisons

---

## Next Steps

- **[Quick Start](/llcuda/quickstart/)** - Getting started guide
- **[Performance](/llcuda/performance/)** - Optimization tips
- **[GitHub](https://github.com/waqasm86/llcuda)** - Source code and more examples
