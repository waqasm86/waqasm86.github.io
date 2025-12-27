# Code Examples

Production-ready code samples for common llcuda use cases. All examples tested on GeForce 940M (1GB VRAM) with Ubuntu 22.04.

---

**Table of Contents:**

1. [Basic Usage](#basic-usage)
2. [Interactive Chat](#interactive-chat)
3. [Context Management](#context-management)
4. [Custom Models](#custom-models)
5. [JupyterLab Integration](#jupyterlab-integration)
6. [Batch Processing](#batch-processing)
7. [Error Handling](#error-handling)
8. [Code Generation](#code-generation)
9. [Data Analysis](#data-analysis)
10. [Production Patterns](#production-patterns)

---

## Basic Usage

### Hello World

Simplest possible llcuda usage:

```python
from llcuda import LLM

# Initialize with defaults
llm = LLM()

# Ask a question
response = llm.chat("What is Python?")
print(response)
```

**Expected output:**
```
Python is a high-level, interpreted programming language known for its
clear syntax and readability. It's widely used for web development, data
analysis, artificial intelligence, automation, and more.
```

### Custom Configuration

Full control over model parameters:

```python
from llcuda import LLM

llm = LLM(
    model="gemma-2-2b-it",        # Model selection
    max_tokens=512,                # Maximum response length
    temperature=0.7,               # Randomness (0-1)
    top_p=0.9,                     # Nucleus sampling
    top_k=40,                      # Top-K sampling
    repeat_penalty=1.1,            # Penalize repetition
    context_length=2048,           # Context window
    verbose=True                   # Show debug info
)

response = llm.chat("Tell me a story")
print(response)
```

---

## Interactive Chat

### Simple CLI Chat

Build a basic command-line chat interface:

```python
from llcuda import LLM

def main():
    print("llcuda Chat Interface")
    print("Type 'quit' to exit\n")

    llm = LLM()

    while True:
        user_input = input("You: ").strip()

        if user_input.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break

        if not user_input:
            continue

        response = llm.chat(user_input)
        print(f"AI: {response}\n")

if __name__ == "__main__":
    main()
```

**Usage:**
```bash
python chat.py
```

### Chat with History Display

Show conversation history:

```python
from llcuda import LLM

class ChatBot:
    def __init__(self):
        self.llm = LLM()
        self.history = []

    def chat(self, message):
        # Store user message
        self.history.append({"role": "user", "content": message})

        # Get response
        response = self.llm.chat(message)

        # Store AI response
        self.history.append({"role": "assistant", "content": response})

        return response

    def show_history(self):
        print("\n--- Conversation History ---")
        for i, msg in enumerate(self.history, 1):
            role = msg["role"].upper()
            content = msg["content"]
            print(f"{i}. [{role}]: {content}\n")

# Usage
bot = ChatBot()

bot.chat("Hi, my name is Alice")
bot.chat("What should I learn about Python?")
bot.chat("What was my name?")

bot.show_history()
```

---

## Context Management

### Multi-Turn Conversation

llcuda automatically maintains context:

```python
from llcuda import LLM

llm = LLM()

# First message sets context
llm.chat("I'm working on a Python project to analyze sales data.")

# Subsequent messages reference previous context
response1 = llm.chat("What libraries should I use?")
print(response1)
# Mentions pandas, matplotlib, etc. (context-aware)

response2 = llm.chat("Show me example code")
print(response2)
# Provides code for sales analysis (remembers topic)

response3 = llm.chat("How can I optimize it?")
print(response3)
# Gives optimization tips for sales analysis (full context)
```

### Manual Context Reset

Clear conversation history when starting a new topic:

```python
from llcuda import LLM

llm = LLM()

# First conversation
llm.chat("Tell me about Python")
llm.chat("What about its history?")

# Reset for new topic
llm.reset_conversation()

# New conversation (no memory of Python discussion)
llm.chat("Tell me about JavaScript")
```

### Context Window Management

Handle long conversations that exceed context length:

```python
from llcuda import LLM

class ContextAwareLLM:
    def __init__(self, context_length=2048):
        self.llm = LLM(context_length=context_length)
        self.message_count = 0
        self.max_messages = 10  # Reset after 10 messages

    def chat(self, message):
        self.message_count += 1

        # Auto-reset if approaching context limit
        if self.message_count >= self.max_messages:
            print("[Context reset - starting fresh conversation]")
            self.llm.reset_conversation()
            self.message_count = 0

        return self.llm.chat(message)

# Usage
llm = ContextAwareLLM()

for i in range(15):
    response = llm.chat(f"Tell me fact #{i} about space")
    print(f"{i}: {response[:50]}...")
```

---

## Custom Models

### Use Different Models

Switch between models for different use cases:

```python
from llcuda import LLM

# Fast model for quick responses
fast_llm = LLM(model="llama-3.2-1b-instruct")
quick_answer = fast_llm.chat("What is 2+2?")
print(f"Fast: {quick_answer}")

# Quality model for complex tasks
quality_llm = LLM(model="gemma-2-2b-it")
detailed_answer = quality_llm.chat("Explain quantum entanglement")
print(f"Quality: {detailed_answer}")
```

### Load Custom GGUF Model

Use any GGUF model from Hugging Face:

```python
from llcuda import LLM

# Load custom model
llm = LLM(
    model="TheBloke/Mistral-7B-Instruct-v0.2-GGUF",
    model_file="mistral-7b-instruct-v0.2.Q4_K_M.gguf"
)

response = llm.chat("Hello!")
print(response)
```

!!! warning "VRAM Requirements"
    7B models require ~4GB VRAM. Won't fit on 1GB GPUs like GeForce 940M.

### Model Comparison Script

Compare outputs from different models:

```python
from llcuda import LLM

models = [
    "gemma-2-2b-it",
    "llama-3.2-1b-instruct",
    "qwen-2.5-0.5b-instruct"
]

prompt = "Explain machine learning in one sentence"

print(f"Prompt: {prompt}\n")

for model_name in models:
    llm = LLM(model=model_name)
    response = llm.chat(prompt)
    print(f"{model_name}:")
    print(f"  {response}\n")
```

---

## JupyterLab Integration

### Basic Jupyter Usage

Perfect for exploratory data analysis:

```python
# Cell 1: Setup
from llcuda import LLM
llm = LLM(model="gemma-2-2b-it")

# Cell 2: Ask questions interactively
response = llm.chat("What is gradient descent?")
print(response)

# Cell 3: Follow-up (maintains context)
response = llm.chat("Give me a Python example")
print(response)

# Cell 4: Test the code
exec(response.split("```python")[1].split("```")[0])
```

### Data Analysis in Jupyter

Integrate with pandas workflows:

```python
import pandas as pd
from llcuda import LLM

# Load data
df = pd.read_csv("sales_data.csv")

# Initialize LLM
llm = LLM()

# Get LLM insights on data
summary = df.describe().to_string()
insights = llm.chat(f"Analyze this sales data:\n{summary}")
print(insights)

# Follow-up questions
question = "What trends do you see?"
response = llm.chat(question)
print(response)
```

### Generate Visualization Code

Let LLM help with plotting:

```python
import pandas as pd
from llcuda import LLM

df = pd.read_csv("sales.csv")
llm = LLM()

# Ask for visualization code
code = llm.chat(f"""
Given this DataFrame with columns {list(df.columns)},
write Python code using matplotlib to create a bar chart
of sales by category.
""")

print(code)

# Execute the code
exec(code.split("```python")[1].split("```")[0])
```

### Jupyter Magic Command (Advanced)

Create a custom Jupyter magic for llcuda:

```python
# In a notebook cell
from IPython.core.magic import register_line_magic
from llcuda import LLM

# Initialize LLM once
_llm = LLM()

@register_line_magic
def ask(line):
    """Magic command: %ask <your question>"""
    response = _llm.chat(line)
    print(response)
    return response

# Usage in subsequent cells:
# %ask What is machine learning?
# %ask Explain neural networks
```

---

## Batch Processing

### Process Multiple Inputs

Efficient batch processing pattern:

```python
from llcuda import LLM

inputs = [
    "Summarize: Python is a programming language...",
    "Summarize: Machine learning is a subset of AI...",
    "Summarize: Data science involves analyzing data...",
]

llm = LLM(model="llama-3.2-1b-instruct")  # Use fast model

results = []
for i, input_text in enumerate(inputs, 1):
    print(f"Processing {i}/{len(inputs)}...")
    result = llm.chat(input_text)
    results.append(result)

# Display results
for i, result in enumerate(results, 1):
    print(f"\nResult {i}:")
    print(result)
```

### Parallel Processing (Multi-Model)

Use multiple model instances for parallelization:

```python
from llcuda import LLM
from concurrent.futures import ThreadPoolExecutor

def process_with_llm(text):
    """Each thread gets its own LLM instance"""
    llm = LLM(model="llama-3.2-1b-instruct")
    return llm.chat(f"Summarize: {text}")

texts = [
    "Python is a programming language...",
    "Machine learning is a subset of AI...",
    "Data science involves analyzing data...",
]

# Process in parallel (if you have VRAM for multiple models)
with ThreadPoolExecutor(max_workers=2) as executor:
    results = list(executor.map(process_with_llm, texts))

for i, result in enumerate(results, 1):
    print(f"\nResult {i}: {result}")
```

!!! warning "VRAM Limitations"
    Multiple LLM instances require VRAM for each. On 1GB GPUs, use sequential processing instead.

---

## Error Handling

### Robust Error Handling

Production-ready error handling:

```python
from llcuda import LLM
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def safe_chat(prompt, max_retries=3):
    """Chat with retry logic"""
    llm = None

    for attempt in range(max_retries):
        try:
            if llm is None:
                llm = LLM()

            response = llm.chat(prompt)
            return response

        except MemoryError:
            logger.error("CUDA out of memory. Try smaller model.")
            return None

        except Exception as e:
            logger.warning(f"Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                logger.info("Retrying...")
                llm = None  # Reinitialize
            else:
                logger.error("Max retries reached")
                return None

# Usage
response = safe_chat("What is Python?")
if response:
    print(response)
else:
    print("Failed to get response")
```

### Timeout Handling

Add timeout for long-running generations:

```python
from llcuda import LLM
import signal
from contextlib import contextmanager

class TimeoutError(Exception):
    pass

@contextmanager
def timeout(seconds):
    def timeout_handler(signum, frame):
        raise TimeoutError(f"Timed out after {seconds} seconds")

    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)

# Usage
llm = LLM()

try:
    with timeout(30):  # 30 second timeout
        response = llm.chat("Write a very long essay...")
        print(response)
except TimeoutError as e:
    print(f"Generation timed out: {e}")
```

---

## Code Generation

### Generate and Execute Code

Safe code generation with execution:

```python
from llcuda import LLM

def generate_and_test_code(task):
    llm = LLM()

    # Generate code
    prompt = f"Write a Python function to {task}. Only output the code, no explanations."
    response = llm.chat(prompt)

    # Extract code block
    if "```python" in response:
        code = response.split("```python")[1].split("```")[0]
    else:
        code = response

    print("Generated code:")
    print(code)

    # Test execution (be careful in production!)
    try:
        exec(code, globals())
        print("\nCode executed successfully!")
        return code
    except Exception as e:
        print(f"\nExecution error: {e}")
        return None

# Usage
generate_and_test_code("calculate fibonacci numbers")
```

### Code Review Assistant

Get LLM feedback on code:

```python
from llcuda import LLM

def review_code(code):
    llm = LLM(model="gemma-2-2b-it")

    prompt = f"""
Review this Python code for:
1. Correctness
2. Performance
3. Best practices
4. Potential bugs

Code:
{code}
"""

    review = llm.chat(prompt)
    return review

# Usage
my_code = """
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
"""

review = review_code(my_code)
print(review)
```

---

## Data Analysis

### Automated Data Insights

Generate insights from pandas DataFrames:

```python
import pandas as pd
from llcuda import LLM

def analyze_dataframe(df, question=None):
    llm = LLM()

    # Generate summary
    summary = f"""
Dataset Shape: {df.shape}
Columns: {list(df.columns)}
Summary Statistics:
{df.describe().to_string()}

First few rows:
{df.head().to_string()}
"""

    # Ask LLM for insights
    if question:
        prompt = f"{summary}\n\nQuestion: {question}"
    else:
        prompt = f"{summary}\n\nProvide key insights and recommendations."

    insights = llm.chat(prompt)
    return insights

# Usage
df = pd.read_csv("sales_data.csv")
insights = analyze_dataframe(df, "What are the main trends?")
print(insights)
```

### Generate SQL Queries

Natural language to SQL:

```python
from llcuda import LLM

def nl_to_sql(natural_language, schema):
    llm = LLM()

    prompt = f"""
Given this database schema:
{schema}

Convert this natural language query to SQL:
"{natural_language}"

Output only the SQL query, no explanations.
"""

    sql = llm.chat(prompt)
    return sql.strip()

# Usage
schema = """
Table: employees
Columns: id, name, department, salary, hire_date
"""

query = nl_to_sql("Find all employees hired after 2020 with salary > 50000", schema)
print(query)
```

---

## Production Patterns

### Singleton LLM Instance

Reuse LLM instance across application:

```python
from llcuda import LLM

class LLMSingleton:
    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = LLM(model="gemma-2-2b-it")
        return cls._instance

# Usage across multiple modules
llm = LLMSingleton.get_instance()
response = llm.chat("Hello")
```

### Cached Responses

Cache LLM responses to avoid redundant generation:

```python
from llcuda import LLM
from functools import lru_cache

class CachedLLM:
    def __init__(self):
        self.llm = LLM()

    @lru_cache(maxsize=128)
    def chat(self, message):
        """Cached chat - same input returns cached response"""
        return self.llm.chat(message)

# Usage
llm = CachedLLM()

# First call: generates response
response1 = llm.chat("What is Python?")

# Second call: returns cached response (instant)
response2 = llm.chat("What is Python?")

assert response1 == response2
```

### Rate Limiting

Limit requests per minute:

```python
from llcuda import LLM
import time
from collections import deque

class RateLimitedLLM:
    def __init__(self, max_requests_per_minute=10):
        self.llm = LLM()
        self.max_requests = max_requests_per_minute
        self.request_times = deque()

    def chat(self, message):
        # Remove requests older than 1 minute
        current_time = time.time()
        while self.request_times and current_time - self.request_times[0] > 60:
            self.request_times.popleft()

        # Check rate limit
        if len(self.request_times) >= self.max_requests:
            wait_time = 60 - (current_time - self.request_times[0])
            print(f"Rate limit reached. Waiting {wait_time:.1f}s...")
            time.sleep(wait_time)
            return self.chat(message)

        # Process request
        self.request_times.append(current_time)
        return self.llm.chat(message)

# Usage
llm = RateLimitedLLM(max_requests_per_minute=5)

for i in range(10):
    response = llm.chat(f"Request #{i}")
    print(f"{i}: {response[:50]}...")
```

### Logging and Monitoring

Track LLM usage and performance:

```python
from llcuda import LLM
import time
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MonitoredLLM:
    def __init__(self):
        self.llm = LLM()
        self.request_count = 0
        self.total_time = 0

    def chat(self, message):
        self.request_count += 1
        logger.info(f"Request #{self.request_count}: {message[:50]}...")

        start = time.time()
        response = self.llm.chat(message)
        elapsed = time.time() - start

        self.total_time += elapsed

        logger.info(f"Response generated in {elapsed:.2f}s")
        logger.info(f"Average time: {self.total_time / self.request_count:.2f}s")

        return response

# Usage
llm = MonitoredLLM()
response = llm.chat("What is machine learning?")
```

---

## Complete Application Example

### Chatbot with All Best Practices

Full production-ready chatbot:

```python
from llcuda import LLM
import logging
import time
from functools import lru_cache

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ProductionChatbot:
    def __init__(self, model="gemma-2-2b-it"):
        self.llm = None
        self.model = model
        self.conversation_count = 0
        self.initialize()

    def initialize(self):
        """Initialize LLM with error handling"""
        try:
            logger.info(f"Initializing {self.model}...")
            self.llm = LLM(model=self.model)
            logger.info("LLM initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {e}")
            raise

    def chat(self, message, max_retries=3):
        """Chat with retry logic and monitoring"""
        self.conversation_count += 1

        for attempt in range(max_retries):
            try:
                logger.info(f"Processing message #{self.conversation_count}")
                start = time.time()

                response = self.llm.chat(message)

                elapsed = time.time() - start
                tokens = len(response.split())
                speed = tokens / elapsed

                logger.info(f"Response: {tokens} tokens in {elapsed:.1f}s ({speed:.1f} tok/s)")

                return response

            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    logger.info("Retrying...")
                    time.sleep(1)
                else:
                    logger.error("Max retries reached")
                    return None

    def reset(self):
        """Reset conversation"""
        logger.info("Resetting conversation")
        self.llm.reset_conversation()
        self.conversation_count = 0

# Usage
if __name__ == "__main__":
    bot = ProductionChatbot()

    print("Chatbot ready! Type 'quit' to exit.\n")

    while True:
        user_input = input("You: ").strip()

        if user_input.lower() in ['quit', 'exit']:
            print("Goodbye!")
            break

        if user_input.lower() == 'reset':
            bot.reset()
            continue

        response = bot.chat(user_input)

        if response:
            print(f"Bot: {response}\n")
        else:
            print("Sorry, I encountered an error. Please try again.\n")
```

---

## Next Steps

Explore more:

1. **[Performance Guide](/llcuda/performance/)** - Optimize for your GPU
2. **[Installation Guide](/llcuda/installation/)** - Advanced setup
3. **[Main Documentation](/llcuda/)** - Full API reference

---

**Happy coding with llcuda!**
