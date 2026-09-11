In 2026, AI agents have become the default paradigm for automating complex workflows: DevOps orchestration, customer support, database triage, and e-commerce transactions.

Yet, every engineering team deploying autonomous agents in production eventually hits the same brick wall:

1. **Crippling Latency:** A standard 4-step tool chain (`think -> tool -> observe -> think`) easily burns **15 to 45 seconds**.
2. **Brutal API Bills:** Running that loop 50,000 times a day costs thousands of dollars every month for redundant reasoning.
3. **Flakiness & Non-Determinism:** Even with a 98% success rate per step, a 4-step chain has an ~8% failure rate.
4. **Redundant Inference:** Here is the unspoken truth of agent workflows: **over 90% of recurring invocations execute the exact same sequence of tool calls**, differing only by input parameters (e.g., `user_id`, `order_id`, or `date`).

Why are we invoking massive 70-billion-parameter neural networks over HTTP dozens of times just to parse an ID and pass it into a database query?

---

## 💡 The Core Insight: Borrowing from V8 and PyTorch

In computer science, this problem was solved decades ago:

- **JavaScript engines (Google V8):** Interpret dynamic code on the first run, profile hot execution paths, and compile them into native machine code.
- **Deep Learning (PyTorch `torch.compile`):** Trace dynamic tensor operations and fuse them into deterministic CUDA/C++ kernels.

What if we did the exact same thing for **AI Agent Trajectories**?

Today, I’m open-sourcing **[AgentJIT](https://github.com/eminsk/agentjit)** — a Just-In-Time trajectory compiler for AI agents that traces dynamic tool chains and compiles them into **sub-millisecond, deterministic Python AST pipelines with zero runtime token cost**.

```
       [Dynamic Agent Task]
                │
         (1st run / warmup)
                ▼
     ┌──────────────────────┐
     │   AgentJIT Tracer    │ ── Captures tool calls, data flow & variables
     └──────────────────────┘
                │
                ▼
     ┌──────────────────────┐
     │  DAG Flow Analyzer   │ ── Resolves dependencies & arithmetic expressions
     └──────────────────────┘
                │
                ▼
     ┌──────────────────────┐
     │  AST Code Generator  │ ── Synthesizes pure Python AST + Runtime Guards
     └──────────────────────┘
                │
                ▼
  ┌────────────────────────────┐
  │   Compiled JIT Pipeline    │ ──► Subsequent runs: < 0.1ms, $0 tokens!
  └────────────────────────────┘
                │
       (Guard failure? Deopt!)
                ▼
     [Fall back to LLM Agent]
```

---

## 🛠️ How It Works Under the Hood

AgentJIT operates in three distinct phases:

### 1. Tracing & Parameter Generalization
When an agent decorated with `@jit` runs for the first time, the `Tracer` records every tool invocation, its inputs, outputs, and execution timings. It builds a directed acyclic graph (DAG) of the data flow, distinguishing between static parameters and dynamic runtime inputs.

### 2. AST Code Synthesis
The compiler examines the trace and synthesizes a pure Python Abstract Syntax Tree (AST). It converts dynamic tool dispatching into hard-wired, type-checked Python function calls, resolving nested dictionaries and mathematical operators.

### 3. Speculative De-Optimization (Bailouts)
What happens if the user inputs an anomalous value or unexpected format? 

AgentJIT automatically inserts **Runtime Input Guards**. If any input violates the expected structure, the compiled pipeline immediately triggers a **speculative bailout (de-optimization)**, gracefully falling back to the original LLM agent. 

**Zero crashes, zero regressions, pure speedup.**

---

## ⚡ 10-Second Quickstart

AgentJIT is **100% self-contained** in a single library with **zero mandatory external dependencies**.

### Installation

```bash
pip install agentjit
```
*(or `uv add agentjit`)*

### Code Example

Simply decorate your agent with `@jit` and mark your tools with `@trace_tool`:

```python
from agentjit import jit, trace_tool

# 1. Define your tools
@trace_tool()
def fetch_product(product_id: str):
    return {"id": product_id, "price": 89.0, "category": "electronics"}

@trace_tool()
def calculate_vat(price: float, tax_rate: float):
    return round(price * (1.0 + tax_rate), 2)

@trace_tool()
def generate_invoice(product_id: str, total_price: float):
    return {"invoice_id": f"INV-{product_id}", "total": total_price}

# 2. Decorate your multi-step agent
@jit
def checkout_agent(product_id: str, tax_rate: float):
    # This dynamic workflow could call an LLM (Claude, GPT, Gemini)
    product = fetch_product(product_id=product_id)
    total = calculate_vat(price=product["price"], tax_rate=tax_rate)
    return generate_invoice(product_id=product["id"], total_price=total)

# Run 1: Warmup & Tracing (captures trajectory, compiles AST)
order1 = checkout_agent("SKU-100", 0.20)

# Run 2+: Instant JIT execution (< 0.1ms, ZERO tokens consumed!)
order2 = checkout_agent("SKU-200", 0.20)
```

You can even inspect the generated Python code at runtime:

```python
print(checkout_agent.source_code)
```

---

## 📊 Live Benchmark: 356x Speedup

We ran a 100-iteration benchmark in Google Colab simulating an uncompiled multi-step LLM chain versus the compiled AgentJIT pipeline:

| Metric | Uncompiled Agent | AgentJIT Pipeline | Advantage |
| :--- | :--- | :--- | :--- |
| **Mean Latency** | `37.21 ms` | `0.1044 ms` | **356.4x Faster** ⚡ |
| **Token Cost (1k runs)** | `$7.50` (2.5M tokens) | `$0.00` (0 tokens) | **100% Free** 💰 |
| **Determinism** | `~94%` (LLM hallucinations) | `100.0%` (Verified AST) | **Rock-Solid** 🛡️ |
| **Fallback Safety** | N/A | Automatic Speculative Deopt | **Zero Crashes** ✅ |

---

## 🧵 Ready for Python 3.14 & Free-Threaded No-GIL (PEP 703)

AgentJIT was architected with Python 3.13 and Python 3.14 in mind:

- **Re-entrant Thread Safety:** All internal JIT compilation caches and pipeline registries use double-checked locking with `threading.Lock`.
- **True Multi-Core Concurrency:** Fully tested on Free-Threaded CPython (`3.13t` and `3.14t`). You can spin up hundreds of concurrent agent threads without GIL contention.
- **Cross-Platform Tested:** 100% green CI matrix across both Ubuntu Linux and Windows runners for Python 3.10 through 3.14.

---

## 🚀 Try It Live in Google Colab

You don't need to configure an environment or install dependencies locally. You can run the interactive demo and benchmark right now in your browser:

👉 **[Launch Interactive Google Colab Demo](https://colab.research.google.com/github/eminsk/agentjit/blob/main/notebooks/AgentJIT_Interactive_Demo.ipynb)**

---

## 📦 Links & Community

- 📦 **PyPI:** [pypi.org/project/agentjit](https://pypi.org/project/agentjit/)
- 🐙 **GitHub:** [github.com/eminsk/agentjit](https://github.com/eminsk/agentjit)
- 📝 **License:** Apache 2.0

If you're building autonomous agents in production and want to slash your latency and API costs, give **AgentJIT** a spin!

If you find the project interesting, please consider dropping a **Star ⭐ on [GitHub](https://github.com/eminsk/agentjit)** — it helps other developers discover the library!
