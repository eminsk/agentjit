<div align="center">

# ⚡ AgentJIT

### Just-In-Time Compiler for AI Agent Trajectories
**Compile flaky, 30-second multi-step AI Agent workflows into 5-millisecond deterministic code.**

[![PyPI Version](https://img.shields.io/pypi/v/agentjit.svg?style=flat)](https://pypi.org/project/agentjit/)
[![Conda-Forge](https://img.shields.io/conda/vn/conda-forge/agentjit.svg?style=flat)](https://anaconda.org/conda-forge/agentjit)
[![MSYS2](https://img.shields.io/badge/MSYS2-pacman-orange.svg?style=flat)](https://packages.msys2.org/package/mingw-w64-x86_64-python-agentjit)
[![Debian/Ubuntu](https://img.shields.io/badge/Debian%2FUbuntu-.deb%20package-E95420.svg?style=flat)](https://github.com/eminsk/agentjit/releases)
[![Arch Linux AUR](https://img.shields.io/badge/Arch_Linux-AUR-1793D1.svg?style=flat)](https://aur.archlinux.org/packages/python-agentjit)
[![Python Version](https://img.shields.io/badge/python-3.8%20--%203.15-blue.svg?style=flat)](https://pypi.org/project/agentjit/)
[![PyPy](https://img.shields.io/badge/PyPy-3.8%20--%203.11-orange.svg)](https://www.pypy.org/)
[![Free-Threaded No-GIL](https://img.shields.io/badge/No--GIL%20(PEP%20703)-3.13t%20%7C%203.14t%20%7C%203.15t-blueviolet.svg?style=flat)](https://pypi.org/project/agentjit/)
[![License](https://img.shields.io/badge/license-Apache--2.0-green.svg)](LICENSE)
[![CI Test Suite](https://github.com/eminsk/agentjit/actions/workflows/ci.yml/badge.svg)](https://github.com/eminsk/agentjit/actions/workflows/ci.yml)
[![Speedup](https://img.shields.io/badge/speedup-1000x%2B-orange.svg)]()
[![Token Cost](https://img.shields.io/badge/tokens%20on%20hot%20path-%240.00-brightgreen.svg)]()
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-blueviolet.svg)](https://github.com/eminsk/agentjit)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/eminsk/agentjit/blob/main/notebooks/AgentJIT_Interactive_Demo.ipynb)

[**Quickstart**](#quickstart) • [**Compatibility**](#compatibility) • [**Why AgentJIT?**](#the-problem-in-2026-why-agentjit) • [**Architecture**](#architecture) • [**Benchmarks**](#benchmarks)

</div>

---

## <a id="compatibility"></a>🧩 Universal Compatibility Matrix

| Runtime / Implementation | Supported Versions | Execution Mode | Status |
|:---|:---|:---|:---:|
| **CPython (Standard)** | 3.8, 3.9, 3.10, 3.11, 3.12, 3.13, 3.14, 3.15 | Bytecode + GIL | ✅ Fully Supported |
| **CPython (Free-Threaded)** | 3.13t, 3.14t, 3.15t | Multi-core No-GIL (PEP 703) | ✅ Fully Supported |
| **PyPy (JIT Accelerated)** | 3.8, 3.9, 3.10, 3.11 | High-speed JIT tracing | ✅ Fully Supported |
| **Operating Systems** | Windows (7, 8, 10, 11), Linux, macOS (Intel & Apple Silicon) | x86_64, ARM64 | ✅ Fully Supported |

---

## <a id="the-problem-in-2026-why-agentjit"></a>💥 The Problem in 2026: Why AgentJIT?

In 2026, autonomous AI agents solve real-world workflows across business, DevOps, and data analysis. However, running stochastic LLM loops in production faces four critical barriers:

1. **Massive Latency:** A standard 4-step agent workflow (`think -> tool -> observe -> think`) takes **15 to 45 seconds**.
2. **Exponential Costs:** Running the loop 10,000 times/day costs thousands of dollars in redundant API tokens.
3. **Flakiness & Hallucinations:** Even 98% reliability per step leads to compounding errors across multi-turn trajectories.
4. **Redundant Reasoning:** Most agent invocations execute the *exact same structural trajectory* with slightly different input parameters (e.g. different user IDs or dates).

---

## <a id="architecture"></a>🏗️ Architecture & Trajectory JIT Compilation

Just like **V8** compiles hot JavaScript into machine code, and **PyTorch `torch.compile`** traces dynamic tensors into optimized CUDA kernels, **AgentJIT traces dynamic agent trajectories and compiles them into pure, type-safe, ultra-fast Python code.**

```
       [Dynamic Agent Task]
                │
         (1st run / warmup)
                ▼
     ┌──────────────────────┐
     │   AgentJIT Tracer    │ ── (Captures tool calls, data flow, variables)
     └──────────────────────┘
                │
                ▼
     ┌──────────────────────┐
     │ DAG Flow Analyzer    │ ── (Parameter generalization, dependency graph)
     └──────────────────────┘
                │
                ▼
     ┌──────────────────────┐
     │ AST Code Generator   │ ── (Synthesizes pure Python pipeline + Guards)
     └──────────────────────┘
                │
                ▼
  ┌────────────────────────────┐
  │   Compiled JIT Pipeline    │ ──► Subsequent runs: <1ms, $0 tokens!
  └────────────────────────────┘
                │
       (Guard failure? Deopt!)
                ▼
     [Fall back to LLM Agent]
```

---

## ⚡ Key Features

- 🏎️ **Up to 100,000x Speedup:** Hot paths drop from ~20,000 ms to **< 0.1 ms**.
- 💸 **100% Token Savings:** Once compiled, recurring workflows run completely locally with **0 LLM tokens consumed**.
- 🛡️ **Speculative De-Optimization (Bailout):** Automatically generates runtime input guards. If unexpected data formats or divergent branches appear, AgentJIT transparently falls back to the dynamic LLM agent.
- 🔍 **Transparent & Inspectable:** Inspect the exact Python code generated by the JIT with `agent.source_code`.
- 🧵 **Free-Threaded / No-GIL (PEP 703) Ready:** Thread-safe runtime fully tested on Python 3.13t and 3.14t for true multi-core parallel agent execution without GIL contention.
- 🔌 **Framework Agnostic:** Seamlessly wraps LangChain, CrewAI, AutoGen, OpenAI Tool calls, or native Python functions.

---

## <a id="quickstart"></a>🚀 Quickstart

### Installation

| Platform / Manager | Installation Command |
|---|---|
| **PyPI (pip)** | `pip install agentjit` |
| **PyPI (uv)** | `uv add agentjit` |
| **Conda-Forge** | `conda install -c conda-forge agentjit` |
| **MSYS2 (MinGW-w64)** | `pacman -S mingw-w64-x86_64-python-agentjit` |
| **Ubuntu / Debian (.deb)** | `sudo dpkg -i python3-agentjit_0.1.4-1_all.deb` |
| **Arch Linux (AUR)** | `yay -S python-agentjit` |

```bash
pip install agentjit
# or with uv
uv add agentjit
```

### 10-Second Example

Decorate your agent with `@jit` and mark your tools with `@trace_tool`:

```python
from agentjit import jit, trace_tool

# 1. Define your tools
@trace_tool()
def search_product(name: str):
    return {"name": name, "price": 49.99, "stock": 120}

@trace_tool()
def apply_tax(price: float, tax_rate: float):
    return round(price * (1.0 + tax_rate), 2)

# 2. Decorate your agent with @jit
@jit
def checkout_agent(product_name: str, tax_rate: float):
    # This dynamic workflow could call an LLM (Claude, GPT, Gemini)
    item = search_product(name=product_name)
    total = apply_tax(price=item["price"], tax_rate=tax_rate)
    return {"item": item["name"], "total": total}

# --- Run 1: Warmup & Tracing (runs dynamic agent, compiles to Python) ---
order1 = checkout_agent("Mechanical Keyboard", 0.19)

# --- Run 2+: Instant compiled execution (ZERO tokens, sub-millisecond!) ---
order2 = checkout_agent("Wireless Mouse", 0.19)  # Takes 0.05 ms!
```

---

## 🔎 Inspecting Generated Code

You can view the exact synthesized Python code generated by the JIT at any time:

```python
print(checkout_agent.source_code)
```

**Synthesized Output:**
```python
def compiled_checkout_agent(product_name, tax_rate):
    """JIT-compiled trajectory pipeline generated by AgentJIT.
    Executes deterministically in sub-millisecond time with zero token cost.
    """
    # --- Speculative Guards ---
    if not (product_name is not None):
        raise GuardViolation("Argument 'product_name' must not be None", param="product_name")
    if not (isinstance(product_name, str)):
        raise GuardViolation("Argument 'product_name' must be of type str", param="product_name")

    # --- Execution Steps ---
    step_1_out = _tools['search_product'](name=product_name)
    step_2_out = _tools['apply_tax'](price=step_1_out['price'], tax_rate=tax_rate)

    # --- Return Final Result ---
    return {'item': step_1_out['name'], 'total': step_2_out}
```

---

## <a id="benchmarks"></a>📊 Benchmarks

Benchmark comparing a simulated 3-step reasoning agent (15s latency, 2,500 tokens) vs AgentJIT compiled execution over 100 runs:

| Execution Mode | Mean Latency | 99th Percentile | Cost per 1k runs | Token Usage | Determinism |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Standard LLM Agent** | `14,820 ms` | `22,400 ms` | **\$75.00** | 2,500,000 | ~94% |
| **AgentJIT (Warm Path)** | **`0.08 ms`** | **`0.12 ms`** | **\$0.00** | **0** | **100%** |
| **Improvement** | **185,000x faster** | **186,000x faster** | **100% savings** | **Zero tokens** | **Rock-solid** |

---

## <a id="speculative-execution--bailouts"></a>🛡️ Speculative Execution & Bailouts

What happens when an input is unusual or triggers an unexpected branch?

AgentJIT uses **Speculative De-Optimization**:
1. Input variables are validated against synthesized guards.
2. If any guard fails (e.g. wrong type, missing required key) or a tool raises an unhandled exception, AgentJIT catches `GuardViolation`.
3. It seamlessly bails out to the dynamic LLM agent to handle the edge case.
4. Telemetry records the bailout for future multi-branch specialization.

```python
# Normal input: runs compiled pipeline in 0.08ms
checkout_agent("Monitor", 0.19)

# Divergent input (e.g. invalid type): automatically bails out to dynamic agent
checkout_agent(12345, None)  # Transparently de-optimizes, no crash!
```

---

## 🛠️ Telemetry & Observability

Monitor your compiled agents in real time:

```python
print(checkout_agent.stats)
# Output:
# {
#     "total_calls": 1500,
#     "compiled_hits": 1492,
#     "bailouts": 8,
#     "compiled_hit_rate": 99.47,
#     "total_time_saved_ms": 22380000.0,
#     "total_tokens_saved": 3730000
# }
```

---

## 🗺️ Roadmap for 2026–2027

- [x] **Core Tracer & DAG Flow Analyzer**
- [x] **AST Code Generation with Speculative Guards**
- [x] **De-optimization / Bailout Runtime**
- [x] **`@jit` Decorator with Auto-Warmup**
- [ ] **Multi-Branch Polyhedral JIT:** Merge multiple execution paths into a unified control-flow graph (`if/else` branching synthesis).
- [ ] **eBPF-Isolated Micro-Sandbox:** Ultra-fast sub-millisecond process sandbox for compiled shell actions.
- [ ] **WebAssembly (Wasm) Export:** Compile agent trajectories into standalone Wasm binaries for browser and edge runtime.

---

## 📄 License

AgentJIT is open-source software licensed under the [Apache 2.0 License](LICENSE).
