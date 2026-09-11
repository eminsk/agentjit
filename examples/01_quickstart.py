"""01_quickstart.py

Quickest way to get started with AgentJIT.
Demonstrates tracing, compilation, and sub-millisecond execution.
"""

from agentjit import jit, trace_tool

CATALOG = {
    "Mechanical Keyboard": 99.99,
    "Wireless Mouse": 49.99,
    "USB-C Hub": 29.99,
    "OLED Monitor": 399.99,
}


# 1. Define tools with @trace_tool
@trace_tool()
def search_product(name: str):
    price = CATALOG.get(name, 19.99)
    return {"name": name, "price": price, "stock": 100}


@trace_tool()
def apply_tax(price: float, tax_rate: float):
    return round(price * (1.0 + tax_rate), 2)


# 2. Decorate your agent with @jit
@jit
def checkout_agent(product_name: str, tax_rate: float):
    item = search_product(name=product_name)
    total = apply_tax(price=item["price"], tax_rate=tax_rate)
    return {"item": item["name"], "total": total}


def main():
    print("=== AgentJIT Quickstart ===\n")

    # Run 1: Warmup & JIT Compile
    print("[1] First run (Agent executes, records trajectory, compiles to Python)...")
    res1 = checkout_agent("Mechanical Keyboard", 0.19)
    print(f"    Result: {res1}")
    print(f"    Is compiled? -> {checkout_agent.is_compiled}\n")

    # Inspect the generated Python code
    print("[2] Inspecting the generated Python code:")
    print("-" * 60)
    print(checkout_agent.source_code)
    print("-" * 60 + "\n")

    # Run 2: Sub-millisecond compiled execution (zero LLM calls!)
    print("[3] Subsequent runs (Deterministic, zero token cost):")
    for item in ["USB-C Hub", "Wireless Mouse", "OLED Monitor"]:
        res = checkout_agent(item, 0.19)
        print(f"    Fast compiled call -> Item: {res['item']:<18} Total: ${res['total']}")

    # Telemetry
    print("\n[4] Pipeline Telemetry:")
    for k, v in checkout_agent.stats.items():
        print(f"    {k}: {v}")


if __name__ == "__main__":
    main()
