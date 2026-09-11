"""02_ecommerce_agent.py

Enterprise multi-tool order processing pipeline compiled with AgentJIT.
Demonstrates 4-tool chaining, nested property extraction, and 100% token savings.
"""

from agentjit import jit, trace_tool

# Mock enterprise databases
CUSTOMERS = {
    "CUST-101": {"name": "Alice Corp", "tier": "gold", "country": "US"},
    "CUST-102": {"name": "Bob Logistics", "tier": "silver", "country": "DE"},
    "CUST-103": {"name": "Charlie Inc", "tier": "diamond", "country": "JP"},
}

INVENTORY = {
    "SKU-SERVER": {"stock": 45, "unit_price": 1200.0},
    "SKU-ROUTER": {"stock": 120, "unit_price": 250.0},
}


@trace_tool()
def get_customer_profile(customer_id: str):
    return CUSTOMERS.get(customer_id, {"name": "Unknown", "tier": "bronze", "country": "US"})


@trace_tool()
def calculate_loyalty_discount(tier: str, amount: float):
    discount_rates = {"bronze": 0.0, "silver": 0.05, "gold": 0.15, "diamond": 0.25}
    rate = discount_rates.get(tier, 0.0)
    discount_val = round(amount * rate, 2)
    return {"rate": rate, "discount": discount_val, "final_amount": round(amount - discount_val, 2)}


@trace_tool()
def reserve_stock(sku: str, quantity: int):
    item = INVENTORY.get(sku, {"stock": 0, "unit_price": 0.0})
    if item["stock"] >= quantity:
        return {"reserved": True, "sku": sku, "unit_price": item["unit_price"]}
    return {"reserved": False, "sku": sku, "unit_price": item["unit_price"]}


@trace_tool()
def emit_invoice(customer_id: str, sku: str, quantity: int, total: float):
    return {
        "invoice_no": f"INV-{customer_id}-{sku}",
        "customer": customer_id,
        "sku": sku,
        "quantity": quantity,
        "amount_due": total,
        "status": "ISSUED",
    }


@jit
def enterprise_order_agent(customer_id: str, sku: str, quantity: int):
    # Step 1: Lookup customer
    cust = get_customer_profile(customer_id=customer_id)

    # Step 2: Check & reserve inventory
    stock_info = reserve_stock(sku=sku, quantity=quantity)
    base_total = stock_info["unit_price"] * quantity

    # Step 3: Apply dynamic loyalty discount
    pricing = calculate_loyalty_discount(tier=cust["tier"], amount=base_total)

    # Step 4: Emit invoice
    invoice = emit_invoice(
        customer_id=customer_id,
        sku=sku,
        quantity=quantity,
        total=pricing["final_amount"],
    )
    return invoice


def main():
    print("=== Enterprise Order Processing with AgentJIT ===\n")

    print("[1] Warmup Run (CUST-101 ordering 2 servers)...")
    res1 = enterprise_order_agent("CUST-101", "SKU-SERVER", 2)
    print(f"    Invoice: {res1}")
    print(f"    Status: JIT Compiled -> {enterprise_order_agent.is_compiled}\n")

    print("[2] Synthesized AST Pipeline:")
    print("-" * 65)
    print(enterprise_order_agent.source_code)
    print("-" * 65 + "\n")

    print("[3] Batch Processing via JIT (Zero Token Cost):")
    orders = [
        ("CUST-102", "SKU-ROUTER", 5),
        ("CUST-103", "SKU-SERVER", 10),
        ("CUST-101", "SKU-ROUTER", 2),
    ]

    for cust_id, sku, qty in orders:
        inv = enterprise_order_agent(cust_id, sku, qty)
        print(f"    Processed {inv['invoice_no']} -> Total: ${inv['amount_due']}")

    print("\n[4] Execution Telemetry:")
    for k, v in enterprise_order_agent.stats.items():
        print(f"    {k}: {v}")


if __name__ == "__main__":
    main()
