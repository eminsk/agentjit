"""03_guard_bailout.py

Demonstration of Speculative Execution and Automatic Bailout (De-optimization).
Shows how AgentJIT safely handles anomalous or edge-case inputs by transparently
falling back to the full dynamic agent.
"""

from agentjit import jit, trace_tool


@trace_tool()
def sanitize_query(query: str):
    return str(query).strip().lower()


@trace_tool()
def search_knowledge_base(clean_query: str):
    kb = {
        "refund policy": "Full refund within 30 days of purchase.",
        "shipping fee": "Free shipping on orders over $50.",
        "contact": "Support email: support@example.com",
    }
    return kb.get(clean_query, f"Article '{clean_query}' not found.")


@jit
def support_faq_agent(topic: str):
    # Dynamic LLM agent simulation (handles edge cases dynamically)
    clean = sanitize_query(query=topic)
    answer = search_knowledge_base(clean_query=clean)
    return {"query": str(topic), "answer": answer}


def main():
    print("=== Speculative Execution & Bailout Demonstration ===\n")

    # Step 1: Warmup run
    print("[1] Warmup run with normal string query:")
    res1 = support_faq_agent("  Refund Policy  ")
    print(f"    Answer: {res1['answer']}")
    print(f"    Compiled? -> {support_faq_agent.is_compiled}\n")

    # Step 2: Valid compiled queries (Fast path)
    print("[2] Fast-path queries (Hits compiled JIT pipeline):")
    res2 = support_faq_agent("shipping fee")
    print(f"    Fast call 1: {res2['answer']}")
    res3 = support_faq_agent("contact")
    print(f"    Fast call 2: {res3['answer']}\n")

    # Step 3: Trigger Guard Violation (Input is not a string, e.g. int 404)
    print("[3] Anomaly query (Passing integer 404 instead of string):")
    print("    Guard will fail -> Triggers automatic speculative bailout to dynamic agent...")
    res4 = support_faq_agent(404)  # Int violates isinstance(topic, str)
    print(f"    De-optimized fallback handled safely -> {res4}\n")

    # Step 4: Inspect telemetry
    print("[4] Observability Metrics:")
    stats = support_faq_agent.stats
    print(f"    Total calls: {stats['total_calls']}")
    print(f"    Compiled hits: {stats['compiled_hits']}")
    print(f"    Bailouts (De-optimizations): {stats['bailouts']}")
    print(f"    Compiled hit rate: {stats['compiled_hit_rate']:.1f}%")
    print(f"    Total tokens saved: {stats['total_tokens_saved']}")


if __name__ == "__main__":
    main()
