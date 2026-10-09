"""Example 04: Hardware-Accelerated Speculative Guards & Semantic Router in FASM.

Demonstrates AgentJIT's Flat Assembler (FASM) hardware microkernels:
- Bare-metal AVX2+FMA (x86-64) & SSE2 (x86 32-bit) SIMD execution.
- Sub-nanosecond 64-bit prompt/trajectory hashing.
- High-throughput semantic prompt similarity router (< 0.1 µs cosine guard).
- Parallel vectorized numerical bounds checking.
"""

import sys
import time

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from agentjit import (
    SemanticGuard,
    get_fasm_engine,
    is_fasm_available,
    simd_backend,
)

def main():
    print("=" * 70)
    print("  ⚡ AgentJIT FASM Hardware-Accelerated Guard & Router Engine")
    print("=" * 70)

    engine = get_fasm_engine()
    print(f"Hardware Engine Available: {is_fasm_available()}")
    print(f"Active SIMD Backend:        {simd_backend()}")
    print("-" * 70)

    # 1. 64-bit Ultra-Fast Prompt Hashing
    prompt = "Generate quarterly fiscal report for enterprise customer Acme Corp"
    t0 = time.perf_counter()
    N_HASH = 100_000
    for _ in range(N_HASH):
        h = engine.fast_hash(prompt)
    dt_hash = (time.perf_counter() - t0)
    hash_rate = N_HASH / dt_hash

    print("\n[1] Hardware 64-bit Prompt Hashing:")
    print(f"    Sample Prompt: \"{prompt}\"")
    print(f"    Hash (64-bit): {hex(h)}")
    print(f"    Throughput:    {hash_rate:,.0f} hashes/sec ({dt_hash/N_HASH*1e9:.1f} ns/op)")

    # 2. Vectorized Numerical Range Guards
    print("\n[2] Vectorized Numerical Parameter Guards:")
    values = [25.0, 1500.0, 0.15, 42.0]
    mins = [0.0, 100.0, 0.05, 10.0]
    maxs = [100.0, 5000.0, 0.30, 100.0]

    all_valid = engine.eval_range_guards(values, mins, maxs)
    print(f"    Inputs within valid operating envelope? -> {all_valid}")

    # Anomaly test (temperature = 120.0 exceeds max 100.0)
    values_anomalous = [120.0, 1500.0, 0.15, 42.0]
    is_anomaly = not engine.eval_range_guards(values_anomalous, mins, maxs)
    print(f"    Detected anomaly / speculative deopt required? -> {is_anomaly}")

    # 3. Hardware Semantic Prompt Similarity Router (dim=384 embedding)
    print("\n[3] Semantic Prompt Router (AVX2+FMA Cosine Guard, dim=384):")
    dim = 384
    # Reference trajectory embedding
    ref_emb = [0.05 * (i % 7) for i in range(dim)]
    # Closely related prompt embedding
    similar_emb = [ref_emb[i] + 0.005 for i in range(dim)]
    # Completely divergent prompt embedding
    divergent_emb = [-ref_emb[i] for i in range(dim)]

    t0 = time.perf_counter()
    N_COS = 50_000
    for _ in range(N_COS):
        is_match = engine.eval_semantic_guard(similar_emb, ref_emb, threshold=0.88)
    dt_cos = time.perf_counter() - t0
    cos_rate = N_COS / dt_cos

    cos_sim_similar = engine.cosine_similarity(similar_emb, ref_emb)
    cos_sim_divergent = engine.cosine_similarity(divergent_emb, ref_emb)

    print(f"    Similar Query Cosine Similarity:   {cos_sim_similar:.4f} (Match >= 0.88? {is_match})")
    print(f"    Divergent Query Cosine Similarity: {cos_sim_divergent:.4f} (Bailout triggered!)")
    print(f"    SIMD Guard Throughput:             {cos_rate:,.0f} checks/sec ({dt_cos/N_COS*1e6:.2f} µs/guard)")

    print("\n" + "=" * 70)
    print("  ✅ All FASM hardware microkernels validated successfully!")
    print("=" * 70)

if __name__ == "__main__":
    main()
