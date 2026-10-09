import concurrent.futures
import math
import unittest
from agentjit import (
    CompilationResult,
    GuardViolation,
    SemanticGuard,
    compile_trajectory,
    create_compiled_pipeline,
    jit,
    trace_tool,
)
from agentjit.fasm import (
    FASMHardwareEngine,
    get_fasm_engine,
    is_fasm_available,
    simd_backend,
)
from agentjit.types import TraceStep, Trajectory


def test_fasm_availability_and_backend():
    engine = get_fasm_engine()
    backend = simd_backend()
    assert isinstance(backend, str)
    assert len(backend) > 0

    if is_fasm_available():
        assert "SIMD" in backend
        assert ("AVX2" in backend) or ("SSE2" in backend)
    else:
        assert "Pure Python" in backend


def test_fasm_fast_hash():
    engine = get_fasm_engine()

    # Determinism
    h1 = engine.fast_hash("Analyze invoice 492 for user Alex")
    h2 = engine.fast_hash("Analyze invoice 492 for user Alex")
    assert h1 == h2
    assert isinstance(h1, int)

    # Sensitivity
    h3 = engine.fast_hash("Analyze invoice 493 for user Alex")
    assert h1 != h3

    # Seed sensitivity
    h_seed1 = engine.fast_hash("Same text", seed=42)
    h_seed2 = engine.fast_hash("Same text", seed=99)
    assert h_seed1 != h_seed2

    # Empty string
    h_empty = engine.fast_hash("")
    assert isinstance(h_empty, int)


def test_fasm_null_guards():
    engine = get_fasm_engine()

    # Empty list
    assert engine.eval_null_guards([]) is True

    # All non-null
    assert engine.eval_null_guards([1, "text", {"k": 2}, [3], 4.5]) is True

    # One null
    assert engine.eval_null_guards([1, "text", None, [3]]) is False
    assert engine.eval_null_guards([None]) is False


def test_fasm_range_guards():
    engine = get_fasm_engine()

    # Empty list
    assert engine.eval_range_guards([], [], []) is True

    # Valid ranges
    values = [10.0, 20.0, 30.0, 40.0]
    mins = [5.0, 15.0, 25.0, 35.0]
    maxs = [15.0, 25.0, 35.0, 45.0]
    assert engine.eval_range_guards(values, mins, maxs) is True

    # Violation: value below min
    bad_values_low = [1.0, 20.0, 30.0, 40.0]
    assert engine.eval_range_guards(bad_values_low, mins, maxs) is False

    # Violation: value above max
    bad_values_high = [10.0, 99.0, 30.0, 40.0]
    assert engine.eval_range_guards(bad_values_high, mins, maxs) is False


def test_fasm_vector_dot():
    engine = get_fasm_engine()

    # Orthogonal
    a = [1.0, 0.0]
    b = [0.0, 1.0]
    assert math.isclose(engine.vector_dot(a, b), 0.0, abs_tol=1e-5)

    # Parallel standard dim=384
    dim = 384
    v1 = [0.5] * dim
    v2 = [2.0] * dim
    expected = 0.5 * 2.0 * dim
    assert math.isclose(engine.vector_dot(v1, v2), expected, rel_tol=1e-4)

    # Non-aligned / odd dimension
    dim_odd = 77
    v_odd1 = [1.5] * dim_odd
    v_odd2 = [3.0] * dim_odd
    assert math.isclose(engine.vector_dot(v_odd1, v_odd2), 1.5 * 3.0 * dim_odd, rel_tol=1e-4)


def test_fasm_cosine_similarity():
    engine = get_fasm_engine()

    # Identical vectors
    v1 = [0.1, 0.2, 0.3, 0.4] * 96  # dim = 384
    v2 = [0.2, 0.4, 0.6, 0.8] * 96  # exact multiple
    cos_sim = engine.cosine_similarity(v1, v2)
    assert math.isclose(cos_sim, 1.0, rel_tol=1e-4)

    # Orthogonal vectors
    o1 = [1.0, 0.0] * 192
    o2 = [0.0, 1.0] * 192
    assert math.isclose(engine.cosine_similarity(o1, o2), 0.0, abs_tol=1e-4)

    # Zero vector
    z = [0.0] * 384
    assert engine.cosine_similarity(v1, z) == 0.0


def test_fasm_semantic_guard():
    engine = get_fasm_engine()

    v_prompt = [0.5] * 384
    v_target = [0.5] * 384

    # Matches threshold 0.85
    assert engine.eval_semantic_guard(v_prompt, v_target, threshold=0.85) is True

    # Violates threshold 1.05
    assert engine.eval_semantic_guard(v_prompt, v_target, threshold=1.05) is False

    # Dissimilar vectors
    v_dissimilar = [-0.5] * 384
    assert engine.eval_semantic_guard(v_prompt, v_dissimilar, threshold=0.85) is False


def test_fasm_batch_arithmetic():
    engine = get_fasm_engine()

    a = [10.0, 20.0, 30.0, 40.0]
    b = [2.0, 4.0, 5.0, 8.0]

    # Add
    out_add = engine.batch_arithmetic(a, b, op="add")
    assert out_add == [12.0, 24.0, 35.0, 48.0]

    # Sub
    out_sub = engine.batch_arithmetic(a, b, op="sub")
    assert out_sub == [8.0, 16.0, 25.0, 32.0]

    # Mul
    out_mul = engine.batch_arithmetic(a, b, op="mul")
    assert out_mul == [20.0, 80.0, 150.0, 320.0]

    # Div
    out_div = engine.batch_arithmetic(a, b, op="div")
    assert out_div == [5.0, 5.0, 6.0, 5.0]


def test_fasm_lookup_hash():
    engine = get_fasm_engine()

    table = [0x1111, 0x2222, 0x3333, 0x4444, 0x5555]
    assert engine.lookup_hash(table, 0x3333) == 2
    assert engine.lookup_hash(table, 0x5555) == 4
    assert engine.lookup_hash(table, 0x9999) == -1


def test_fasm_semantic_guard_pipeline_execution_and_bailout():
    # Tools
    def format_greeting(name: str):
        return f"Hello, {name}!"

    tools = {"format_greeting": format_greeting}

    # Recorded trajectory with 384-dim dummy embedding
    dim = 384
    traj_emb = [1.0] * dim

    traj = Trajectory(
        entry_args={"name": "Alex", "query_vec": traj_emb},
        final_result="Hello, Alex!",
        steps=[
            TraceStep(
                step_id=1,
                tool_name="format_greeting",
                inputs={"name": "Alex"},
                output="Hello, Alex!",
            )
        ],
    )

    fallback_called = False

    def fallback_agent(name: str, query_vec: list):
        nonlocal fallback_called
        fallback_called = True
        return f"Fallback: Hello, {name}!"

    pipeline = create_compiled_pipeline(
        trajectory=traj,
        tool_registry=tools,
        fallback_fn=fallback_agent,
    )

    # Inject SemanticGuard into pipeline
    sem_guard = SemanticGuard(
        target_param="query_vec",
        trajectory_embedding=traj_emb,
        similarity_threshold=0.85,
    )
    pipeline.compilation.guards.append(sem_guard)

    # Recompile with semantic guard
    from agentjit.codegen import CodeGenerator
    from agentjit.analyzer import TrajectoryAnalyzer

    analyzer = TrajectoryAnalyzer(traj)
    nodes, guards, final_src = analyzer.analyze()
    guards.append(sem_guard)

    gen = CodeGenerator(
        entry_params=["name", "query_vec"],
        nodes=nodes,
        guards=guards,
        final_source=final_src,
        tool_registry=tools,
    )
    comp = gen.compile()
    assert comp.success is True
    pipeline.compilation = comp

    # 1. Matching prompt embedding -> executes fast path
    res = pipeline(name="Alex", query_vec=[1.0] * dim)
    assert res == "Hello, Alex!"
    assert fallback_called is False
    assert pipeline.stats["compiled_hits"] == 1

    # 2. Divergent prompt embedding -> triggers bailout to fallback_agent!
    res_bailout = pipeline(name="Alex", query_vec=[-1.0] * dim)
    assert res_bailout == "Fallback: Hello, Alex!"
    assert fallback_called is True
    assert pipeline.stats["bailouts"] == 1


def test_fasm_concurrent_multithreading_stress():
    engine = get_fasm_engine()

    def worker(idx: int):
        # Stress test hashing, dot product, and cosine concurrently across threads
        h = engine.fast_hash(f"Worker task prompt #{idx}")
        dot = engine.vector_dot([1.0] * 128, [2.0] * 128)
        cos = engine.cosine_similarity([1.0] * 64, [1.0] * 64)
        return h, dot, cos

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(worker, i) for i in range(50)]
        results = [f.result() for f in futures]

    assert len(results) == 50
    for h, dot, cos in results:
        assert isinstance(h, int)
        assert math.isclose(dot, 256.0, rel_tol=1e-4)
        assert math.isclose(cos, 1.0, rel_tol=1e-4)


class TestFasm(unittest.TestCase):
    """Standard unittest wrapper for non-pytest runners (e.g. CPython 3.16/3.16t No-GIL)."""

    def test_availability(self):
        test_fasm_availability_and_backend()

    def test_hash(self):
        test_fasm_fast_hash()

    def test_null_guards(self):
        test_fasm_null_guards()

    def test_range_guards(self):
        test_fasm_range_guards()

    def test_vector_dot(self):
        test_fasm_vector_dot()

    def test_cosine_similarity(self):
        test_fasm_cosine_similarity()

    def test_semantic_guard(self):
        test_fasm_semantic_guard()

    def test_batch_arithmetic(self):
        test_fasm_batch_arithmetic()

    def test_lookup_hash(self):
        test_fasm_lookup_hash()

    def test_pipeline_with_semantic_guard(self):
        test_fasm_semantic_guard_pipeline_execution_and_bailout()

    def test_multithreading(self):
        test_fasm_concurrent_multithreading_stress()


if __name__ == "__main__":
    unittest.main()
