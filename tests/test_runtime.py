import pytest
from agentjit.codegen import CodeGenerator, GuardViolation
from agentjit.runtime import CompiledPipeline
from agentjit.types import ExecutionNode, Guard, SourceType, ValueSource


def test_compiled_pipeline_execution_and_bailout():
    tools = {"increment": lambda x: x + 1}
    node = ExecutionNode(
        node_id=1,
        tool_name="increment",
        argument_bindings={"x": ValueSource(SourceType.ENTRY_ARG, name="val")},
        output_var_name="res",
    )
    guard = Guard(
        target_param="val",
        guard_type="type_match",
        condition_code="isinstance(val, int) and val > 0",
        description="val must be positive int",
    )
    final_src = ValueSource(SourceType.STEP_OUTPUT, step_id=1)

    gen = CodeGenerator(
        entry_params=["val"],
        nodes=[node],
        guards=[guard],
        final_source=final_src,
        tool_registry=tools,
    )
    compilation = gen.compile()

    fallback_called = False

    def fallback_agent(val):
        nonlocal fallback_called
        fallback_called = True
        return f"Fallback handled: {val}"

    pipeline = CompiledPipeline(
        compilation=compilation,
        fallback_fn=fallback_agent,
    )

    # 1. Normal run (compiled hit)
    res = pipeline(10)
    assert res == 11
    assert not fallback_called
    assert pipeline.stats["compiled_hits"] == 1
    assert pipeline.stats["bailouts"] == 0

    # 2. Guard violation -> Automatic bailout to fallback
    res_fallback = pipeline(-5)  # val is not > 0
    assert fallback_called is True
    assert res_fallback == "Fallback handled: -5"
    assert pipeline.stats["bailouts"] == 1
