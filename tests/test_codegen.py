import pytest
from agentjit.codegen import CodeGenerator, GuardViolation
from agentjit.types import (
    ExecutionNode,
    Guard,
    SourceType,
    ValueSource,
)


def test_codegen_compiles_and_executes():
    tools = {
        "square": lambda n: n * n,
        "add_offset": lambda val, offset: val + offset,
    }

    node1 = ExecutionNode(
        node_id=1,
        tool_name="square",
        argument_bindings={
            "n": ValueSource(source_type=SourceType.ENTRY_ARG, name="num"),
        },
        output_var_name="step_1_out",
    )

    node2 = ExecutionNode(
        node_id=2,
        tool_name="add_offset",
        argument_bindings={
            "val": ValueSource(source_type=SourceType.STEP_OUTPUT, step_id=1),
            "offset": ValueSource(source_type=SourceType.LITERAL, literal_value=10),
        },
        output_var_name="step_2_out",
    )

    guard = Guard(
        target_param="num",
        guard_type="type_match",
        condition_code="isinstance(num, int)",
        description="num must be an int",
    )

    final_src = ValueSource(source_type=SourceType.STEP_OUTPUT, step_id=2)

    gen = CodeGenerator(
        entry_params=["num"],
        nodes=[node1, node2],
        guards=[guard],
        final_source=final_src,
        tool_registry=tools,
        func_name="compute_pipeline",
    )

    res = gen.compile()
    assert res.success is True
    assert res.compiled_callable is not None
    assert "def compute_pipeline(num):" in res.source_code

    # Test execution
    val = res.compiled_callable(num=5)  # 5*5 + 10 = 35
    assert val == 35

    # Test guard violation
    with pytest.raises(GuardViolation):
        res.compiled_callable(num="not-an-int")
