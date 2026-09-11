from agentjit.analyzer import TrajectoryAnalyzer
from agentjit.types import SourceType, TraceStep, Trajectory


def test_analyzer_resolves_data_flow():
    # Simulate a 2-step trajectory:
    # 1. get_user(id=101) -> {"id": 101, "tier": "gold"}
    # 2. apply_discount(tier="gold", amount=500) -> 450
    step1 = TraceStep(
        step_id=1,
        tool_name="get_user",
        inputs={"user_id": 101},
        output={"id": 101, "tier": "gold"},
    )
    step2 = TraceStep(
        step_id=2,
        tool_name="apply_discount",
        inputs={"tier": "gold", "amount": 500},
        output=450,
    )

    traj = Trajectory(
        entry_args={"user_id": 101, "amount": 500},
        steps=[step1, step2],
        final_result=450,
    )

    analyzer = TrajectoryAnalyzer(traj)
    nodes, guards, final_source = analyzer.analyze()

    assert len(nodes) == 2
    node1 = nodes[0]
    assert node1.argument_bindings["user_id"].source_type == SourceType.ENTRY_ARG
    assert node1.argument_bindings["user_id"].name == "user_id"

    node2 = nodes[1]
    assert node2.argument_bindings["tier"].source_type == SourceType.NESTED_ACCESS
    assert node2.argument_bindings["tier"].step_id == 1
    assert node2.argument_bindings["tier"].key_path == ["tier"]

    assert node2.argument_bindings["amount"].source_type == SourceType.ENTRY_ARG
    assert node2.argument_bindings["amount"].name == "amount"

    assert final_source.source_type == SourceType.STEP_OUTPUT
    assert final_source.step_id == 2
    assert len(guards) >= 2


def test_analyzer_synthesizes_arithmetic():
    # Test arithmetic deduction: unit_price (50.0) * qty (4) = 200.0
    step1 = TraceStep(
        step_id=1,
        tool_name="lookup_price",
        inputs={"sku": "KEYBOARD"},
        output={"unit_price": 50.0},
    )
    step2 = TraceStep(
        step_id=2,
        tool_name="charge_card",
        inputs={"amount": 200.0},
        output={"status": "PAID"},
    )

    traj = Trajectory(
        entry_args={"sku": "KEYBOARD", "qty": 4},
        steps=[step1, step2],
        final_result={"status": "PAID"},
    )

    analyzer = TrajectoryAnalyzer(traj)
    nodes, guards, final_src = analyzer.analyze()

    amount_src = nodes[1].argument_bindings["amount"]
    assert amount_src.source_type == SourceType.BINARY_OP
    assert amount_src.operator == "*"
