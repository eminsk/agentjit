import time
from agentjit.tracer import Tracer, trace_tool


def test_tracer_records_steps():
    @trace_tool()
    def add(a: int, b: int) -> int:
        return a + b

    @trace_tool()
    def multiply(x: int, y: int) -> int:
        return x * y

    with Tracer(task_prompt="Calculate formula", entry_args={"num1": 10, "num2": 5}) as tracer:
        s1 = add(10, 5)
        s2 = multiply(s1, 2)
        tracer.set_final_result(s2)

    traj = tracer.trajectory
    assert len(traj.steps) == 2
    assert traj.steps[0].tool_name == "add"
    assert traj.steps[0].inputs == {"a": 10, "b": 5}
    assert traj.steps[0].output == 15

    assert traj.steps[1].tool_name == "multiply"
    assert traj.steps[1].inputs == {"x": 15, "y": 2}
    assert traj.steps[1].output == 30
    assert traj.final_result == 30
    assert traj.total_duration_ms > 0.0


def test_wrap_tool():
    tracer = Tracer()

    def greet(name: str, shout: bool = False) -> str:
        msg = f"Hello {name}"
        return msg.upper() if shout else msg

    wrapped = tracer.wrap_tool(greet)

    with tracer:
        res = wrapped("Alice", shout=True)
        tracer.set_final_result(res)

    assert tracer.trajectory.steps[0].tool_name == "greet"
    assert tracer.trajectory.steps[0].inputs == {"name": "Alice", "shout": True}
    assert tracer.trajectory.steps[0].output == "HELLO ALICE"
