"""Non-intrusive runtime tracer for capturing agent tool executions and data flow."""

from __future__ import annotations

import contextvars
import functools
import inspect
import time
from typing import Any, Callable, Dict, List, Optional

from agentjit.types import TraceStep, Trajectory

# Context variable to hold the currently active Tracer instance in the async/thread context
_active_tracer: contextvars.ContextVar[Optional[Tracer]] = contextvars.ContextVar(
    "_active_tracer", default=None
)


class Tracer:
    """Captures agent tool calls, arguments, outputs, and execution metrics."""

    def __init__(
        self,
        task_prompt: Optional[str] = None,
        entry_args: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.trajectory = Trajectory(
            task_prompt=task_prompt,
            entry_args=entry_args.copy() if entry_args else {},
        )
        self._token: Optional[contextvars.Token] = None
        self._start_time: float = 0.0
        self._registered_tools: Dict[str, Callable] = {}

    @classmethod
    def current(cls) -> Optional[Tracer]:
        """Return the currently active tracer in the local context, if any."""
        return _active_tracer.get()

    def __enter__(self) -> Tracer:
        self._start_time = time.perf_counter()
        self._token = _active_tracer.set(self)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        duration_ms = (time.perf_counter() - self._start_time) * 1000.0
        self.trajectory.total_duration_ms = duration_ms
        if self._token:
            _active_tracer.reset(self._token)
            self._token = None

    def register_tool(self, name: str, func: Callable) -> None:
        """Keep a reference to tool callables for future AST code generation."""
        self._registered_tools[name] = func

    def record_step(
        self,
        tool_name: str,
        inputs: Dict[str, Any],
        output: Any,
        duration_ms: float = 0.0,
        is_llm_call: bool = False,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> TraceStep:
        """Record an individual step in the trajectory."""
        step_id = len(self.trajectory.steps) + 1
        step = TraceStep(
            step_id=step_id,
            tool_name=tool_name,
            inputs=inputs,
            output=output,
            duration_ms=duration_ms,
            is_llm_call=is_llm_call,
            metadata=metadata or {},
        )
        self.trajectory.steps.append(step)
        return step

    def set_final_result(self, result: Any) -> None:
        """Set the final output produced by the agent workflow."""
        self.trajectory.final_result = result

    def wrap_tool(self, func: Callable, name: Optional[str] = None) -> Callable:
        """Wrap a tool function to automatically record calls when a tracer is active."""
        tool_name = name or getattr(func, "__name__", str(func))
        self.register_tool(tool_name, func)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Bind arguments to parameter names
            sig = inspect.signature(func)
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()
            input_dict = dict(bound.arguments)

            t0 = time.perf_counter()
            out = func(*args, **kwargs)
            dt_ms = (time.perf_counter() - t0) * 1000.0

            active = Tracer.current()
            if active is not None:
                active.record_step(
                    tool_name=tool_name,
                    inputs=input_dict,
                    output=out,
                    duration_ms=dt_ms,
                )
                active.register_tool(tool_name, func)

            return out

        wrapper._agentjit_tool_name = tool_name  # type: ignore[attr-defined]
        wrapper._agentjit_original_func = func  # type: ignore[attr-defined]
        return wrapper


def trace_tool(name: Optional[str] = None) -> Callable:
    """Decorator to mark a function as a traceable agent tool."""
    def decorator(func: Callable) -> Callable:
        tool_name = name or getattr(func, "__name__", str(func))

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            sig = inspect.signature(func)
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()
            input_dict = dict(bound.arguments)

            t0 = time.perf_counter()
            out = func(*args, **kwargs)
            dt_ms = (time.perf_counter() - t0) * 1000.0

            active = Tracer.current()
            if active is not None:
                active.record_step(
                    tool_name=tool_name,
                    inputs=input_dict,
                    output=out,
                    duration_ms=dt_ms,
                )
                active.register_tool(tool_name, func)

            return out

        wrapper._agentjit_tool_name = tool_name  # type: ignore[attr-defined]
        wrapper._agentjit_original_func = func  # type: ignore[attr-defined]
        return wrapper

    return decorator
