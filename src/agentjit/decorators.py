"""High-level @jit decorator for automatic tracing, compilation, and execution."""

from __future__ import annotations

import functools
import inspect
import threading
from typing import Any, Callable, Dict, Optional

from agentjit.compiler import create_compiled_pipeline
from agentjit.runtime import CompiledPipeline
from agentjit.tracer import Tracer


class JITWrapper:
    """Manages the lifecycle of a JIT-compiled agent function."""

    def __init__(
        self,
        func: Callable,
        warmup_runs: int = 1,
        auto_compile: bool = True,
        function_name: Optional[str] = None,
    ) -> None:
        self.func = func
        self.warmup_runs = max(1, warmup_runs)
        self.auto_compile = auto_compile
        self.function_name = function_name or f"compiled_{func.__name__}"

        self._compile_lock = threading.Lock()
        self._runs_completed = 0
        self._pipeline: Optional[CompiledPipeline] = None
        self._last_trajectory = None

        functools.update_wrapper(self, func)

    @property
    def is_compiled(self) -> bool:
        """True if the agent workflow has been successfully compiled."""
        return self._pipeline is not None

    @property
    def pipeline(self) -> Optional[CompiledPipeline]:
        """Access the underlying CompiledPipeline instance."""
        return self._pipeline

    @property
    def source_code(self) -> Optional[str]:
        """View the compiled Python code."""
        if self._pipeline:
            return self._pipeline.source_code
        return None

    @property
    def stats(self) -> Dict[str, Any]:
        """Access performance and execution statistics."""
        if self._pipeline:
            s = dict(self._pipeline.stats)
            s["total_calls"] += self._runs_completed
            return s
        return {"status": "uncompiled", "warmup_progress": f"{self._runs_completed}/{self.warmup_runs}"}

    def __call__(self, *args, **kwargs) -> Any:
        # Fast path if already compiled
        if self._pipeline is not None:
            return self._pipeline(*args, **kwargs)

        with self._compile_lock:
            # Double-check inside lock
            if self._pipeline is not None:
                return self._pipeline(*args, **kwargs)

            # Warmup / Tracing phase
            sig = inspect.signature(self.func)
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()
            entry_args = dict(bound.arguments)

            with Tracer(entry_args=entry_args) as tracer:
                result = self.func(*args, **kwargs)
                tracer.set_final_result(result)
                self._last_trajectory = tracer.trajectory

            self._runs_completed += 1

            # Check if warmup threshold reached
            if self.auto_compile and self._runs_completed >= self.warmup_runs:
                self._compile_from_tracer(tracer)

            return result

    def _compile_from_tracer(self, tracer: Tracer) -> None:
        """Trigger compilation using the captured trajectory and tools."""
        try:
            self._pipeline = create_compiled_pipeline(
                trajectory=tracer.trajectory,
                tool_registry=tracer._registered_tools,
                fallback_fn=self.func,
                function_name=self.function_name,
            )
        except Exception as err:
            # If compilation fails, remain in uncompiled fallback mode
            self._pipeline = None


def jit(
    func: Optional[Callable] = None,
    *,
    warmup_runs: int = 1,
    auto_compile: bool = True,
    function_name: Optional[str] = None,
) -> Any:
    """Decorator to JIT-compile an AI agent workflow.

    Usage:
        @jit
        def my_agent(order_id: int, country: str):
            order = fetch_order(order_id)
            tax = calculate_vat(order["amount"], country)
            return update_db(order_id, tax)

        # 1st run: Executes dynamic agent, records trajectory, compiles to pure Python
        result1 = my_agent(402, "Germany")

        # 2nd run: Executes in <1ms directly with ZERO LLM calls and zero token cost!
        result2 = my_agent(403, "France")
    """
    def decorator(fn: Callable) -> JITWrapper:
        return JITWrapper(
            fn,
            warmup_runs=warmup_runs,
            auto_compile=auto_compile,
            function_name=function_name,
        )

    if func is not None:
        return decorator(func)
    return decorator
