"""Runtime executor and speculative de-optimization (bailout) manager for AgentJIT."""

from __future__ import annotations

import inspect
import time
import uuid
from typing import Any, Callable, Dict, List, Optional

from agentjit.codegen import GuardViolation
from agentjit.types import CompilationResult, ExecutionMetrics


class CompiledPipeline:
    """Executable wrapper for a JIT-compiled agent workflow with speculative execution and bailout."""

    def __init__(
        self,
        compilation: CompilationResult,
        fallback_fn: Optional[Callable] = None,
        estimated_uncompiled_latency_ms: float = 15000.0,
        estimated_uncompiled_tokens: int = 2500,
    ) -> None:
        self.compilation = compilation
        self.fallback_fn = fallback_fn
        self.estimated_uncompiled_latency_ms = estimated_uncompiled_latency_ms
        self.estimated_uncompiled_tokens = estimated_uncompiled_tokens

        self._call_count = 0
        self._compiled_hits = 0
        self._bailout_count = 0
        self._total_time_saved_ms = 0.0
        self._metrics_history: List[ExecutionMetrics] = []

    @property
    def source_code(self) -> str:
        """View the synthesized Python source code."""
        return self.compilation.source_code

    @property
    def stats(self) -> Dict[str, Any]:
        """Telemetry and performance statistics."""
        return {
            "total_calls": self._call_count,
            "compiled_hits": self._compiled_hits,
            "bailouts": self._bailout_count,
            "compiled_hit_rate": (
                (self._compiled_hits / self._call_count * 100.0)
                if self._call_count > 0
                else 0.0
            ),
            "total_time_saved_ms": self._total_time_saved_ms,
            "total_tokens_saved": self._compiled_hits * self.estimated_uncompiled_tokens,
        }

    def __call__(self, *args, **kwargs) -> Any:
        self._call_count += 1
        run_id = str(uuid.uuid4())[:8]
        t0 = time.perf_counter()

        # Bind args into kwargs matching entry parameters
        bound_kwargs = self._bind_arguments(*args, **kwargs)

        # Attempt compiled execution
        if self.compilation.compiled_callable is not None:
            try:
                result = self.compilation.compiled_callable(**bound_kwargs)
                dt_ms = (time.perf_counter() - t0) * 1000.0

                self._compiled_hits += 1
                time_saved = max(0.0, self.estimated_uncompiled_latency_ms - dt_ms)
                self._total_time_saved_ms += time_saved

                metric = ExecutionMetrics(
                    run_id=run_id,
                    duration_ms=dt_ms,
                    is_compiled=True,
                    bailed_out=False,
                    tokens_saved=self.estimated_uncompiled_tokens,
                    estimated_cost_saved_usd=(self.estimated_uncompiled_tokens / 1000.0) * 0.003,
                )
                self._metrics_history.append(metric)
                return result

            except GuardViolation as gv:
                # Guard failed -> Trigger speculative de-optimization (bailout)
                self._bailout_count += 1
                if self.fallback_fn is not None:
                    return self._execute_fallback(run_id, t0, *args, **kwargs)
                raise gv

            except Exception as ex:
                # Execution error inside tool -> De-optimize to fallback if present
                self._bailout_count += 1
                if self.fallback_fn is not None:
                    return self._execute_fallback(run_id, t0, *args, **kwargs)
                raise ex

        # If not compiled, run fallback directly
        if self.fallback_fn is not None:
            return self._execute_fallback(run_id, t0, *args, **kwargs)
        raise RuntimeError("No compiled callable or fallback function available.")

    def _execute_fallback(self, run_id: str, t0: float, *args, **kwargs) -> Any:
        """De-optimize by falling back to the full dynamic agent loop."""
        res = self.fallback_fn(*args, **kwargs)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        metric = ExecutionMetrics(
            run_id=run_id,
            duration_ms=dt_ms,
            is_compiled=False,
            bailed_out=True,
            tokens_saved=0,
            estimated_cost_saved_usd=0.0,
        )
        self._metrics_history.append(metric)
        return res

    def _bind_arguments(self, *args, **kwargs) -> Dict[str, Any]:
        """Align positional and keyword arguments with entry parameter names."""
        params = self.compilation.entry_parameters
        bound: Dict[str, Any] = {}
        for idx, arg_val in enumerate(args):
            if idx < len(params):
                bound[params[idx]] = arg_val
        for k, v in kwargs.items():
            bound[k] = v
        return bound
