"""High-level compiler coordinating analysis, code generation, and pipeline assembly."""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional

from agentjit.analyzer import TrajectoryAnalyzer
from agentjit.codegen import CodeGenerator
from agentjit.runtime import CompiledPipeline
from agentjit.types import CompilationResult, Trajectory


def compile_trajectory(
    trajectory: Trajectory,
    tool_registry: Dict[str, Callable],
    function_name: str = "compiled_agent_pipeline",
) -> CompilationResult:
    """Compile a recorded agent trajectory into an optimized Python pipeline.

    Args:
        trajectory: The traced sequence of tool calls and entry parameters.
        tool_registry: Mapping from tool names to their Python callable implementations.
        function_name: Name of the generated Python function.

    Returns:
        CompilationResult containing synthesized source code and executable function.
    """
    analyzer = TrajectoryAnalyzer(trajectory)
    nodes, guards, final_source = analyzer.analyze()

    entry_params = list(trajectory.entry_args.keys())

    generator = CodeGenerator(
        entry_params=entry_params,
        nodes=nodes,
        guards=guards,
        final_source=final_source,
        tool_registry=tool_registry,
        func_name=function_name,
    )

    return generator.compile()


def create_compiled_pipeline(
    trajectory: Trajectory,
    tool_registry: Dict[str, Callable],
    fallback_fn: Optional[Callable] = None,
    function_name: str = "compiled_agent_pipeline",
) -> CompiledPipeline:
    """Compile a trajectory and wrap it in an executable CompiledPipeline."""
    compilation = compile_trajectory(trajectory, tool_registry, function_name)
    if not compilation.success:
        raise RuntimeError(f"Failed to compile trajectory: {compilation.error}")

    return CompiledPipeline(
        compilation=compilation,
        fallback_fn=fallback_fn,
        estimated_uncompiled_latency_ms=max(100.0, trajectory.total_duration_ms),
    )
