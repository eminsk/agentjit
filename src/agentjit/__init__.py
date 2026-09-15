"""AgentJIT: Just-In-Time Compiler for AI Agent Trajectories.

Compiles non-deterministic, expensive multi-step LLM workflows into ultra-fast
deterministic Python code with zero token costs and automatic speculative fallbacks.
"""

from agentjit.codegen import GuardViolation
from agentjit.compiler import compile_trajectory, create_compiled_pipeline
from agentjit.decorators import jit, JITWrapper
from agentjit.integrations.base import ToolDispatcher
from agentjit.runtime import CompiledPipeline
from agentjit.tracer import Tracer, trace_tool
from agentjit.types import (
    CompilationResult,
    ExecutionMetrics,
    ExecutionNode,
    Guard,
    SourceType,
    TraceStep,
    Trajectory,
    ValueSource,
)
from agentjit.utils import format_table

__version__ = "0.1.4"
__all__ = [
    "jit",
    "JITWrapper",
    "Tracer",
    "trace_tool",
    "compile_trajectory",
    "create_compiled_pipeline",
    "CompiledPipeline",
    "GuardViolation",
    "Trajectory",
    "TraceStep",
    "ExecutionNode",
    "Guard",
    "ValueSource",
    "SourceType",
    "CompilationResult",
    "ExecutionMetrics",
    "ToolDispatcher",
    "format_table",
]
