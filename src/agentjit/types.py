"""Core data structures and types for AgentJIT."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union


class SourceType(str, Enum):
    """Origin of a variable or parameter in the compiled data-flow graph."""
    ENTRY_ARG = "entry_arg"          # Direct parameter passed to the entrypoint function
    STEP_OUTPUT = "step_output"      # Output of a previous tool/step in the trajectory
    NESTED_ACCESS = "nested_access"  # Attribute or dictionary key of an input/step output
    LITERAL = "literal"              # Invariant constant detected across execution
    DICT_SYNTHESIS = "dict_synthesis"# Dynamically constructed dictionary composed of sources
    LIST_SYNTHESIS = "list_synthesis"# Dynamically constructed list composed of sources
    BINARY_OP = "binary_op"          # Synthesized arithmetic operation between two sources (e.g. price * qty)


@dataclass
class ValueSource:
    """Represents the provenance of an argument passed into a step."""
    source_type: SourceType
    name: str = ""                         # Name of argument or variable
    step_id: Optional[int] = None          # If from a prior step
    key_path: List[Union[str, int]] = field(default_factory=list)  # e.g. ["items", 0, "id"]
    literal_value: Any = None
    dict_fields: Dict[str, ValueSource] = field(default_factory=dict)
    list_items: List[ValueSource] = field(default_factory=list)
    left: Optional[ValueSource] = None
    right: Optional[ValueSource] = None
    operator: str = ""

    def render_expression(self, var_map: Dict[int, str]) -> str:
        """Render this source as valid Python expression code."""
        if self.source_type == SourceType.ENTRY_ARG:
            return self.name
        elif self.source_type == SourceType.STEP_OUTPUT:
            return var_map.get(self.step_id, f"step_{self.step_id}_out")
        elif self.source_type == SourceType.NESTED_ACCESS:
            base = var_map.get(self.step_id, self.name)
            expr = base
            for k in self.key_path:
                if isinstance(k, int):
                    expr += f"[{k}]"
                else:
                    expr += f"[{repr(k)}]"
            return expr
        elif self.source_type == SourceType.DICT_SYNTHESIS:
            inner = ", ".join(
                f"{repr(k)}: {v.render_expression(var_map)}"
                for k, v in self.dict_fields.items()
            )
            return "{" + inner + "}"
        elif self.source_type == SourceType.LIST_SYNTHESIS:
            inner = ", ".join(
                v.render_expression(var_map) for v in self.list_items
            )
            return "[" + inner + "]"
        elif self.source_type == SourceType.BINARY_OP:
            if self.left and self.right:
                return f"({self.left.render_expression(var_map)} {self.operator} {self.right.render_expression(var_map)})"
            return repr(self.literal_value)
        elif self.source_type == SourceType.LITERAL:
            return repr(self.literal_value)
        else:
            return repr(self.literal_value)


@dataclass
class TraceStep:
    """Individual execution step (tool call or LLM turn) captured by Tracer."""
    step_id: int
    tool_name: str
    inputs: Dict[str, Any]
    output: Any
    timestamp: float = field(default_factory=time.time)
    duration_ms: float = 0.0
    is_llm_call: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Trajectory:
    """Complete sequence of steps executed by an agent to solve a single task."""
    trajectory_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    task_prompt: Optional[str] = None
    entry_args: Dict[str, Any] = field(default_factory=dict)
    final_result: Any = None
    steps: List[TraceStep] = field(default_factory=list)
    total_duration_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Guard:
    """Speculative assertion checked before running the compiled pipeline."""
    target_param: str
    guard_type: str  # 'type_match', 'non_null', 'value_in', 'custom'
    condition_code: str
    description: str = ""

    def validate(self, kwargs: Dict[str, Any]) -> bool:
        """Evaluate the guard against input kwargs."""
        if self.target_param not in kwargs:
            return False
        val = kwargs[self.target_param]
        try:
            if self.guard_type == "non_null":
                return val is not None
            if self.guard_type == "type_match":
                return True
            return True
        except Exception:
            return False


@dataclass
class ExecutionNode:
    """An optimized node in the compiled execution graph."""
    node_id: int
    tool_name: str
    tool_callable: Optional[Callable] = None
    argument_bindings: Dict[str, ValueSource] = field(default_factory=dict)
    output_var_name: str = ""
    is_pure: bool = False


@dataclass
class CompilationResult:
    """Result of compiling a trajectory into a deterministic pipeline."""
    success: bool
    source_code: str = ""
    compiled_callable: Optional[Callable] = None
    nodes: List[ExecutionNode] = field(default_factory=list)
    guards: List[Guard] = field(default_factory=list)
    entry_parameters: List[str] = field(default_factory=list)
    output_expression: str = ""
    estimated_speedup: float = 1.0
    error: Optional[str] = None


@dataclass
class ExecutionMetrics:
    """Execution telemetry for benchmarking and observability."""
    run_id: str
    duration_ms: float
    is_compiled: bool
    bailed_out: bool = False
    tokens_saved: int = 0
    estimated_cost_saved_usd: float = 0.0
    timestamp: float = field(default_factory=time.time)
