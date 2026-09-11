"""Data-flow dependency analyzer and graph synthesis for agent trajectories."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple, Union

from agentjit.types import (
    ExecutionNode,
    Guard,
    SourceType,
    TraceStep,
    Trajectory,
    ValueSource,
)


class TrajectoryAnalyzer:
    """Analyzes a recorded agent trajectory to construct a deterministic execution DAG."""

    def __init__(self, trajectory: Trajectory) -> None:
        self.trajectory = trajectory
        self.entry_args = trajectory.entry_args
        self.steps = trajectory.steps

    def analyze(self) -> Tuple[List[ExecutionNode], List[Guard], ValueSource]:
        """Perform data-flow analysis and return execution nodes, guards, and final output source."""
        nodes: List[ExecutionNode] = []
        step_outputs: Dict[int, Any] = {}

        # 1. Analyze each step and determine argument provenance
        for step in self.steps:
            bindings: Dict[str, ValueSource] = {}
            for param_name, param_val in step.inputs.items():
                source = self._resolve_value_source(param_val, step.step_id, step_outputs)
                bindings[param_name] = source

            node = ExecutionNode(
                node_id=step.step_id,
                tool_name=step.tool_name,
                argument_bindings=bindings,
                output_var_name=f"step_{step.step_id}_out",
            )
            nodes.append(node)
            step_outputs[step.step_id] = step.output

        # 2. Determine where the final result comes from
        final_source = self._resolve_final_output(step_outputs)

        # 3. Generate speculative execution guards
        guards = self._generate_guards()

        return nodes, guards, final_source

    def _resolve_value_source(
        self,
        target_val: Any,
        current_step_id: int,
        step_outputs: Dict[int, Any],
    ) -> ValueSource:
        """Trace backwards to locate the origin of target_val."""
        # 1. Direct match with entry argument
        for arg_name, arg_val in self.entry_args.items():
            if self._safe_equals(target_val, arg_val):
                return ValueSource(source_type=SourceType.ENTRY_ARG, name=arg_name)

        # 2. Direct match with output of an earlier step
        for prev_id in range(current_step_id - 1, 0, -1):
            if prev_id in step_outputs and self._safe_equals(target_val, step_outputs[prev_id]):
                return ValueSource(source_type=SourceType.STEP_OUTPUT, step_id=prev_id)

        # 3. Nested match within entry arguments (e.g. user['id'])
        for arg_name, arg_val in self.entry_args.items():
            found, path = self._search_nested(target_val, arg_val)
            if found:
                return ValueSource(
                    source_type=SourceType.NESTED_ACCESS,
                    name=arg_name,
                    key_path=path,
                )

        # 4. Nested match within previous step outputs (e.g. order['customer_id'])
        for prev_id in range(current_step_id - 1, 0, -1):
            if prev_id in step_outputs:
                found, path = self._search_nested(target_val, step_outputs[prev_id])
                if found:
                    return ValueSource(
                        source_type=SourceType.NESTED_ACCESS,
                        step_id=prev_id,
                        key_path=path,
                    )

        # 5. Arithmetic binary operation synthesis (e.g. unit_price * quantity)
        if isinstance(target_val, (int, float)) and not isinstance(target_val, bool):
            bin_op = self._try_synthesize_arithmetic(target_val, current_step_id, step_outputs)
            if bin_op is not None:
                return bin_op

        # 6. Nested dictionary structure check
        if isinstance(target_val, dict):
            dict_sources = {
                k: self._resolve_value_source(v, current_step_id, step_outputs)
                for k, v in target_val.items()
            }
            return ValueSource(
                source_type=SourceType.DICT_SYNTHESIS,
                dict_fields=dict_sources,
            )

        # 7. Nested list structure check
        if isinstance(target_val, (list, tuple)):
            list_sources = [
                self._resolve_value_source(item, current_step_id, step_outputs)
                for item in target_val
            ]
            return ValueSource(
                source_type=SourceType.LIST_SYNTHESIS,
                list_items=list_sources,
            )

        # 8. Invariant literal
        return ValueSource(source_type=SourceType.LITERAL, literal_value=target_val)

    def _try_synthesize_arithmetic(
        self,
        target_val: Union[int, float],
        current_step_id: int,
        step_outputs: Dict[int, Any],
    ) -> Optional[ValueSource]:
        """Attempt to synthesize a binary arithmetic operation matching target_val."""
        candidates: List[Tuple[Union[int, float], ValueSource]] = []

        # Collect numeric candidates from entry args
        for arg_name, arg_val in self.entry_args.items():
            if isinstance(arg_val, (int, float)) and not isinstance(arg_val, bool):
                candidates.append(
                    (arg_val, ValueSource(source_type=SourceType.ENTRY_ARG, name=arg_name))
                )

        # Collect numeric candidates from previous step outputs
        for prev_id in range(1, current_step_id):
            out = step_outputs.get(prev_id)
            if isinstance(out, (int, float)) and not isinstance(out, bool):
                candidates.append(
                    (out, ValueSource(source_type=SourceType.STEP_OUTPUT, step_id=prev_id))
                )
            elif isinstance(out, dict):
                for k, v in out.items():
                    if isinstance(v, (int, float)) and not isinstance(v, bool):
                        candidates.append(
                            (
                                v,
                                ValueSource(
                                    source_type=SourceType.NESTED_ACCESS,
                                    step_id=prev_id,
                                    key_path=[k],
                                ),
                            )
                        )

        # Test pairwise operations (prefer multiplication and addition)
        ops = [
            ("*", lambda a, b: a * b),
            ("+", lambda a, b: a + b),
            ("-", lambda a, b: a - b),
            ("/", lambda a, b: (a / b) if b != 0 else None),
        ]

        for i in range(len(candidates)):
            for j in range(len(candidates)):
                if i == j:
                    continue
                v1, src1 = candidates[i]
                v2, src2 = candidates[j]

                for op_symbol, op_fn in ops:
                    try:
                        computed = op_fn(v1, v2)
                        if computed is not None and math.isclose(
                            computed, target_val, rel_tol=1e-5, abs_tol=1e-5
                        ):
                            return ValueSource(
                                source_type=SourceType.BINARY_OP,
                                left=src1,
                                right=src2,
                                operator=op_symbol,
                            )
                    except Exception:
                        continue

        return None

    def _resolve_final_output(self, step_outputs: Dict[int, Any]) -> ValueSource:
        """Find the source expression for the trajectory's final result."""
        final_res = self.trajectory.final_result
        if not self.steps:
            return ValueSource(source_type=SourceType.LITERAL, literal_value=final_res)

        # 1. Check if final result directly equals a step's output
        for step_id in reversed(sorted(step_outputs.keys())):
            if self._safe_equals(final_res, step_outputs[step_id]):
                return ValueSource(source_type=SourceType.STEP_OUTPUT, step_id=step_id)

        # 2. Check if final result is a dict composed of step outputs/entry args
        if isinstance(final_res, dict):
            dict_sources = {
                k: self._resolve_value_source(v, len(self.steps) + 1, step_outputs)
                for k, v in final_res.items()
            }
            return ValueSource(
                source_type=SourceType.DICT_SYNTHESIS,
                dict_fields=dict_sources,
            )

        # 3. Check if final result is a list composed of step outputs/entry args
        if isinstance(final_res, (list, tuple)):
            list_sources = [
                self._resolve_value_source(item, len(self.steps) + 1, step_outputs)
                for item in final_res
            ]
            return ValueSource(
                source_type=SourceType.LIST_SYNTHESIS,
                list_items=list_sources,
            )

        # 4. Check if nested in the last step output
        last_id = self.steps[-1].step_id
        if last_id in step_outputs:
            found, path = self._search_nested(final_res, step_outputs[last_id])
            if found:
                return ValueSource(
                    source_type=SourceType.NESTED_ACCESS,
                    step_id=last_id,
                    key_path=path,
                )

        # Default to output of last step
        return ValueSource(source_type=SourceType.STEP_OUTPUT, step_id=last_id)

    def _generate_guards(self) -> List[Guard]:
        """Generate speculative validation guards for the entry arguments."""
        guards: List[Guard] = []
        for arg_name, arg_val in self.entry_args.items():
            if arg_val is not None:
                # Non-null guard
                guards.append(
                    Guard(
                        target_param=arg_name,
                        guard_type="non_null",
                        condition_code=f"{arg_name} is not None",
                        description=f"Argument '{arg_name}' must not be None",
                    )
                )
                # Type guard
                type_name = type(arg_val).__name__
                guards.append(
                    Guard(
                        target_param=arg_name,
                        guard_type="type_match",
                        condition_code=f"isinstance({arg_name}, {type_name})",
                        description=f"Argument '{arg_name}' must be of type {type_name}",
                    )
                )
        return guards

    def _search_nested(
        self,
        target: Any,
        haystack: Any,
        max_depth: int = 4,
    ) -> Tuple[bool, List[Union[str, int]]]:
        """Recursively search for target inside a nested dictionary/list structure."""
        if max_depth <= 0 or target is None or haystack is None:
            return False, []

        if isinstance(haystack, dict):
            for k, v in haystack.items():
                if self._safe_equals(target, v):
                    return True, [k]
                found, subpath = self._search_nested(target, v, max_depth - 1)
                if found:
                    return True, [k] + subpath

        elif isinstance(haystack, (list, tuple)):
            for idx, item in enumerate(haystack):
                if self._safe_equals(target, item):
                    return True, [idx]
                found, subpath = self._search_nested(target, item, max_depth - 1)
                if found:
                    return True, [idx] + subpath

        return False, []

    @staticmethod
    def _safe_equals(a: Any, b: Any) -> bool:
        """Safe equality comparison avoiding numpy/pandas ambiguity."""
        if a is b:
            return True
        try:
            return bool(a == b)
        except Exception:
            return False
