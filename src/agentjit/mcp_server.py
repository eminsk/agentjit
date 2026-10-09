"""
Native Model Context Protocol (MCP) Server for AgentJIT.
Exposes AI Agent Trajectory JIT Compilation, DAG flow analysis,
speculative guard synthesis, and token/latency savings projection
to Claude Desktop, Cursor, Windsurf, Antigravity, and any MCP client over JSON-RPC 2.0 stdio.

Copyright (c) 2026 eminsk (M_N_Nik@yahoo.com)
Licensed under the Apache License, Version 2.0.
"""

from __future__ import annotations

import contextlib
import json
import os
import platform
import sys
from typing import Any, Callable, Dict, List, Optional

from agentjit import __version__
from agentjit.analyzer import TrajectoryAnalyzer
from agentjit.codegen import CodeGenerator
from agentjit.types import TraceStep, Trajectory

SUPPORTED_PROTOCOL_VERSIONS = ("2025-11-25", "2025-06-18", "2024-11-05")
LATEST_PROTOCOL_VERSION = SUPPORTED_PROTOCOL_VERSIONS[0]

MCP_TOOLS_SCHEMA: List[Dict[str, Any]] = [
    {
        "name": "agentjit_compile",
        "title": "Compile Multi-Step Agent Trajectory to Deterministic Python",
        "description": (
            "Compile a multi-step AI Agent workflow (sequence of tool calls, inputs, and outputs) "
            "into clean, deterministic Python code with input type guards, data-flow piping, "
            "and sub-millisecond execution runtime with zero token cost."
        ),
        "annotations": {
            "readOnlyHint": True,
            "openWorldHint": False,
        },
        "inputSchema": {
            "type": "object",
            "properties": {
                "entry_args": {
                    "type": "object",
                    "description": "Initial input arguments dictionary passed to the agent (e.g. {'user_id': 100, 'region': 'us-east'})",
                },
                "steps": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "tool_name": {"type": "string", "description": "Name of the tool called in this step"},
                            "inputs": {"type": "object", "description": "Arguments passed to the tool"},
                            "output": {"description": "Output returned by the tool (dict, list, or primitive)"},
                            "duration_ms": {"type": "number", "default": 100.0, "description": "Observed execution time in ms"},
                        },
                        "required": ["tool_name", "inputs", "output"],
                    },
                    "description": "Sequential list of execution steps / tool invocations",
                },
                "function_name": {
                    "type": "string",
                    "default": "compiled_agent_pipeline",
                    "description": "Name of the synthesized Python function (default: 'compiled_agent_pipeline')",
                },
                "final_result": {
                    "description": "Optional final return value of the agent trajectory",
                },
            },
            "required": ["entry_args", "steps"],
        },
        "outputSchema": {
            "type": "object",
            "properties": {
                "success": {"type": "boolean"},
                "source_code": {"type": "string"},
                "guards_count": {"type": "integer"},
                "guards": {"type": "array", "items": {"type": "string"}},
                "steps_compiled": {"type": "integer"},
                "function_name": {"type": "string"},
            },
        },
    },
    {
        "name": "agentjit_analyze",
        "title": "Analyze Trajectory Data-Flow & Provenance Graph",
        "description": (
            "Analyze the data dependencies across an AI agent's execution history: maps how "
            "prior tool outputs feed into subsequent steps, isolates constant literals vs generalized "
            "parameters, and builds the DAG data-flow graph."
        ),
        "annotations": {
            "readOnlyHint": True,
            "openWorldHint": False,
        },
        "inputSchema": {
            "type": "object",
            "properties": {
                "entry_args": {
                    "type": "object",
                    "description": "Initial entry arguments dictionary",
                },
                "steps": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "tool_name": {"type": "string"},
                            "inputs": {"type": "object"},
                            "output": {},
                        },
                        "required": ["tool_name", "inputs", "output"],
                    },
                },
            },
            "required": ["entry_args", "steps"],
        },
        "outputSchema": {
            "type": "object",
            "properties": {
                "steps_count": {"type": "integer"},
                "parameter_count": {"type": "integer"},
                "guards_count": {"type": "integer"},
                "data_dependencies": {"type": "array"},
            },
        },
    },
    {
        "name": "agentjit_simulate_savings",
        "title": "Calculate Cost & Latency Savings from JIT Compilation",
        "description": (
            "Calculate estimated token savings, latency drop, and dollar cost reduction "
            "when substituting dynamic LLM agent loops with compiled AgentJIT pipelines."
        ),
        "annotations": {
            "readOnlyHint": True,
            "openWorldHint": False,
        },
        "inputSchema": {
            "type": "object",
            "properties": {
                "invocations": {
                    "type": "integer",
                    "default": 10000,
                    "description": "Projected number of agent workflow executions (default: 10,000)",
                },
                "llm_latency_sec": {
                    "type": "number",
                    "default": 25.0,
                    "description": "Average uncompiled LLM multi-turn latency in seconds (default: 25s)",
                },
                "llm_tokens_per_run": {
                    "type": "integer",
                    "default": 3500,
                    "description": "Average tokens consumed per uncompiled run (default: 3,500)",
                },
                "cost_per_million_tokens": {
                    "type": "number",
                    "default": 3.0,
                    "description": "Blended token cost in USD per 1M tokens (default: $3.00)",
                },
            },
        },
        "outputSchema": {
            "type": "object",
            "properties": {
                "invocations": {"type": "integer"},
                "total_tokens_saved": {"type": "integer"},
                "total_dollar_savings": {"type": "number"},
                "uncompiled_time_hours": {"type": "number"},
                "compiled_time_seconds": {"type": "number"},
                "latency_speedup_factor": {"type": "string"},
            },
        },
    },
    {
        "name": "agentjit_info",
        "title": "Get AgentJIT System & Compiler Runtime Info",
        "description": (
            "Returns AgentJIT version, Python version, Free-Threaded (PEP 703 No-GIL) state, "
            "PyPy JIT details, and compiler optimization capabilities."
        ),
        "annotations": {
            "readOnlyHint": True,
            "openWorldHint": False,
        },
        "inputSchema": {
            "type": "object",
            "properties": {},
        },
        "outputSchema": {
            "type": "object",
            "properties": {
                "version": {"type": "string"},
                "python_version": {"type": "string"},
                "is_free_threaded": {"type": "boolean"},
                "implementation": {"type": "string"},
                "platform": {"type": "string"},
                "compiler_capabilities": {"type": "array"},
            },
        },
    },
]


class AgentJITMCPServer:
    """Model Context Protocol (MCP) JSON-RPC 2.0 stdio server for AgentJIT."""

    def handle_request(self, request: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process a single JSON-RPC 2.0 message and return a response dict (or None for notifications)."""
        method = request.get("method", "")
        req_id = request.get("id")
        params = request.get("params") or {}

        if req_id is None and (method.startswith("notifications/") or method == "initialized"):
            return None

        try:
            if method == "initialize":
                requested_version = params.get("protocolVersion")
                negotiated_version = (
                    requested_version
                    if isinstance(requested_version, str) and requested_version in SUPPORTED_PROTOCOL_VERSIONS
                    else LATEST_PROTOCOL_VERSION
                )
                result = {
                    "protocolVersion": negotiated_version,
                    "capabilities": {"tools": {}},
                    "serverInfo": {
                        "name": "agentjit-mcp",
                        "version": __version__,
                    },
                    "instructions": (
                        "AgentJIT provides Just-In-Time compilation for AI agent trajectories. "
                        "Use 'agentjit_compile' to turn multi-turn tool traces into deterministic "
                        "Python code with guards, 'agentjit_analyze' to map data-flow dependencies, "
                        "and 'agentjit_simulate_savings' to estimate token and latency speedups."
                    ),
                }
                return {"jsonrpc": "2.0", "id": req_id, "result": result}

            if method == "ping":
                return {"jsonrpc": "2.0", "id": req_id, "result": {}}

            if method == "tools/list":
                return {"jsonrpc": "2.0", "id": req_id, "result": {"tools": MCP_TOOLS_SCHEMA}}

            if method == "tools/call":
                tool_name = params.get("name", "")
                args = params.get("arguments") or {}
                with contextlib.redirect_stdout(sys.stderr):
                    tool_output = self._call_tool(tool_name, args)
                call_result: dict[str, Any] = {
                    "content": [
                        {
                            "type": "text",
                            "text": json.dumps(tool_output, ensure_ascii=False, indent=2),
                        }
                    ],
                    "isError": False,
                }
                if isinstance(tool_output, dict):
                    call_result["structuredContent"] = tool_output
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": call_result,
                }

            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {"code": -32601, "message": f"Method not found: {method}"},
            }
        except Exception as exc:
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "content": [{"type": "text", "text": f"Error: {exc}"}],
                    "isError": True,
                },
            }

    def _call_tool(self, name: str, args: Dict[str, Any]) -> Any:
        if name == "agentjit_compile":
            entry_args = dict(args.get("entry_args", {}))
            raw_steps = list(args.get("steps", []))
            func_name = str(args.get("function_name", "compiled_agent_pipeline"))
            final_res = args.get("final_result")

            steps: List[TraceStep] = []
            dummy_tools: Dict[str, Callable] = {}
            for i, s in enumerate(raw_steps, 1):
                t_name = str(s["tool_name"])
                inputs = dict(s.get("inputs", {}))
                output = s.get("output")
                duration = float(s.get("duration_ms", 100.0))
                steps.append(TraceStep(step_id=i, tool_name=t_name, inputs=inputs, output=output, duration_ms=duration))
                if t_name not in dummy_tools:
                    dummy_tools[t_name] = lambda **kw: kw

            if final_res is None and steps:
                final_res = steps[-1].output

            traj = Trajectory(
                entry_args=entry_args,
                steps=steps,
                final_result=final_res,
                total_duration_ms=sum(st.duration_ms for st in steps),
            )

            analyzer = TrajectoryAnalyzer(traj)
            nodes, guards, final_source = analyzer.analyze()
            generator = CodeGenerator(
                entry_params=list(entry_args.keys()),
                nodes=nodes,
                guards=guards,
                final_source=final_source,
                tool_registry=dummy_tools,
                func_name=func_name,
            )
            comp_res = generator.compile()

            rendered_guards = []
            for g in guards:
                rendered_guards.append(f"{g.target_param}: {g.guard_type} ({g.condition_code})")

            return {
                "success": comp_res.success,
                "function_name": func_name,
                "steps_compiled": len(nodes),
                "guards_count": len(guards),
                "guards": rendered_guards,
                "source_code": comp_res.source_code,
                "error": comp_res.error,
            }

        if name == "agentjit_analyze":
            entry_args = dict(args.get("entry_args", {}))
            raw_steps = list(args.get("steps", []))
            steps: List[TraceStep] = []
            for i, s in enumerate(raw_steps, 1):
                steps.append(TraceStep(step_id=i, tool_name=str(s["tool_name"]), inputs=dict(s.get("inputs", {})), output=s.get("output")))

            traj = Trajectory(entry_args=entry_args, steps=steps, final_result=steps[-1].output if steps else None)
            analyzer = TrajectoryAnalyzer(traj)
            nodes, guards, final_source = analyzer.analyze()

            deps = []
            for n in nodes:
                arg_provenances = {k: v.source_type.value for k, v in n.argument_bindings.items()}
                deps.append({
                    "node_id": n.node_id,
                    "tool_name": n.tool_name,
                    "argument_provenance": arg_provenances,
                })

            return {
                "steps_count": len(nodes),
                "parameter_count": len(entry_args),
                "guards_count": len(guards),
                "data_dependencies": deps,
            }

        if name == "agentjit_simulate_savings":
            invocations = int(args.get("invocations", 10000))
            uncompiled_latency_sec = float(args.get("llm_latency_sec", 25.0))
            tokens_per_run = int(args.get("llm_tokens_per_run", 3500))
            cost_per_m = float(args.get("cost_per_million_tokens", 3.0))

            total_tokens = invocations * tokens_per_run
            total_cost_usd = (total_tokens / 1_000_000.0) * cost_per_m
            uncompiled_hours = (invocations * uncompiled_latency_sec) / 3600.0
            # Compiled execution runs in ~0.1 ms (0.0001 sec)
            compiled_seconds = invocations * 0.0001

            return {
                "invocations": invocations,
                "total_tokens_saved": total_tokens,
                "total_dollar_savings": round(total_cost_usd, 2),
                "uncompiled_time_hours": round(uncompiled_hours, 1),
                "compiled_time_seconds": round(compiled_seconds, 3),
                "latency_speedup_factor": f"{int(uncompiled_latency_sec / 0.0001):,}x",
            }

        if name == "agentjit_info":
            is_nogil = getattr(sys, "_is_gil_enabled", lambda: True)() is False
            from agentjit.fasm import is_fasm_available, simd_backend
            return {
                "version": __version__,
                "python_version": sys.version.split()[0],
                "is_free_threaded": is_nogil,
                "implementation": platform.python_implementation(),
                "platform": platform.platform(),
                "simd_backend": simd_backend(),
                "is_fasm_accelerated": is_fasm_available(),
                "compiler_capabilities": [
                    "AST Code Synthesis",
                    "Dynamic Tool Dispatch",
                    "Speculative Guards & Bailout",
                    "FASM Hardware AVX2+FMA/SSE2 Microkernels",
                    "Semantic Prompt Similarity Router",
                    "Sub-millisecond Zero-Token Runtime",
                    "Free-Threaded PEP 703 No-GIL Support",
                ],
            }

        raise ValueError(f"Unknown tool name: {name}")

    def run_stdio(self) -> None:
        """Run the JSON-RPC stdio event loop."""
        if sys.platform == "win32":
            try:
                import msvcrt
                msvcrt.setmode(sys.stdin.fileno(), os.O_BINARY)
                msvcrt.setmode(sys.stdout.fileno(), os.O_BINARY)
            except Exception:
                pass

        while True:
            try:
                line = sys.stdin.readline()
                if not line:
                    break

                line_stripped = line.strip()
                if not line_stripped:
                    continue

                try:
                    request = json.loads(line_stripped)
                except json.JSONDecodeError:
                    continue

                response = self.handle_request(request)
                if response is not None:
                    out = json.dumps(response, ensure_ascii=False) + "\n"
                    sys.stdout.write(out)
                    sys.stdout.flush()

            except (KeyboardInterrupt, BrokenPipeError):
                break
            except Exception as e:
                err_resp = {
                    "jsonrpc": "2.0",
                    "id": None,
                    "error": {"code": -32603, "message": f"Internal error: {e}"},
                }
                sys.stdout.write(json.dumps(err_resp, ensure_ascii=False) + "\n")
                sys.stdout.flush()


def main_mcp() -> None:
    """Entry point for agentjit-mcp standalone command."""
    server = AgentJITMCPServer()
    server.run_stdio()


if __name__ == "__main__":
    main_mcp()
