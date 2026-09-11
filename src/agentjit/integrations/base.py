"""Integrations and adapters for AI tool schemas and external agent frameworks."""

from __future__ import annotations

import json
from typing import Any, Callable, Dict, List, Optional

from agentjit.tracer import Tracer


class ToolDispatcher:
    """Dispatches tool calls from LLM function-call payloads while tracing execution."""

    def __init__(self) -> None:
        self.tools: Dict[str, Callable] = {}
        self.schemas: List[Dict[str, Any]] = []

    def register(self, func: Callable, name: Optional[str] = None, description: str = "") -> Callable:
        """Register a callable tool with optional schema metadata."""
        tool_name = name or getattr(func, "__name__", str(func))
        self.tools[tool_name] = func

        # Build basic OpenAI-compatible function schema
        schema = {
            "type": "function",
            "function": {
                "name": tool_name,
                "description": description or func.__doc__ or f"Execute {tool_name}",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            },
        }
        self.schemas.append(schema)
        return func

    def call_from_llm(self, tool_name: str, arguments_json_or_dict: Any) -> Any:
        """Execute a tool called by an LLM (JSON string or parsed dict)."""
        if tool_name not in self.tools:
            raise KeyError(f"Tool '{tool_name}' is not registered in ToolDispatcher")

        if isinstance(arguments_json_or_dict, str):
            kwargs = json.loads(arguments_json_or_dict)
        elif isinstance(arguments_json_or_dict, dict):
            kwargs = arguments_json_or_dict
        else:
            kwargs = {}

        func = self.tools[tool_name]
        tracer = Tracer.current()

        if tracer is not None:
            # Wrap with active tracer
            wrapped = tracer.wrap_tool(func, name=tool_name)
            return wrapped(**kwargs)

        return func(**kwargs)
