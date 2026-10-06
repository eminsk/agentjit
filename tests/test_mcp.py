"""
Unit Tests for AgentJIT Native MCP Server (JSON-RPC 2.0 stdio)
"""

import json
import pytest

from agentjit import __version__
from agentjit.mcp_server import (
    AgentJITMCPServer,
    LATEST_PROTOCOL_VERSION,
    SUPPORTED_PROTOCOL_VERSIONS,
)


def test_mcp_initialize_negotiation():
    server = AgentJITMCPServer()

    # 1. Request latest version
    req = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {"protocolVersion": "2025-11-25"},
    }
    resp = server.handle_request(req)
    assert resp["id"] == 1
    assert resp["result"]["protocolVersion"] == "2025-11-25"
    assert resp["result"]["serverInfo"]["name"] == "agentjit-mcp"
    assert resp["result"]["serverInfo"]["version"] == __version__

    # 2. Request older supported version
    req_old = {
        "jsonrpc": "2.0",
        "id": 2,
        "method": "initialize",
        "params": {"protocolVersion": "2024-11-05"},
    }
    resp_old = server.handle_request(req_old)
    assert resp_old["result"]["protocolVersion"] == "2024-11-05"

    # 3. Request unknown version -> fallback to latest supported
    req_unknown = {
        "jsonrpc": "2.0",
        "id": 3,
        "method": "initialize",
        "params": {"protocolVersion": "9999-99-99"},
    }
    resp_unknown = server.handle_request(req_unknown)
    assert resp_unknown["result"]["protocolVersion"] == LATEST_PROTOCOL_VERSION


def test_mcp_tools_list_schema():
    server = AgentJITMCPServer()
    resp = server.handle_request({"jsonrpc": "2.0", "id": 10, "method": "tools/list"})
    assert resp["id"] == 10
    tools = resp["result"]["tools"]
    tool_names = [t["name"] for t in tools]
    assert "agentjit_compile" in tool_names
    assert "agentjit_analyze" in tool_names
    assert "agentjit_simulate_savings" in tool_names
    assert "agentjit_info" in tool_names

    for tool in tools:
        assert "name" in tool
        assert "title" in tool
        assert "description" in tool
        assert "annotations" in tool
        assert "readOnlyHint" in tool["annotations"]
        assert "inputSchema" in tool
        assert "outputSchema" in tool


def test_mcp_agentjit_compile():
    server = AgentJITMCPServer()
    req = {
        "jsonrpc": "2.0",
        "id": 20,
        "method": "tools/call",
        "params": {
            "name": "agentjit_compile",
            "arguments": {
                "entry_args": {"user_id": 42},
                "steps": [
                    {
                        "tool_name": "fetch_user",
                        "inputs": {"id": 42},
                        "output": {"tier": "platinum", "credit": 500},
                    },
                    {
                        "tool_name": "apply_bonus",
                        "inputs": {"tier": "platinum"},
                        "output": {"bonus": 100},
                    },
                ],
                "function_name": "loyalty_pipeline",
            },
        },
    }
    resp = server.handle_request(req)
    assert resp["id"] == 20
    assert resp["result"]["isError"] is False
    content = json.loads(resp["result"]["content"][0]["text"])
    assert content["success"] is True
    assert content["function_name"] == "loyalty_pipeline"
    assert content["steps_compiled"] == 2
    assert "def loyalty_pipeline(user_id):" in content["source_code"]
    assert "GuardViolation" in content["source_code"]


def test_mcp_agentjit_analyze():
    server = AgentJITMCPServer()
    req = {
        "jsonrpc": "2.0",
        "id": 30,
        "method": "tools/call",
        "params": {
            "name": "agentjit_analyze",
            "arguments": {
                "entry_args": {"order_id": 999},
                "steps": [
                    {
                        "tool_name": "lookup_order",
                        "inputs": {"order_id": 999},
                        "output": {"total": 150.0},
                    }
                ],
            },
        },
    }
    resp = server.handle_request(req)
    content = json.loads(resp["result"]["content"][0]["text"])
    assert content["steps_count"] == 1
    assert content["parameter_count"] == 1


def test_mcp_agentjit_simulate_savings():
    server = AgentJITMCPServer()
    req = {
        "jsonrpc": "2.0",
        "id": 40,
        "method": "tools/call",
        "params": {
            "name": "agentjit_simulate_savings",
            "arguments": {
                "invocations": 5000,
                "llm_latency_sec": 30.0,
                "llm_tokens_per_run": 4000,
                "cost_per_million_tokens": 5.0,
            },
        },
    }
    resp = server.handle_request(req)
    content = json.loads(resp["result"]["content"][0]["text"])
    assert content["total_tokens_saved"] == 20_000_000
    assert content["total_dollar_savings"] == 100.0
    assert content["uncompiled_time_hours"] > 40.0


def test_mcp_agentjit_info():
    server = AgentJITMCPServer()
    req = {
        "jsonrpc": "2.0",
        "id": 50,
        "method": "tools/call",
        "params": {"name": "agentjit_info", "arguments": {}},
    }
    resp = server.handle_request(req)
    content = json.loads(resp["result"]["content"][0]["text"])
    assert content["version"] == __version__
    assert "AST Code Synthesis" in content["compiler_capabilities"]
