"""
AgentJIT Command Line Interface (CLI)
Copyright (c) 2026 eminsk (M_N_Nik@yahoo.com)
Licensed under the Apache License, Version 2.0.
"""

import argparse
import sys

from agentjit import __version__
from agentjit.mcp_server import main_mcp


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="agentjit",
        description="Just-In-Time Compiler for AI Agent Trajectories",
    )
    parser.add_argument("--version", "-v", action="version", version=f"agentjit {__version__}")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # mcp command
    subparsers.add_parser("mcp", help="Run AgentJIT Model Context Protocol (MCP) server over stdio")

    # info command
    subparsers.add_parser("info", help="Show system environment and compilation capabilities")

    args = parser.parse_args()
    if args.command == "mcp":
        main_mcp()
    elif args.command == "info":
        from agentjit.mcp_server import AgentJITMCPServer
        srv = AgentJITMCPServer()
        info = srv._call_tool("agentjit_info", {})
        print(f"AgentJIT Version:    {info['version']}")
        print(f"Python Version:      {info['python_version']}")
        print(f"Implementation:      {info['implementation']}")
        print(f"Free-Threaded NoGIL: {info['is_free_threaded']}")
        print(f"Platform:            {info['platform']}")
        print("\nCapabilities:")
        for cap in info['compiler_capabilities']:
            print(f"  • {cap}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
