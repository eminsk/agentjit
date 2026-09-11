"""Utility helpers for AgentJIT with zero external dependencies."""

from __future__ import annotations

from typing import Any, Sequence


def format_table(data: Sequence[Sequence[Any]], headers: Sequence[str]) -> str:
    """Format tabular rows and headers into a clean GitHub-flavored Markdown table.

    Operates using pure Python standard library with zero external dependencies.
    """
    str_headers = [str(h) for h in headers]
    str_data = [[str(cell) for cell in row] for row in data]

    col_widths = [
        max(len(str_headers[i]), *(len(row[i]) if i < len(row) else 0 for row in str_data))
        for i in range(len(str_headers))
    ]

    header_row = "| " + " | ".join(f"{h:<{col_widths[i]}}" for i, h in enumerate(str_headers)) + " |"
    separator_row = "| " + " | ".join("-" * col_widths[i] for i in range(len(str_headers))) + " |"
    data_rows = [
        "| " + " | ".join(f"{cell:<{col_widths[i]}}" for i, cell in enumerate(row)) + " |"
        for row in str_data
    ]

    return "\n".join([header_row, separator_row] + data_rows)
