"""Test suite for agentjit.utils zero-dependency helpers."""

from agentjit.utils import format_table


def test_format_table_markdown():
    headers = ["Metric", "Uncompiled", "Compiled", "Speedup"]
    data = [
        ["Latency", "35.2 ms", "0.05 ms", "700x"],
        ["Tokens", "1500", "0", "100%"],
    ]
    table = format_table(data, headers)
    assert "| Metric" in table
    assert "| Latency" in table
    assert "700x" in table
    assert "---" in table
