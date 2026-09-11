from agentjit import jit, trace_tool


@trace_tool()
def fetch_order(order_id: int):
    return {"id": order_id, "items_count": 3, "total": 150.0}


@trace_tool()
def calculate_vat(total: float, rate: float):
    return total * rate


@trace_tool()
def create_invoice(order_id: int, tax: float):
    return {"invoice_id": f"INV-{order_id}", "tax": tax}


def test_jit_end_to_end():
    agent_executions = 0

    @jit
    def process_order_agent(order_id: int, rate: float):
        nonlocal agent_executions
        agent_executions += 1
        order = fetch_order(order_id=order_id)
        tax = calculate_vat(total=order["total"], rate=rate)
        return create_invoice(order_id=order_id, tax=tax)

    assert process_order_agent.is_compiled is False

    # 1. Warmup run: executes the dynamic agent function and compiles the trajectory
    res1 = process_order_agent(1001, 0.20)
    assert res1 == {"invoice_id": "INV-1001", "tax": 30.0}
    assert agent_executions == 1
    assert process_order_agent.is_compiled is True
    assert "def compiled_process_order_agent" in process_order_agent.source_code

    # 2. Subsequent runs: executed purely via the compiled Python pipeline without entering the dynamic body!
    res2 = process_order_agent(1002, 0.10)
    assert res2 == {"invoice_id": "INV-1002", "tax": 15.0}
    assert agent_executions == 1  # Agent dynamic body was NOT called!
    assert process_order_agent.stats["compiled_hits"] == 1

    res3 = process_order_agent(2005, 0.25)
    assert res3 == {"invoice_id": "INV-2005", "tax": 37.5}
    assert agent_executions == 1
    assert process_order_agent.stats["compiled_hits"] == 2
    assert process_order_agent.stats["total_tokens_saved"] > 0


def test_concurrent_multithreaded_execution():
    import concurrent.futures

    @jit
    def concurrent_pipeline(order_id: int, rate: float):
        order = fetch_order(order_id=order_id)
        tax = calculate_vat(total=order["total"], rate=rate)
        return create_invoice(order_id=order_id, tax=tax)

    # 40 concurrent invocations across 8 threads (Free-threaded No-GIL stress test)
    def worker(idx: int):
        return concurrent_pipeline(idx, 0.10)

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(worker, 1000 + i) for i in range(40)]
        results = [f.result() for f in futures]

    assert len(results) == 40
    assert concurrent_pipeline.is_compiled is True
    assert concurrent_pipeline.stats["total_calls"] == 40
    assert concurrent_pipeline.stats["compiled_hits"] >= 39
