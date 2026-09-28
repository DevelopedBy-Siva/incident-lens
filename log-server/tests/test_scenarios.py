import asyncio
import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import AsyncMock, patch

import httpx


ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("incidentlens_log_server", ROOT / "log-server" / "server.py")
server = importlib.util.module_from_spec(spec)
spec.loader.exec_module(server)


def test_only_replay_endpoints_are_public():
    assert set(server.app.openapi()["paths"]) == {"/api/start", "/api/stop"}


def test_datadog_payload_preserves_project_routing_service_and_event_service():
    captured = {}

    def handler(request):
        captured["request"] = request
        return httpx.Response(202, request=request)

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await server.push_to_datadog(
                ["ERROR [payments-api] checkout failed"],
                server.DatadogWriteConfig("key", "datadoghq.com", "checkout"),
                client=client,
            )

    assert asyncio.run(run()) is True
    event = json.loads(captured["request"].content)[0]
    assert event["message"] == "ERROR [payments-api] checkout failed"
    assert event["service"] == "checkout"
    assert event["incidentlens_service"] == "payments-api"
    assert event["status"] == "error"
    assert event["ddtags"] == "env:prod"


def test_replay_sends_each_line_in_order():
    async def run():
        with TemporaryDirectory() as directory:
            data = Path(directory) / "data.log"
            data.write_text("WARN first\nERROR second\n", encoding="utf-8")
            streamer = server.LogStreamer()
            with patch.object(server, "DATA_LOG_PATH", data), patch.object(server, "push_to_datadog", new=AsyncMock(return_value=True)) as ship:
                await streamer.start(server.DatadogWriteConfig("key", "datadoghq.com", "checkout"), 10, 0.01)
                await streamer.task
            return streamer, ship

    streamer, ship = asyncio.run(run())
    assert streamer.stats == {"loaded": 2, "shipped": 2, "failed": 0}
    assert [call.args[0] for call in ship.await_args_list] == [["WARN first"], ["ERROR second"]]


def test_replay_keeps_stacktrace_and_derives_service_from_event():
    records = server._records([
        "2026-09-22T14:00:00Z ERROR [checkout] task failed",
        "Traceback (most recent call last):",
        "  File '/app/task.py', line 1",
        "KeyError: 'order_id'",
        "2026-09-22T14:00:01Z ERROR k8s event Pod checkout-abc container checkout OOMKilled",
    ])

    assert len(records) == 2
    assert "KeyError" in records[0]
    assert server._service(records[0], "project") == "checkout"
    assert server._service(records[1], "project") == "checkout"
