"""A deliberately small Datadog log replay service for IncidentLens demos."""

import asyncio
from dataclasses import dataclass
from pathlib import Path

import httpx
from fastapi import FastAPI, Header, HTTPException


app = FastAPI(title="IncidentLens Log Server", version="2.0.0")
DATA_LOG_PATH = Path(__file__).resolve().parent / "data" / "data.log"
SUPPORTED_DATADOG_SITES = {"datadoghq.com", "datadoghq.eu", "us3.datadoghq.com", "us5.datadoghq.com", "ap1.datadoghq.com", "ap2.datadoghq.com", "uk1.datadoghq.com", "ddog-gov.com", "us2.ddog-gov.com"}


@dataclass(frozen=True)
class DatadogWriteConfig:
    api_key: str
    site: str
    service: str

    def __post_init__(self):
        api_key = self.api_key.strip()
        site = self.site.strip().lower().removeprefix("https://").rstrip("/")
        service = self.service.strip()
        if not api_key or not service or site not in SUPPORTED_DATADOG_SITES:
            raise ValueError("A Datadog API key, supported site, and service are required")
        object.__setattr__(self, "api_key", api_key)
        object.__setattr__(self, "site", site)
        object.__setattr__(self, "service", service)


def _intake_url(config: DatadogWriteConfig) -> str:
    return f"https://http-intake.logs.{config.site}/api/v2/logs"


def _status(line: str) -> str:
    line = line.upper()
    if "CRITICAL" in line or "ERROR" in line:
        return "error"
    return "warn" if "WARN" in line else "info"


async def push_to_datadog(lines: list[str], config: DatadogWriteConfig, *, client: httpx.AsyncClient | None = None) -> bool:
    payload = [{"message": line, "service": config.service, "status": _status(line), "ddsource": "incidentlens", "ddtags": "env:prod"} for line in lines]
    owns_client = client is None
    request_client = client or httpx.AsyncClient(timeout=10)
    try:
        for attempt in range(3):
            try:
                response = await request_client.post(_intake_url(config), headers={"DD-API-KEY": config.api_key, "Content-Type": "application/json"}, json=payload)
                if response.status_code == 202:
                    return True
                if response.status_code < 500:
                    return False
            except httpx.HTTPError:
                pass
            if attempt < 2:
                await asyncio.sleep(0.5 * (attempt + 1))
        return False
    finally:
        if owns_client:
            await request_client.aclose()


class LogStreamer:
    def __init__(self):
        self.task: asyncio.Task | None = None
        self.stop_event = asyncio.Event()
        self.stats = {"loaded": 0, "shipped": 0, "failed": 0}

    @property
    def running(self) -> bool:
        return self.task is not None and not self.task.done()

    async def start(self, config: DatadogWriteConfig, duration: int, interval_seconds: float) -> None:
        if self.running:
            raise RuntimeError("Log replay is already running")
        if not DATA_LOG_PATH.is_file():
            raise FileNotFoundError(f"Log data not found: {DATA_LOG_PATH}")
        lines = [line.strip() for line in DATA_LOG_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
        self.stop_event.clear()
        self.stats = {"loaded": len(lines), "shipped": 0, "failed": 0}
        self.task = asyncio.create_task(self._run(lines, config, max(1, duration), max(0.01, interval_seconds)))

    async def stop(self) -> None:
        self.stop_event.set()
        if self.task:
            await self.task

    async def _run(self, lines, config, duration, interval):
        started = asyncio.get_running_loop().time()
        for line in lines:
            if self.stop_event.is_set() or asyncio.get_running_loop().time() - started >= duration:
                break
            if await push_to_datadog([line], config):
                self.stats["shipped"] += 1
            else:
                self.stats["failed"] += 1
            try:
                await asyncio.wait_for(self.stop_event.wait(), timeout=interval)
            except TimeoutError:
                pass


streamer = LogStreamer()


@app.post("/api/start")
async def start_generation(
    duration: int = 300,
    interval_seconds: float = 0.25,
    x_datadog_api_key: str | None = Header(default=None),
    x_datadog_site: str = Header(default="datadoghq.com"),
    x_datadog_service: str | None = Header(default=None),
):
    try:
        config = DatadogWriteConfig(x_datadog_api_key or "", x_datadog_site, x_datadog_service or "")
        await streamer.start(config, duration, interval_seconds)
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {"status": "running", "duration_seconds": duration, "interval_seconds": interval_seconds, "stats": streamer.stats}


@app.post("/api/stop")
async def stop_generation():
    await streamer.stop()
    return {"status": "stopped", "stats": streamer.stats}
