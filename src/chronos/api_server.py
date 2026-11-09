"""FastAPI + WebSocket backend for the telemetry UI.

REST
    GET  /api/health              build info for the C++ engine
    GET  /api/stats               index + engine statistics
    POST /api/index               {"path"?: str}               re-index (or switch repository)
    GET  /api/graph?limit=N       nodes + links for the force graph
    GET  /api/symbol?id=...       source, callers, callees
    POST /api/retrieve            {"query", "token_budget", "hops", "centrality_weight"}
    GET  /api/path?src=&dst=      shortest call chain
    POST /api/heal                {"demo"?: bool, "test_target"?: str, "apply"?: bool}
    GET  /api/heal/{run_id}       final result of a heal run
    POST /api/benchmark           {"nodes"?: int}               C++ vs NetworkX micro-benchmark

WebSocket /ws streams every event: indexing progress, retrievals, healer state
transitions and a 1 Hz telemetry heartbeat.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import resource
import sys
import tempfile
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

import chronos_cpp
import numpy as np
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, Field

from .sandbox_healer import SelfHealer, create_demo_project
from .workspace import Workspace

WEB_DIR = Path(os.environ.get("CHRONOS_WEB_DIR", Path(__file__).resolve().parents[2] / "web"))


class EventBus:
    """Fan-out of events to every connected WebSocket; safe to publish from any thread."""

    def __init__(self) -> None:
        self.loop: asyncio.AbstractEventLoop | None = None
        self.clients: set[asyncio.Queue] = set()
        self.recent: list[dict] = []
        self._lock = threading.Lock()

    def publish(self, event: dict) -> None:
        event.setdefault("ts", time.time())
        with self._lock:
            self.recent.append(event)
            del self.recent[:-300]
        loop = self.loop
        if loop is None or loop.is_closed():
            return
        for q in list(self.clients):
            loop.call_soon_threadsafe(self._put, q, event)

    @staticmethod
    def _put(q: asyncio.Queue, event: dict) -> None:
        if q.full():
            try:
                q.get_nowait()  # drop the oldest event for slow clients
            except asyncio.QueueEmpty:
                pass
        q.put_nowait(event)


class IndexRequest(BaseModel):
    path: str | None = None


class RetrieveRequest(BaseModel):
    query: str
    token_budget: int = Field(4000, ge=50, le=200_000)
    hops: int = Field(2, ge=0, le=6)
    centrality_weight: float = Field(0.25, ge=0.0, le=1.0)
    seeds: int = Field(6, ge=1, le=50)


class HealRequest(BaseModel):
    demo: bool = True
    test_target: str = "tests"
    apply: bool = True
    max_attempts: int = Field(40, ge=1, le=500)


class BenchmarkRequest(BaseModel):
    nodes: int = Field(20_000, ge=100, le=2_000_000)
    queries: int = Field(100, ge=1, le=5000)


def _rss_mb() -> float:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return round(peak / (1024 * 1024) if sys.platform == "darwin" else peak / 1024, 1)


def create_app(root: str | os.PathLike | None = None, telemetry_interval: float = 1.0) -> FastAPI:
    bus = EventBus()
    ws = Workspace(root, on_event=bus.publish)
    heal_runs: dict[str, dict] = {}
    started = time.time()

    async def telemetry() -> None:
        while True:
            await asyncio.sleep(telemetry_interval)
            log = ws.retriever.query_log[-50:] if ws.retriever else []
            lat = [q["ms"] for q in log]
            bus.publish({
                "type": "telemetry",
                "uptime_s": round(time.time() - started, 1),
                "rss_peak_mb": _rss_mb(),
                "nodes": ws.index.engine.num_nodes if ws.index else 0,
                "edges": ws.index.engine.num_edges if ws.index else 0,
                "queries": len(ws.retriever.query_log) if ws.retriever else 0,
                "latency_p50_ms": round(float(np.percentile(lat, 50)), 3) if lat else None,
                "latency_p95_ms": round(float(np.percentile(lat, 95)), 3) if lat else None,
                "clients": len(bus.clients),
                "heal_running": sum(1 for r in heal_runs.values() if r["status"] == "running"),
            })

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        bus.loop = asyncio.get_running_loop()
        task = asyncio.create_task(telemetry())
        try:
            yield
        finally:
            task.cancel()

    app = FastAPI(title="Chronos-Graph", version="1.0.0", lifespan=lifespan)
    app.state.workspace = ws
    app.state.bus = bus
    app.state.heal_runs = heal_runs

    @app.get("/", response_class=HTMLResponse)
    def index_page():
        page = WEB_DIR / "index.html"
        if not page.exists():
            raise HTTPException(404, f"UI not found at {page}")
        return FileResponse(page)

    @app.get("/api/health")
    def health():
        return {
            "ok": True, "version": "1.0.0", "engine": chronos_cpp.__version__, "cplusplus": chronos_cpp.CPLUSPLUS,
            "arch": chronos_cpp.ARCH, "root": str(ws.root), "indexed": ws.index is not None,
        }

    @app.get("/api/stats")
    async def stats():
        return await asyncio.to_thread(ws.stats)

    @app.post("/api/index")
    async def reindex(req: IndexRequest):
        try:
            return await asyncio.to_thread(ws.reindex, req.path)
        except FileNotFoundError as exc:
            raise HTTPException(404, str(exc)) from exc

    @app.get("/api/graph")
    async def graph(limit: int = 4000):
        return await asyncio.to_thread(ws.graph, limit)

    @app.get("/api/symbol")
    async def symbol(id: str):
        try:
            return await asyncio.to_thread(ws.symbol, id)
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc

    @app.post("/api/retrieve")
    async def retrieve(req: RetrieveRequest):
        _, retriever = await asyncio.to_thread(ws.ensure)
        res = await asyncio.to_thread(
            retriever.retrieve, req.query, req.token_budget, req.hops, req.seeds, req.centrality_weight
        )
        bus.publish({"type": "retrieval", "query": req.query, "nodes": res.ids(), "seeds": [s for s, _ in res.seeds],
                     "timings": res.timings, "tokens": res.tokens_used, "budget": req.token_budget, "source": "ui"})
        return res.to_dict()

    @app.get("/api/path")
    async def path(src: str, dst: str):
        _, retriever = await asyncio.to_thread(ws.ensure)
        try:
            a, b = ws.resolve(src), ws.resolve(dst)
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc
        p = retriever.shortest_path(a, b, "out")
        bus.publish({"type": "path", "nodes": p})
        return {"source": a, "target": b, "path": p, "reachable": bool(p)}

    def _heal_job(run_id: str, req: HealRequest) -> None:
        record = heal_runs[run_id]
        try:
            if req.demo:
                demo_root = create_demo_project(Path(tempfile.mkdtemp(prefix="chronos-demo-")) / "demo-inventory")
                bus.publish({"type": "heal", "run_id": run_id, "state": "IDLE",
                             "message": f"Created demo project at {demo_root}"})
                ws.reindex(demo_root)
            index, _ = ws.ensure()
            healer = SelfHealer(ws.root, req.test_target, index=index, max_attempts=req.max_attempts,
                                on_event=lambda e: bus.publish({**e, "run_id": run_id}))
            result = healer.heal(apply=req.apply)
            if result.applied:
                ws.reindex()
            record.update(status="done", result=result.to_dict())
            bus.publish({"type": "heal_done", "run_id": run_id, "success": result.success, "diff": result.diff,
                         "attempts": result.attempts, "duration_ms": result.duration_ms, "message": result.message,
                         "target": (result.patch or {}).get("target")})
        except Exception as exc:  # surface failures to the UI instead of dying silently
            record.update(status="error", error=f"{type(exc).__name__}: {exc}")
            bus.publish({"type": "heal_done", "run_id": run_id, "success": False, "message": record["error"]})

    @app.post("/api/heal")
    def heal(req: HealRequest):
        if any(r["status"] == "running" for r in heal_runs.values()):
            raise HTTPException(409, "a heal run is already in progress")
        run_id = f"heal-{int(time.time() * 1000)}"
        heal_runs[run_id] = {"status": "running", "request": req.model_dump()}
        threading.Thread(target=_heal_job, args=(run_id, req), daemon=True).start()
        return {"run_id": run_id, "status": "running"}

    @app.get("/api/heal/{run_id}")
    def heal_status(run_id: str):
        if run_id not in heal_runs:
            raise HTTPException(404, "unknown run")
        return heal_runs[run_id]

    @app.post("/api/benchmark")
    async def benchmark(req: BenchmarkRequest):
        from .benchmark import bench_graph

        bus.publish({"type": "benchmark", "phase": "start", "nodes": req.nodes})
        result = await asyncio.to_thread(bench_graph, req.nodes, 5, 3, req.queries)
        bus.publish({"type": "benchmark", "phase": "done", "result": result})
        return result

    @app.websocket("/ws")
    async def websocket(sock: WebSocket):
        await sock.accept()
        q: asyncio.Queue = asyncio.Queue(maxsize=1000)
        bus.clients.add(q)
        try:
            await sock.send_json({"type": "hello", "root": str(ws.root), "indexed": ws.index is not None,
                                  "version": ws.version, "recent": bus.recent[-40:]})

            async def pump_out() -> None:
                while True:
                    await sock.send_text(json.dumps(await q.get(), default=str))

            async def pump_in() -> None:
                while True:
                    msg = await sock.receive_text()
                    if msg == "ping":
                        await sock.send_json({"type": "pong", "ts": time.time()})

            done, pending = await asyncio.wait(
                [asyncio.create_task(pump_out()), asyncio.create_task(pump_in())],
                return_when=asyncio.FIRST_COMPLETED,
            )
            for t in pending:
                t.cancel()
            for t in done:
                exc = t.exception()
                if exc and not isinstance(exc, WebSocketDisconnect):
                    raise exc
        except WebSocketDisconnect:
            pass
        finally:
            bus.clients.discard(q)

    return app


def main(argv: list[str] | None = None) -> None:
    import uvicorn

    ap = argparse.ArgumentParser(description="Chronos-Graph telemetry API + UI")
    ap.add_argument("--repo", default=os.environ.get("CHRONOS_REPO"), help="repository to index (default: cwd)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args(argv)
    app = create_app(args.repo)
    threading.Thread(target=app.state.workspace.reindex, daemon=True).start()
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
