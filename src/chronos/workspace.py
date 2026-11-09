"""Shared, thread-safe state for the MCP and API servers: one indexed repository."""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from typing import Callable

import chronos_cpp

from .ast_indexer import CodeIndex, CodeIndexer
from .hybrid_retriever import HybridRetriever

EventSink = Callable[[dict], None]


class Workspace:
    def __init__(self, root: str | os.PathLike | None = None, on_event: EventSink | None = None):
        self.root = Path(root or os.environ.get("CHRONOS_REPO") or os.getcwd()).resolve()
        self.indexer = CodeIndexer()
        self.index: CodeIndex | None = None
        self.retriever: HybridRetriever | None = None
        self.on_event = on_event
        self.version = 0
        self._lock = threading.RLock()

    def emit(self, event: dict) -> None:
        if self.on_event:
            try:
                self.on_event(event)
            except Exception:
                pass

    # ---- indexing ---------------------------------------------------------------
    def reindex(self, root: str | os.PathLike | None = None) -> dict:
        with self._lock:
            if root:
                new_root = Path(root).expanduser().resolve()
                if not new_root.exists():
                    raise FileNotFoundError(f"no such path: {new_root}")
                if new_root != self.root:
                    self.indexer = CodeIndexer()
                self.root = new_root
            self.emit({"type": "index", "phase": "start", "root": str(self.root), "ts": time.time()})
            last = [0.0]

            def progress(done: int, total: int, rel: str) -> None:
                now = time.perf_counter()
                if now - last[0] > 0.05 or done == total:
                    last[0] = now
                    self.emit({"type": "index", "phase": "progress", "done": done, "total": total, "file": rel})

            index = self.indexer.index(self.root, progress=progress)
            t0 = time.perf_counter()
            retriever = HybridRetriever(index)
            build_ms = (time.perf_counter() - t0) * 1000
            self.index, self.retriever = index, retriever
            self.version += 1
            stats = self.stats()
            stats["retriever_build_ms"] = round(build_ms, 2)
            self.emit({"type": "index", "phase": "done", "stats": stats, "version": self.version, "ts": time.time()})
            return stats

    def ensure(self) -> tuple[CodeIndex, HybridRetriever]:
        with self._lock:
            if self.index is None or self.retriever is None:
                self.reindex()
            return self.index, self.retriever  # type: ignore[return-value]

    # ---- views ------------------------------------------------------------------
    def stats(self) -> dict:
        index, _ = self.ensure()
        s = index.stats()
        s["engine"] = index.engine.stats()
        s["version"] = self.version
        return s

    def resolve(self, name: str) -> str:
        index, _ = self.ensure()
        hits = index.find(name, limit=1)
        if not hits:
            raise KeyError(f"no symbol matches {name!r}")
        return hits[0].id

    def symbol(self, name: str, with_source: bool = True) -> dict:
        index, _ = self.ensure()
        sid = self.resolve(name)
        sym = index.symbols[sid]
        d = sym.to_dict(with_source=with_source)
        d["callers"] = index.callers(sid)
        d["callees"] = index.callees(sid)
        pr = index.engine.pagerank()
        d["pagerank"] = float(pr[index.id_to_idx[sid]])
        return d

    def graph(self, limit: int = 4000) -> dict:
        index, retriever = self.ensure()
        engine = index.engine
        pr = engine.pagerank()
        deg = engine.degrees(chronos_cpp.Direction.BOTH)
        ids = index.ids
        keep = list(range(len(ids)))
        if len(keep) > limit:
            keep = sorted(keep, key=lambda i: -pr[i])[:limit]
        keep_set = set(keep)
        pr_max = float(pr.max()) if len(pr) else 1.0
        nodes = []
        for i in keep:
            s = index.symbols[ids[i]]
            nodes.append({
                "id": s.id, "name": s.name, "kind": s.kind, "lang": s.language, "file": s.file, "line": s.line,
                "tokens": s.token_cost, "pagerank": round(float(pr[i]) / pr_max, 5), "degree": int(deg[i]),
            })
        src, dst, kind = engine.edge_arrays()
        names = {0: "call", 1: "import", 2: "contains", 3: "inherits", 4: "reference"}
        links = [
            {"source": ids[a], "target": ids[b], "kind": names[int(k)]}
            for a, b, k in zip(src.tolist(), dst.tolist(), kind.tolist())
            if a in keep_set and b in keep_set
        ]
        return {"nodes": nodes, "links": links, "version": self.version, "root": str(self.root)}
