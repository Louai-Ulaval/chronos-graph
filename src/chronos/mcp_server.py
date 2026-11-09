"""Model Context Protocol server exposing the code graph, GraphRAG retrieval and the healer.

Run over stdio (the default, for Claude Code / Claude Desktop):

    python -m chronos.mcp_server --repo /path/to/repo

or over HTTP:

    python -m chronos.mcp_server --transport sse --port 8765
    python -m chronos.mcp_server --transport streamable-http --port 8765

Works with the MCP Python SDK 2.x (``MCPServer``) and 1.x (``FastMCP``).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from typing import Literal

from pydantic import BaseModel, Field

try:  # MCP Python SDK >= 2
    from mcp.server.mcpserver import MCPServer as _Server

    MCP_V2 = True
except ImportError:  # MCP Python SDK 1.x
    from mcp.server.fastmcp import FastMCP as _Server  # type: ignore[no-redef]

    MCP_V2 = False

from .sandbox_healer import SelfHealer, run_tests_in_sandbox, try_patch_in_sandbox
from .workspace import Workspace

INSTRUCTIONS = """Chronos-Graph indexes a repository (Python + C/C++) into a call / import /
containment graph backed by a C++20 engine.

Typical flow:
1. `retrieve_context` with a natural-language question to get the relevant code, already
   pruned to a token budget along call-graph edges. Prefer it over reading whole files.
2. `get_symbol`, `graph_neighbors` and `call_path` to inspect specific functions.
3. `run_tests`, then `try_patch` to check an edit in a throwaway sandbox, or `self_heal`
   to let the built-in repair loop search for a fix. Nothing touches the working tree
   unless `self_heal` is called with apply=true."""


class EditSpec(BaseModel):
    file: str = Field(description="Path relative to the repository root")
    search: str = Field(description="Exact text to replace; must match exactly once")
    replace: str = Field(description="Replacement text")


def build_server(root: str | None = None, workspace: Workspace | None = None):
    ws = workspace or Workspace(root)
    mcp = _Server(name="chronos-graph", instructions=INSTRUCTIONS)

    @mcp.tool()
    async def index_repository(path: str | None = None) -> dict:
        """(Re)index a repository. Without `path`, re-indexes the current root (changed files only)."""
        return await asyncio.to_thread(ws.reindex, path)

    @mcp.tool()
    async def search_symbols(query: str, top_k: int = 10) -> dict:
        """Vector search over symbols (names, signatures, docstrings, bodies). Returns ids and scores."""
        index, retriever = await asyncio.to_thread(ws.ensure)
        hits = retriever.vector_search(query, top_k=top_k)
        out = []
        for sid, score in hits:
            s = index.symbols[sid]
            out.append({"id": sid, "kind": s.kind, "file": s.file, "line": s.line, "signature": s.signature,
                        "score": round(score, 4)})
        return {"query": query, "results": out}

    @mcp.tool()
    async def retrieve_context(query: str, token_budget: int = 4000, hops: int = 2,
                               centrality_weight: float = 0.25) -> str:
        """Hybrid GraphRAG: vector seeds, k-hop call-graph expansion and PageRank, pruned to `token_budget`.

        Returns a markdown context block (source grouped by file plus the call edges between them)."""
        _, retriever = await asyncio.to_thread(ws.ensure)
        res = await asyncio.to_thread(retriever.retrieve, query, token_budget, hops, 6, centrality_weight)
        ws.emit({"type": "retrieval", "query": query, "nodes": res.ids(), "seeds": [s for s, _ in res.seeds],
                 "timings": res.timings, "tokens": res.tokens_used, "source": "mcp"})
        footer = (f"\n\n_{len(res.nodes)} symbols, {res.tokens_used}/{token_budget} tokens, "
                  f"{res.candidates} candidates, {res.timings['total_ms']} ms_")
        return res.context + footer

    @mcp.tool()
    async def get_symbol(name: str) -> dict:
        """Source, signature, callers and callees of a symbol (exact id, qualified suffix or short name)."""
        return await asyncio.to_thread(ws.symbol, name)

    @mcp.tool()
    async def graph_neighbors(name: str, hops: int = 1, direction: Literal["out", "in", "both"] = "both") -> dict:
        """Symbols within `hops` edges of `name`. direction=out follows calls, in finds callers."""
        _, retriever = await asyncio.to_thread(ws.ensure)
        sid = ws.resolve(name)
        hood = retriever.neighborhood(sid, hops, direction)
        return {"symbol": sid, "neighbors": [{"id": k, "hops": v} for k, v in sorted(hood.items(), key=lambda kv: kv[1]) if v]}

    @mcp.tool()
    async def call_path(source: str, target: str) -> dict:
        """Shortest call chain from `source` to `target` (bidirectional BFS in the C++ engine)."""
        _, retriever = await asyncio.to_thread(ws.ensure)
        a, b = ws.resolve(source), ws.resolve(target)
        path = retriever.shortest_path(a, b, "out")
        return {"source": a, "target": b, "path": path, "reachable": bool(path)}

    @mcp.tool()
    async def central_symbols(limit: int = 15) -> dict:
        """Most central functions/classes by PageRank: the code everything else leans on."""
        _, retriever = await asyncio.to_thread(ws.ensure)
        top = retriever.top_pagerank(limit, kinds={"function", "method", "class"})
        return {"results": [{"id": sid, "pagerank": round(v, 6)} for sid, v in top]}

    @mcp.tool()
    async def run_tests(target: str | None = None) -> dict:
        """Run pytest (optionally one file / node id) in an isolated sandbox copy of the repository."""
        run = await asyncio.to_thread(run_tests_in_sandbox, ws.root, target)
        return run.to_dict(with_output=True)

    @mcp.tool()
    async def try_patch(edits: list[EditSpec], target: str | None = None) -> dict:
        """Apply search/replace edits in a sandbox copy and run the tests. The working tree is untouched."""
        payload = [e.model_dump() for e in edits]
        try:
            return await asyncio.to_thread(try_patch_in_sandbox, ws.root, payload, target)
        except ValueError as exc:
            return {"error": str(exc)}

    @mcp.tool()
    async def self_heal(test_target: str = "tests", apply: bool = False, max_attempts: int = 40) -> dict:
        """Reproduce failing tests, localise the bug on the call graph, search for a patch in parallel
        sandboxes and verify it against the full target suite. apply=true writes the fix to disk."""
        index, _ = await asyncio.to_thread(ws.ensure)
        healer = SelfHealer(ws.root, test_target, index=index, max_attempts=max_attempts, on_event=ws.emit)
        result = await asyncio.to_thread(healer.heal, apply)
        if result.applied:
            await asyncio.to_thread(ws.reindex)
        d = result.to_dict()
        d["trace"] = [{k: v for k, v in e.items() if k in ("state", "message")} for e in d["trace"]]
        return d

    @mcp.resource("chronos://stats")
    def stats_resource() -> str:
        """Index and engine statistics."""
        return json.dumps(ws.stats(), indent=2)

    @mcp.resource("chronos://symbol/{name}")
    def symbol_resource(name: str) -> str:
        """A symbol's source and its graph neighbourhood."""
        return json.dumps(ws.symbol(name), indent=2)

    @mcp.prompt()
    def debug_failing_test(test_id: str) -> str:
        """Walk through debugging one failing test with the graph tools."""
        return (
            f"The test `{test_id}` is failing. Run `run_tests` with target `{test_id}` to see the failure, "
            "then call `retrieve_context` with the error message to load the code it exercises. "
            "Use `call_path` from the test to the suspect function to confirm the chain, check a fix with "
            "`try_patch`, and only then decide whether to apply it."
        )

    mcp._chronos_workspace = ws  # handy for tests and embedding
    return mcp


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Chronos-Graph MCP server")
    ap.add_argument("--repo", default=os.environ.get("CHRONOS_REPO"), help="repository to index (default: cwd)")
    ap.add_argument("--transport", choices=["stdio", "sse", "streamable-http"], default="stdio")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--eager", action="store_true", help="index at startup instead of on first use")
    args = ap.parse_args(argv)

    server = build_server(args.repo)
    if args.eager:
        server._chronos_workspace.reindex()
    if args.transport == "stdio":
        server.run("stdio")
    elif MCP_V2:
        server.run(args.transport, host=args.host, port=args.port)
    else:
        server.settings.host, server.settings.port = args.host, args.port
        server.run(args.transport)


if __name__ == "__main__":
    sys.exit(main())
