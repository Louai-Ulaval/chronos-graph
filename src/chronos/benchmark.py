"""Benchmarks.

1. Graph engine: C++20 CSR engine vs NetworkX on a synthetic scale-free call graph
   (build, k-hop BFS, bidirectional shortest path, PageRank). Every timing is paired with a
   correctness check against the NetworkX answer.
2. Retrieval: hybrid GraphRAG vs naive fixed-size chunking at the same token budget.
   For each documented function with resolved callees, the query is its docstring and the
   ground truth is the function plus its direct callees (what an agent needs in context to
   change it safely). Reported: dependency recall, target hit rate, relevant-token share.

    python -m chronos.benchmark --nodes 100000 --repo .
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from pathlib import Path

import networkx as nx
import numpy as np

import chronos_cpp

from .ast_indexer import index_repository
from .hybrid_retriever import HybridRetriever, NaiveChunkRetriever


def synthetic_edges(n: int, m: int = 5, seed: int = 7) -> tuple[np.ndarray, np.ndarray]:
    """Each node calls ~m earlier nodes, biased toward low ids, giving hub-heavy in-degrees."""
    rng = np.random.default_rng(seed)
    src = np.repeat(np.arange(1, n, dtype=np.int64), m)
    dst = np.floor(src * rng.random(src.size) ** 2.5).astype(np.int64)
    keep = dst != src
    return src[keep].astype(np.uint32), dst[keep].astype(np.uint32)


def _ms(t0: float) -> float:
    return (time.perf_counter() - t0) * 1000


def _nx_pagerank(G, alpha: float, tol: float):
    try:
        return nx.pagerank(G, alpha=alpha, tol=tol), "networkx (scipy)"
    except (ImportError, ModuleNotFoundError):
        from networkx.algorithms.link_analysis.pagerank_alg import _pagerank_python

        return _pagerank_python(G, alpha=alpha, tol=tol), "networkx (pure python)"


def bench_graph(n: int = 100_000, m: int = 5, k: int = 3, queries: int = 200, seed: int = 7) -> dict:
    src, dst = synthetic_edges(n, m, seed)
    rng = np.random.default_rng(seed + 1)

    t0 = time.perf_counter()
    g = chronos_cpp.GraphEngine()
    g.add_anonymous_nodes(n)
    g.add_edges(src, dst)
    g.finalize()
    cpp_build = _ms(t0)

    t0 = time.perf_counter()
    G = nx.DiGraph()
    G.add_nodes_from(range(n))
    G.add_edges_from(zip(src.tolist(), dst.tolist()))
    nx_build = _ms(t0)
    U = G.to_undirected()  # built outside the timed region (favours NetworkX)

    seeds = rng.integers(0, n, queries).tolist()
    t0 = time.perf_counter()
    cpp_sizes = [len(g.k_hop([s], k, chronos_cpp.Direction.BOTH)) for s in seeds]
    cpp_khop = _ms(t0)
    t0 = time.perf_counter()
    arr_sizes = [len(g.k_hop_arrays([s], k, chronos_cpp.Direction.BOTH)[0]) for s in seeds]
    cpp_khop_arr = _ms(t0)
    t0 = time.perf_counter()
    nx_sizes = [len(nx.single_source_shortest_path_length(U, s, cutoff=k)) for s in seeds]
    nx_khop = _ms(t0)

    pairs = [(int(a), int(b)) for a, b in rng.integers(0, n, (queries, 2))]
    t0 = time.perf_counter()
    cpp_paths = [g.shortest_path(a, b, chronos_cpp.Direction.OUT) for a, b in pairs]
    cpp_sp = _ms(t0)
    t0 = time.perf_counter()
    nx_paths = []
    for a, b in pairs:
        try:
            nx_paths.append(nx.bidirectional_shortest_path(G, a, b))
        except nx.NetworkXNoPath:
            nx_paths.append([])
    nx_sp = _ms(t0)
    paths_ok = all(len(p) == len(q) for p, q in zip(cpp_paths, nx_paths))
    paths_valid = all(all(G.has_edge(p[i], p[i + 1]) for i in range(len(p) - 1)) for p in cpp_paths)

    tol = 1e-6
    t0 = time.perf_counter()
    pr = g.pagerank(0.85, 200, tol * n)  # same stopping rule as NetworkX: L1 error < N * tol
    cpp_pr = _ms(t0)
    t0 = time.perf_counter()
    nx_pr, nx_pr_impl = _nx_pagerank(G, 0.85, tol)
    nx_pr_ms = _ms(t0)
    nx_vec = np.array([nx_pr[i] for i in range(n)])
    pr_diff = float(np.abs(pr - nx_vec).max())

    stats = g.stats()

    def row(name, c, p, ok):
        return {"op": name, "cpp_ms": round(c, 2), "networkx_ms": round(p, 2),
                "speedup": round(p / c, 1) if c > 0 else None, "matches": ok}

    return {
        "graph": {"nodes": n, "edges": int(stats["edges"]), "max_in_degree": stats["max_in_degree"],
                  "cpp_memory_mb": round(stats["memory_bytes"] / 2**20, 2),
                  "bytes_per_edge": round(stats["memory_bytes"] / max(stats["edges"], 1), 1)},
        "params": {"k": k, "queries": queries, "seed": seed, "edges_per_node": m},
        "results": [
            row("build graph", cpp_build, nx_build, True),
            row(f"{k}-hop BFS x{queries}, dict result", cpp_khop, nx_khop, cpp_sizes == nx_sizes),
            row(f"{k}-hop BFS x{queries}, numpy result", cpp_khop_arr, nx_khop, arr_sizes == nx_sizes),
            row(f"shortest path x{queries} (bidirectional)", cpp_sp, nx_sp, paths_ok and paths_valid),
            row("PageRank", cpp_pr, nx_pr_ms, pr_diff < 1e-4),
        ],
        "pagerank_max_abs_diff": pr_diff,
        "networkx_pagerank_impl": nx_pr_impl,
        "mean_khop_size": round(statistics.mean(cpp_sizes), 1),
        "reachable_pairs": sum(1 for p in cpp_paths if p),
        "platform": f"{platform.system()} {platform.machine()} / Python {platform.python_version()}",
        "networkx_version": nx.__version__,
    }


def bench_retrieval(repo: str | Path = ".", token_budget: int = 2000, max_queries: int = 80, chunk_lines: int = 40) -> dict:
    t0 = time.perf_counter()
    idx = index_repository(repo)
    index_ms = _ms(t0)
    hybrid = HybridRetriever(idx)
    naive = NaiveChunkRetriever(idx, chunk_lines=chunk_lines)

    tasks = []
    for sid in sorted(idx.symbols):
        s = idx.symbols[sid]
        if s.kind not in ("function", "method") or not s.docstring.strip():
            continue
        callees = [c for c in idx.callees(sid) if c != sid and idx.symbols[c].kind in ("function", "method")]
        if not callees:
            continue
        query = s.docstring.strip().split("\n\n")[0].replace("\n", " ")[:300]
        tasks.append((sid, query, {sid, *callees}))
    tasks = tasks[:max_queries]

    def summarize(rows: list[dict]) -> dict:
        if not rows:
            return {}
        return {key: round(statistics.mean(r[key] for r in rows), 4) for key in rows[0]}

    graph_rows, naive_rows = [], []
    for sid, query, truth in tasks:
        r = hybrid.retrieve(query, token_budget=token_budget, render=False)
        got = set(r.ids())
        rel_tokens = sum(idx.symbols[x].token_cost for x in got & truth)
        graph_rows.append({
            "dependency_recall": len(got & truth) / len(truth),
            "target_hit": float(sid in got),
            "relevant_token_share": rel_tokens / max(r.tokens_used, 1),
            "tokens_used": r.tokens_used,
            "latency_ms": r.timings["total_ms"],
        })
        t1 = time.perf_counter()
        chunks, used = naive.retrieve(query, token_budget=token_budget)
        lat = _ms(t1)
        covered = naive.covered_symbols(chunks)
        rel_tokens = sum(idx.symbols[x].token_cost for x in covered & truth)
        naive_rows.append({
            "dependency_recall": len(covered & truth) / len(truth),
            "target_hit": float(sid in covered),
            "relevant_token_share": min(1.0, rel_tokens / max(used, 1)),
            "tokens_used": used,
            "latency_ms": lat,
        })
    return {
        "repo": str(Path(repo).resolve()),
        "index": idx.stats() | {"index_ms": round(index_ms, 1)},
        "token_budget": token_budget,
        "queries": len(tasks),
        "graphrag": summarize(graph_rows),
        "naive_chunking": summarize(naive_rows),
        "chunk_lines": chunk_lines,
    }


def _print(graph: dict | None, retrieval: dict | None) -> None:
    try:
        from rich.console import Console
        from rich.table import Table
    except ImportError:
        print(json.dumps({"graph": graph, "retrieval": retrieval}, indent=2))
        return
    console = Console()
    if graph:
        gi = graph["graph"]
        t = Table(title=f"C++20 engine vs NetworkX - {gi['nodes']:,} nodes / {gi['edges']:,} edges", title_justify="left")
        for col in ("operation", "C++ ms", "NetworkX ms", "speedup", "correct"):
            t.add_column(col, justify="right" if col != "operation" else "left")
        for r in graph["results"]:
            t.add_row(r["op"], f"{r['cpp_ms']:,.2f}", f"{r['networkx_ms']:,.2f}", f"{r['speedup']}x",
                      "yes" if r["matches"] else "NO")
        console.print(t)
        console.print(f"  C++ memory {gi['cpp_memory_mb']} MB ({gi['bytes_per_edge']} B/edge), "
                      f"PageRank max |diff| {graph['pagerank_max_abs_diff']:.2e} vs {graph['networkx_pagerank_impl']}\n")
    if retrieval and retrieval.get("graphrag"):
        t = Table(title=f"GraphRAG vs naive chunking - {retrieval['queries']} queries, budget {retrieval['token_budget']} tokens",
                  title_justify="left")
        for col in ("metric", "GraphRAG", "naive chunks"):
            t.add_column(col, justify="right" if col != "metric" else "left")
        g, nv = retrieval["graphrag"], retrieval["naive_chunking"]
        for key, label in [("dependency_recall", "dependency recall"), ("target_hit", "target hit rate"),
                           ("relevant_token_share", "relevant token share"), ("tokens_used", "tokens used"),
                           ("latency_ms", "latency ms")]:
            t.add_row(label, f"{g[key]:.3f}", f"{nv[key]:.3f}")
        console.print(t)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Chronos-Graph benchmarks")
    ap.add_argument("--nodes", type=int, default=100_000)
    ap.add_argument("--edges-per-node", type=int, default=5)
    ap.add_argument("--hops", type=int, default=3)
    ap.add_argument("--queries", type=int, default=200)
    ap.add_argument("--repo", default=".")
    ap.add_argument("--budget", type=int, default=2000)
    ap.add_argument("--skip-graph", action="store_true")
    ap.add_argument("--skip-retrieval", action="store_true")
    ap.add_argument("--json", default="benchmark_results.json", help="write results here ('' to skip)")
    args = ap.parse_args(argv)

    graph = None if args.skip_graph else bench_graph(args.nodes, args.edges_per_node, args.hops, args.queries)
    retrieval = None if args.skip_retrieval else bench_retrieval(args.repo, args.budget)
    _print(graph, retrieval)
    if args.json:
        Path(args.json).write_text(json.dumps({"graph": graph, "retrieval": retrieval}, indent=2))


if __name__ == "__main__":
    main()
