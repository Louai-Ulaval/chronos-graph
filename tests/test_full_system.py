"""End-to-end tests: C++ bindings, AST indexing, hybrid retrieval, the healer loop, the API and MCP."""

from __future__ import annotations

import asyncio
import json
import os
import random
import textwrap
import threading
import time
from pathlib import Path

import networkx as nx
import numpy as np
import pytest

import chronos_cpp as cpp
from chronos.ast_indexer import CodeIndexer, index_repository
from chronos.hybrid_retriever import HybridRetriever, NaiveChunkRetriever, tokenize
from chronos.sandbox_healer import (
    HealState,
    Sandbox,
    SelfHealer,
    create_demo_project,
    run_tests_in_sandbox,
    try_patch_in_sandbox,
)

# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

MIXED_REPO = {
    "app/__init__.py": "from .service import OrderService\n",
    "app/models.py": '''
        class Base:
            def validate(self) -> bool:
                """Every model is valid unless a subclass says otherwise."""
                return True


        class Order(Base):
            def __init__(self, lines):
                self.lines = lines

            def validate(self) -> bool:
                """An order is valid when the base check passes and it is not empty."""
                return super().validate() and self.total() > 0

            def total(self) -> float:
                """Sum of price times quantity over all order lines."""
                return sum(price * qty for price, qty in self.lines)
    ''',
    "app/pricing.py": '''
        TAX_RATE = 0.15


        def compute_tax(amount: float) -> float:
            """Sales tax for an amount, rounded to cents."""
            return round(amount * TAX_RATE, 2)


        def apply_coupon(amount: float, code: str) -> float:
            """Ten percent off with the SAVE10 coupon."""
            return amount * 0.9 if code == "SAVE10" else amount
    ''',
    "app/service.py": '''
        from . import pricing as p
        from .models import Order


        class OrderService:
            def checkout(self, order: Order) -> float:
                """Validate an order, then charge its total including sales tax."""
                if not order.validate():
                    raise ValueError("invalid order")
                subtotal = order.total()
                return subtotal + p.compute_tax(subtotal)

            def quote(self, lines, coupon: str = "") -> float:
                """Price a quote for raw order lines with an optional coupon."""
                order = Order(lines)
                return p.apply_coupon(order.total(), coupon)
    ''',
    "app/reports.py": '''
        def monthly_tax_report(orders):
            """Validate each order and report the charged total including sales tax for the month."""
            rows = []
            for order in orders:
                charged = order["total"] + order["tax"]
                rows.append({"id": order["id"], "charged": charged, "tax": order["tax"]})
            return rows


        def charge_summary(orders):
            """Summarise how many orders were charged, their total and the sales tax collected."""
            total = sum(o["total"] for o in orders)
            tax = sum(o["tax"] for o in orders)
            return {"orders": len(orders), "total": total, "tax": tax, "charged": total + tax}


        def invalid_order_ratio(orders):
            """Share of orders that failed to validate before they could be charged."""
            bad = sum(1 for o in orders if not o.get("valid", True))
            return bad / max(len(orders), 1)
    ''',
    "app/geometry.py": '''
        import fastgeo


        def length(x: float, y: float) -> float:
            """Euclidean length of a 2-D vector, computed by the native extension."""
            v = fastgeo.Vec(x, y)
            return v.norm()
    ''',
    "native/vec.hpp": '''
        #pragma once
        namespace geo {
        // A 2-D vector.
        class Vec {
         public:
          Vec(double x, double y);
          double dot(const Vec& o) const;
          double norm() const;
         private:
          double x_, y_;
        };
        double distance(const Vec& a, const Vec& b);
        }  // namespace geo
    ''',
    "native/vec.cpp": '''
        #include "vec.hpp"
        #include <cmath>

        namespace geo {

        Vec::Vec(double x, double y) : x_(x), y_{y} {}

        double Vec::dot(const Vec& o) const { return x_ * o.x_ + y_ * o.y_; }

        // Euclidean norm via the dot product.
        double Vec::norm() const {
          return std::sqrt(dot(*this));
        }

        double distance(const Vec& a, const Vec& b) {
          Vec d(a.norm() - b.norm(), 0.0);
          return d.norm();
        }

        }  // namespace geo
    ''',
    "native/bind.cpp": '''
        #include <pybind11/pybind11.h>
        #include "vec.hpp"
        namespace py = pybind11;

        PYBIND11_MODULE(fastgeo, m) {
          py::class_<geo::Vec>(m, "Vec")
              .def(py::init<double, double>())
              .def("dot", &geo::Vec::dot)
              .def("norm", &geo::Vec::norm);
          m.def("distance", &geo::distance);
        }
    ''',
}


def write_tree(root: Path, files: dict[str, str]) -> Path:
    for rel, text in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(textwrap.dedent(text).lstrip("\n"), encoding="utf-8")
    return root


@pytest.fixture(scope="module")
def mixed_repo(tmp_path_factory) -> Path:
    return write_tree(tmp_path_factory.mktemp("mixed"), MIXED_REPO)


@pytest.fixture(scope="module")
def mixed_index(mixed_repo):
    return index_repository(mixed_repo)


def has_edge(index, src, dst, kind="call") -> bool:
    return any(e.src == src and e.dst == dst and e.kind == kind for e in index.edges)


def random_graph(n: int, m: int, seed: int = 1) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    src = rng.integers(0, n, m).astype(np.uint32)
    dst = rng.integers(0, n, m).astype(np.uint32)
    return src, dst


def build_pair(n: int, m: int, seed: int = 1):
    src, dst = random_graph(n, m, seed)
    g = cpp.GraphEngine()
    g.add_anonymous_nodes(n)
    g.add_edges(src, dst)
    G = nx.DiGraph()
    G.add_nodes_from(range(n))
    G.add_edges_from(zip(src.tolist(), dst.tolist()))
    return g, G


# ---------------------------------------------------------------------------
# 1. C++20 engine through pybind11
# ---------------------------------------------------------------------------


class TestEngine:
    def test_build_metadata(self):
        assert cpp.CPLUSPLUS >= 202002
        assert cpp.ARCH in ("arm64", "x86_64", "unknown")

    def test_nodes_edges_and_dedup(self):
        g = cpp.GraphEngine()
        a = g.add_node("a", "function", "x.py", 3, 40)
        b = g.add_node("b")
        assert g.add_node("a") == a  # names are unique
        g.add_edge(a, b)
        g.add_edge(a, b)  # duplicate
        g.add_edge(a, b, cpp.EdgeKind.IMPORT)  # same pair, other kind: still one edge
        assert g.add_edge_by_name("b", "a", cpp.EdgeKind.CONTAINS)
        assert not g.add_edge_by_name("b", "missing")
        assert g.num_nodes == 2 and g.num_edges == 2
        assert g.find("a") == a and g.find("zzz") is None
        info = g.node(a)
        assert (info.name, info.kind, info.file, info.line, info.token_cost) == ("a", "function", "x.py", 3, 40)
        stats = g.stats()
        assert stats["edges_by_kind"]["call"] == 1 and stats["edges_by_kind"]["contains"] == 1
        src, dst, kind = g.edge_arrays()
        assert src.dtype == np.uint32 and len(src) == 2

    def test_errors_map_to_python_exceptions(self):
        g = cpp.GraphEngine()
        g.add_node("a")
        with pytest.raises(IndexError):
            g.add_edge(0, 5)
        with pytest.raises(IndexError):
            g.k_hop([9], 1)
        with pytest.raises(ValueError):
            g.add_edges(np.array([0], dtype=np.uint32), np.array([0, 0], dtype=np.uint32))
        with pytest.raises(ValueError):
            g.pagerank(1.5)

    @pytest.mark.parametrize("direction", ["OUT", "IN", "BOTH"])
    def test_k_hop_matches_networkx(self, direction):
        g, G = build_pair(1500, 4000)
        view = {"OUT": G, "IN": G.reverse(copy=False), "BOTH": G.to_undirected()}[direction]
        d = getattr(cpp.Direction, direction)
        for seed in random.Random(3).sample(range(1500), 25):
            got = g.k_hop([seed], 3, d)
            want = nx.single_source_shortest_path_length(view, seed, cutoff=3)
            assert got == want
            nodes, dist = g.k_hop_arrays([seed], 3, d)
            assert dict(zip(nodes.tolist(), dist.tolist())) == want
            assert g.k_hop_count([seed], 3, d) == len(want)

    def test_multi_source_k_hop(self):
        g, G = build_pair(800, 2000, seed=4)
        got = g.k_hop([1, 2, 3], 2, cpp.Direction.OUT)
        want = {}
        for s in (1, 2, 3):
            for n, dd in nx.single_source_shortest_path_length(G, s, cutoff=2).items():
                want[n] = min(dd, want.get(n, 99))
        assert got == want

    def test_bidirectional_shortest_path(self):
        g, G = build_pair(2000, 5000, seed=7)
        rng = random.Random(11)
        reachable = 0
        for _ in range(200):
            a, b = rng.randrange(2000), rng.randrange(2000)
            path = g.shortest_path(a, b, cpp.Direction.OUT)
            try:
                want = nx.shortest_path_length(G, a, b)
            except nx.NetworkXNoPath:
                assert path == []
                continue
            reachable += 1
            assert len(path) - 1 == want
            assert path[0] == a and path[-1] == b
            assert all(G.has_edge(u, v) for u, v in zip(path, path[1:]))
        assert reachable > 20
        assert g.shortest_path(5, 5) == [5]

    def test_pagerank_matches_networkx(self):
        g, G = build_pair(3000, 9000, seed=5)
        pr = g.pagerank(0.85, 200, 1e-12)
        assert pr.shape == (3000,)
        assert abs(pr.sum() - 1.0) < 1e-9
        want = nx.pagerank(G, alpha=0.85, tol=1e-12, max_iter=500)
        assert np.abs(pr - np.array([want[i] for i in range(3000)])).max() < 1e-8

    def test_personalized_pagerank_concentrates_on_seeds(self):
        g, G = build_pair(1000, 3000, seed=9)
        ppr = g.personalized_pagerank([10], [1.0], 0.85, 200, 1e-12)
        assert abs(ppr.sum() - 1.0) < 1e-9
        assert ppr[10] == ppr.max()
        want = nx.pagerank(G, alpha=0.85, personalization={10: 1.0}, tol=1e-12, max_iter=500)
        assert np.abs(ppr - np.array([want[i] for i in range(1000)])).max() < 1e-8

    def test_prune_respects_budget_and_ranks_seeds_first(self):
        g = cpp.GraphEngine()
        ids = [g.add_node(f"n{i}", "function", "f.py", i, 100) for i in range(30)]
        for i in range(29):
            g.add_edge(ids[i], ids[i + 1])
        g.set_token_cost(ids[5], 50)
        opts = cpp.PruneOptions()
        opts.token_budget = 350
        opts.max_hops = 3
        res = g.prune([ids[10]], [1.0], opts)
        assert res.tokens_used <= 350 and res.token_budget == 350
        assert res.nodes[0].node == ids[10] and res.nodes[0].seed
        assert {n.node for n in res.nodes} <= set(range(7, 14))
        assert res.candidates == 7  # 3 hops in both directions
        assert all(e.src in {n.node for n in res.nodes} and e.dst in {n.node for n in res.nodes} for e in res.edges)
        # relevance decays with distance
        by_node = {n.node: n for n in res.nodes}
        if ids[11] in by_node and ids[12] in by_node:
            assert by_node[ids[11]].relevance > by_node[ids[12]].relevance
        opts.max_nodes = 2
        assert len(g.prune([ids[10]], [1.0], opts).nodes) == 2
        opts.kind_weight = [1.0, 0.5, 0.5, 0.5, 0.5]
        assert opts.kind_weight[1] == 0.5
        with pytest.raises(ValueError):
            opts.kind_weight = [1.0]

    def test_concurrent_queries_release_the_gil(self):
        g, G = build_pair(20_000, 80_000, seed=2)
        expected = g.k_hop_count([1], 3, cpp.Direction.BOTH)
        errors = []

        def worker():
            try:
                for _ in range(30):
                    assert g.k_hop_count([1], 3, cpp.Direction.BOTH) == expected
                    g.pagerank()
            except Exception as exc:  # pragma: no cover - surfaced below
                errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(6)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors


# ---------------------------------------------------------------------------
# 2. AST indexing
# ---------------------------------------------------------------------------


class TestIndexer:
    def test_symbols_and_kinds(self, mixed_index):
        s = mixed_index.symbols
        assert s["app.models.Order"].kind == "class"
        assert s["app.models.Order.total"].kind == "method"
        assert s["app.pricing.compute_tax"].kind == "function"
        assert s["app.service"].kind == "module"
        assert "Sum of price" in s["app.models.Order.total"].docstring
        assert s["app.models.Order.total"].token_cost > 5
        assert mixed_index.stats()["parse_errors"] == 0

    def test_python_call_resolution(self, mixed_index):
        idx = mixed_index
        # annotated parameter -> method on the annotated class
        assert has_edge(idx, "app.service.OrderService.checkout", "app.models.Order.validate")
        assert has_edge(idx, "app.service.OrderService.checkout", "app.models.Order.total")
        # module alias: `from . import pricing as p`; p.compute_tax(...)
        assert has_edge(idx, "app.service.OrderService.checkout", "app.pricing.compute_tax")
        # local variable bound by a constructor call
        assert has_edge(idx, "app.service.OrderService.quote", "app.models.Order")
        assert has_edge(idx, "app.service.OrderService.quote", "app.models.Order.total")
        # super() and self.method()
        assert has_edge(idx, "app.models.Order.validate", "app.models.Base.validate")
        assert has_edge(idx, "app.models.Order.validate", "app.models.Order.total")

    def test_inheritance_imports_and_containment(self, mixed_index):
        idx = mixed_index
        assert has_edge(idx, "app.models.Order", "app.models.Base", "inherits")
        assert has_edge(idx, "app.service", "app.models", "import")
        assert has_edge(idx, "app.service", "app.pricing", "import")
        assert has_edge(idx, "app.models", "app.models.Order", "contains")
        assert has_edge(idx, "app.models.Order", "app.models.Order.total", "contains")

    def test_cpp_scopes_and_calls(self, mixed_index):
        idx = mixed_index
        s = idx.symbols
        assert s["geo::Vec"].kind == "class"
        assert s["geo::Vec::norm"].kind == "method"
        assert s["geo::Vec::Vec"].kind == "method"  # constructor with an initializer list
        assert s["geo::distance"].kind == "function"
        assert "Euclidean norm" in s["geo::Vec::norm"].docstring
        assert has_edge(idx, "geo::Vec::norm", "geo::Vec::dot")
        assert has_edge(idx, "geo::distance", "geo::Vec::norm")
        assert has_edge(idx, "native/vec.cpp", "native/vec.hpp", "import")

    def test_pybind11_bridge_links_python_to_cpp(self, mixed_index):
        idx = mixed_index
        assert has_edge(idx, "app.geometry.length", "geo::Vec")
        assert has_edge(idx, "app.geometry.length", "geo::Vec::norm")
        assert has_edge(idx, "app.geometry", "native/bind.cpp", "import")

    def test_engine_mirrors_index(self, mixed_index):
        g = mixed_index.engine
        assert g.num_nodes == len(mixed_index.symbols)
        i = mixed_index.id_to_idx["app.service.OrderService.checkout"]
        callees = {mixed_index.ids[n] for n in g.neighbors(i, cpp.Direction.OUT)}
        assert "app.pricing.compute_tax" in callees

    def test_find(self, mixed_index):
        assert mixed_index.find("compute_tax")[0].id == "app.pricing.compute_tax"
        assert mixed_index.find("Vec::norm")[0].id == "geo::Vec::norm"
        assert mixed_index.find("OrderService.checkout")[0].id == "app.service.OrderService.checkout"

    def test_incremental_reindex(self, tmp_path):
        repo = write_tree(tmp_path, {"m.py": "def a():\n    return b()\n\ndef b():\n    return 1\n"})
        indexer = CodeIndexer()
        first = indexer.index(repo)
        assert has_edge(first, "m.a", "m.b")
        (repo / "m.py").write_text("def a():\n    return c()\n\ndef c():\n    return 2\n")
        second = indexer.index(repo)
        assert "m.c" in second.symbols and "m.b" not in second.symbols
        assert has_edge(second, "m.a", "m.c")

    def test_syntax_errors_are_reported_not_fatal(self, tmp_path):
        repo = write_tree(tmp_path, {"ok.py": "def f():\n    return 1\n", "bad.py": "def broken(:\n"})
        idx = index_repository(repo)
        assert "ok.f" in idx.symbols
        assert "bad.py" in idx.errors


# ---------------------------------------------------------------------------
# 3. Hybrid retrieval
# ---------------------------------------------------------------------------


class TestRetriever:
    def test_tokenizer(self):
        assert tokenize("parseHTTPResponse snake_case_name") == ["pars", "http", "respons", "snak", "case", "name"]

    def test_vector_search_finds_target(self, mixed_index):
        r = HybridRetriever(mixed_index)
        top = r.vector_search("compute the sales tax for an amount", top_k=3)
        assert top[0][0] == "app.pricing.compute_tax"
        assert r.vector_search("OrderService.checkout", top_k=1)[0][0] == "app.service.OrderService.checkout"

    def test_graph_expansion_pulls_in_dependencies(self, mixed_index):
        r = HybridRetriever(mixed_index)
        res = r.retrieve("validate an order then charge its total including sales tax", token_budget=1500, hops=2)
        ids = set(res.ids())
        assert "app.service.OrderService.checkout" in ids
        # callees in other files arrive through the call graph, not text similarity
        assert {"app.pricing.compute_tax", "app.models.Order.total", "app.models.Order.validate"} <= ids
        assert res.tokens_used <= 1500
        assert any(n.seed for n in res.nodes)
        assert "```python" in res.context and "## Call edges" in res.context
        assert res.timings["total_ms"] >= res.timings["graph_ms"]
        assert all(e["src"] in ids and e["dst"] in ids for e in res.edges)

    def test_budget_is_hard_limit(self, mixed_index):
        r = HybridRetriever(mixed_index)
        for budget in (60, 200, 800):
            assert r.retrieve("order total", token_budget=budget).tokens_used <= budget

    def test_graph_helpers(self, mixed_index):
        r = HybridRetriever(mixed_index)
        path = r.shortest_path("app.geometry.length", "geo::Vec::dot")
        assert path == ["app.geometry.length", "geo::Vec::norm", "geo::Vec::dot"]
        hood = r.neighborhood("app.models.Order.total", 1, "in")
        assert "app.service.OrderService.checkout" in hood
        assert r.top_pagerank(3)

    def test_graphrag_beats_naive_chunking_on_dependency_recall(self, mixed_index):
        r = HybridRetriever(mixed_index)
        naive = NaiveChunkRetriever(mixed_index, chunk_lines=6)
        sid = "app.service.OrderService.checkout"
        truth = {sid, *mixed_index.callees(sid)}
        q = mixed_index.symbols[sid].docstring
        # app/reports.py reads like the query but is not a dependency: text similarity alone is fooled
        graph_hits = set(r.retrieve(q, token_budget=250).ids()) & truth
        chunks, _ = naive.retrieve(q, token_budget=250)
        naive_hits = naive.covered_symbols(chunks) & truth
        assert graph_hits == truth
        assert len(naive_hits) < len(truth)


# ---------------------------------------------------------------------------
# 4. Sandbox + self-healing loop
# ---------------------------------------------------------------------------


@pytest.fixture()
def demo_repo(tmp_path) -> Path:
    return create_demo_project(tmp_path / "demo")


class TestHealer:
    def test_sandbox_isolation_and_patch(self, demo_repo):
        original = (demo_repo / "inventory/pricing.py").read_text()
        out = try_patch_in_sandbox(demo_repo, [{
            "file": "inventory/pricing.py", "search": "if quantity > 50:", "replace": "if quantity >= 50:",
        }], "tests")
        assert out["test"]["passed"]
        assert "+    if quantity >= 50:" in out["diff"]
        assert (demo_repo / "inventory/pricing.py").read_text() == original  # repo untouched

    def test_baseline_reports_failures(self, demo_repo):
        run = run_tests_in_sandbox(demo_repo, "tests")
        assert not run.passed
        assert run.counts.get("failed") == 2 and run.counts.get("passed") == 5
        ids = {f.test_id for f in run.failures}
        assert "tests/test_cart.py::test_fifty_units_get_ten_percent" in ids
        assert all(f.error_type == "AssertionError" for f in run.failures)

    def test_sandbox_rejects_path_escape(self, demo_repo):
        with Sandbox(demo_repo) as sb, pytest.raises(ValueError):
            sb.read("../../etc/passwd")

    def test_heals_planted_bug_and_applies_fix(self, demo_repo):
        events = []
        healer = SelfHealer(demo_repo, "tests", workers=3, on_event=events.append)
        result = healer.heal(apply=True)
        assert result.success and result.applied
        assert result.patch["source"] == "mutation"
        assert result.patch["target"] == "inventory.pricing.bulk_discount_percent"
        assert "+    if quantity >= 50:" in result.diff
        states = [e["state"] for e in events]
        for s in ("REPRODUCE", "LOCALIZE", "PROPOSE", "APPLY", "VERIFY", "HEALED"):
            assert s in states
        assert states.index("REPRODUCE") < states.index("LOCALIZE") < states.index("VERIFY") < states.index("HEALED")
        assert result.suspects[0]["symbol"] == "inventory.pricing.bulk_discount_percent"
        assert run_tests_in_sandbox(demo_repo, "tests").passed  # the working tree is fixed

    def test_regressions_are_rolled_back(self, demo_repo):
        healer = SelfHealer(demo_repo, "tests", workers=1)
        result = healer.heal()
        rollbacks = [e for e in result.trace if e["state"] == "ROLLBACK"]
        # with one worker the search walks candidates in order; `>= 100` -> `> 100` is tried
        # first, fixes nothing, and must be rolled back before the real fix verifies
        assert result.success and rollbacks
        assert "inventory/pricing.py" in result.diff

    def test_green_suite_needs_no_healing(self, demo_repo):
        p = demo_repo / "inventory/pricing.py"
        p.write_text(p.read_text().replace("quantity > 50", "quantity >= 50"))
        result = SelfHealer(demo_repo, "tests").heal()
        assert result.success and result.attempts == 0 and result.patch is None

    def test_unfixable_bug_fails_within_budget(self, tmp_path):
        repo = write_tree(tmp_path, {
            "lib/core.py": "def answer():\n    raise NotImplementedError('todo')\n",
            "lib/__init__.py": "",
            "tests/test_core.py": "from lib.core import answer\n\ndef test_answer():\n    assert answer() == 42\n",
        })
        result = SelfHealer(repo, "tests", max_attempts=6, use_llm=False).heal()
        assert not result.success and result.state == HealState.FAILED.value
        assert result.attempts <= 6

    def test_timeouts_kill_runaway_tests(self, tmp_path):
        repo = write_tree(tmp_path, {
            "tests/test_spin.py": "def test_forever():\n    while True:\n        pass\n",
        })
        with Sandbox(repo, timeout=3) as sb:
            t0 = time.time()
            run = sb.run_tests("tests")
        assert run.timed_out and not run.passed
        assert time.time() - t0 < 15


# ---------------------------------------------------------------------------
# 5. API server
# ---------------------------------------------------------------------------


class TestAPI:
    @pytest.fixture()
    def client(self, mixed_repo):
        from fastapi.testclient import TestClient

        from chronos.api_server import create_app

        app = create_app(mixed_repo, telemetry_interval=0.2)
        with TestClient(app) as c:
            yield c

    def test_health_index_graph(self, client, mixed_repo):
        h = client.get("/api/health").json()
        assert h["ok"] and h["cplusplus"] >= 202002
        stats = client.post("/api/index", json={}).json()
        assert stats["symbols"] > 10 and stats["engine"]["nodes"] == stats["symbols"]
        g = client.get("/api/graph").json()
        ids = {n["id"] for n in g["nodes"]}
        assert "app.service.OrderService.checkout" in ids
        assert any(l["kind"] == "call" for l in g["links"])
        assert all(l["source"] in ids and l["target"] in ids for l in g["links"])
        assert client.post("/api/index", json={"path": "/definitely/not/here"}).status_code == 404

    def test_retrieve_symbol_path(self, client):
        r = client.post("/api/retrieve", json={"query": "charge the order total with tax", "token_budget": 1200}).json()
        assert r["tokens_used"] <= 1200 and r["nodes"] and "context" in r
        s = client.get("/api/symbol", params={"id": "compute_tax"}).json()
        assert s["id"] == "app.pricing.compute_tax" and "app.service.OrderService.checkout" in s["callers"]
        assert client.get("/api/symbol", params={"id": "no_such_symbol_xyz"}).status_code == 404
        p = client.get("/api/path", params={"src": "app.geometry.length", "dst": "geo::Vec::dot"}).json()
        assert p["reachable"] and p["path"][-1] == "geo::Vec::dot"

    def test_websocket_streams_events(self, client):
        client.post("/api/index", json={})
        with client.websocket_connect("/ws") as ws:
            hello = ws.receive_json()
            assert hello["type"] == "hello" and hello["indexed"]
            client.post("/api/retrieve", json={"query": "vector norm"})
            seen = set()
            deadline = time.time() + 10
            while time.time() < deadline and not {"retrieval", "telemetry"} <= seen:
                ev = ws.receive_json()
                seen.add(ev["type"])
                if ev["type"] == "telemetry":
                    assert ev["nodes"] > 0
            assert {"retrieval", "telemetry"} <= seen
            ws.send_text("ping")

    def test_heal_demo_over_http(self, client):
        run = client.post("/api/heal", json={"demo": True, "apply": True}).json()
        deadline = time.time() + 120
        status = {}
        while time.time() < deadline:
            status = client.get(f"/api/heal/{run['run_id']}").json()
            if status["status"] != "running":
                break
            time.sleep(0.2)
        assert status["status"] == "done", status
        assert status["result"]["success"]
        assert client.get("/api/health").json()["root"].endswith("demo-inventory")

    def test_benchmark_endpoint(self, client):
        b = client.post("/api/benchmark", json={"nodes": 2000, "queries": 10}).json()
        assert all(r["matches"] for r in b["results"])


# ---------------------------------------------------------------------------
# 6. MCP server
# ---------------------------------------------------------------------------


def test_mcp_server_tools(mixed_repo):
    from chronos.mcp_server import build_server

    try:
        from mcp import Client
    except ImportError:  # MCP SDK 1.x: the in-process client lives elsewhere
        pytest.skip("in-process MCP client needs the MCP Python SDK 2.x")

    def payload(result):
        assert not result.is_error, result.content
        return result.content[0].text

    async def scenario():
        async with Client(build_server(str(mixed_repo))) as c:
            names = {t.name for t in (await c.list_tools()).tools}
            assert {"retrieve_context", "get_symbol", "call_path", "self_heal", "try_patch", "run_tests"} <= names
            ctx = payload(await c.call_tool("retrieve_context", {"query": "sales tax on checkout", "token_budget": 1500}))
            assert "compute_tax" in ctx and "GraphRAG context" in ctx
            sym = json.loads(payload(await c.call_tool("get_symbol", {"name": "Vec::norm"})))
            assert sym["id"] == "geo::Vec::norm" and "geo::Vec::dot" in sym["callees"]
            path = json.loads(payload(await c.call_tool("call_path", {"source": "length", "target": "Vec::dot"})))
            assert path["reachable"]
            hits = json.loads(payload(await c.call_tool("search_symbols", {"query": "coupon discount", "top_k": 3})))
            assert hits["results"][0]["id"] == "app.pricing.apply_coupon"
            central = json.loads(payload(await c.call_tool("central_symbols", {"limit": 5})))
            assert len(central["results"]) == 5
            res = await c.read_resource("chronos://stats")
            assert json.loads(res.contents[0].text)["symbols"] > 10
            prompts = {p.name for p in (await c.list_prompts()).prompts}
            assert "debug_failing_test" in prompts

    asyncio.run(scenario())


def test_mcp_self_heal_tool(tmp_path):
    from chronos.mcp_server import build_server

    try:
        from mcp import Client
    except ImportError:
        pytest.skip("in-process MCP client needs the MCP Python SDK 2.x")
    repo = create_demo_project(tmp_path / "demo")

    async def scenario():
        async with Client(build_server(str(repo))) as c:
            out = json.loads((await c.call_tool("self_heal", {"test_target": "tests", "apply": False})).content[0].text)
            assert out["success"] and not out["applied"] and ">=" in out["diff"]

    asyncio.run(scenario())


# ---------------------------------------------------------------------------
# 7. Benchmark smoke test
# ---------------------------------------------------------------------------


def test_benchmarks_agree_with_networkx(mixed_repo):
    from chronos.benchmark import bench_graph, bench_retrieval

    g = bench_graph(3000, 4, 3, 30)
    assert all(r["matches"] for r in g["results"]), g["results"]
    assert g["pagerank_max_abs_diff"] < 1e-4
    r = bench_retrieval(mixed_repo, token_budget=600)
    assert r["queries"] >= 1
    assert r["graphrag"]["dependency_recall"] >= r["naive_chunking"]["dependency_recall"]


# ---------------------------------------------------------------------------
# 8. Claude patch proposer (simulated API) and the docker sandbox
# ---------------------------------------------------------------------------

from types import SimpleNamespace  # noqa: E402

from chronos.sandbox_healer import PATCH_SCHEMA, ClaudeProposer, MutationProposer  # noqa: E402


class FakeAnthropic:
    """Stands in for anthropic.Anthropic: records requests, replays canned responses."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.requests.append({**kwargs, "messages": list(kwargs["messages"])})
        return self.responses.pop(0)


def fake_response(edits=None, analysis="", stop_reason="end_turn"):
    body = json.dumps({"analysis": analysis, "edits": edits or []})
    return SimpleNamespace(stop_reason=stop_reason, content=[SimpleNamespace(type="text", text=body)])


def install_fake(monkeypatch, fake):
    anthropic = pytest.importorskip("anthropic")
    monkeypatch.setattr(anthropic, "Anthropic", lambda *a, **k: fake)


def test_claude_proposer_heals_what_mutation_cannot(tmp_path, monkeypatch):
    repo = create_demo_project(tmp_path / "stats", "missing-sort")
    wrong = [{"file": "stats/summary.py", "search": "        return values[mid]\n",
              "replace": "        return values[-mid]\n"}]
    right = [{"file": "stats/summary.py", "search": "    n = len(values)\n",
              "replace": "    values = sorted(values)\n    n = len(values)\n"}]
    fake = FakeAnthropic([fake_response(wrong, "index from the end"), fake_response(right, "median never sorts")])
    install_fake(monkeypatch, fake)

    # a tight budget: the 20-candidate mutation search must leave room for the model
    healer = SelfHealer(repo, "tests", proposers=[MutationProposer(), ClaudeProposer()], max_attempts=8)
    result = healer.heal(apply=True)

    assert result.success and result.applied
    assert result.patch["source"] == "claude"
    assert "+    values = sorted(values)" in result.diff
    assert run_tests_in_sandbox(repo, "tests").passed

    first, second = fake.requests
    assert first["model"] == "claude-opus-5-5"
    assert first["betas"] == ["server-side-fallback-2026-07-01"] and first["fallbacks"] == "default"
    assert first["output_config"]["format"] == {"type": "json_schema", "schema": PATCH_SCHEMA}
    assert "Never edit test files" in first["system"]
    prompt = first["messages"][0]["content"]
    assert "test_median_of_unsorted_even" in prompt and "GraphRAG context" in prompt and "def median" in prompt
    # the failed first attempt goes back to the model, append-only
    assert [m["role"] for m in second["messages"]] == ["user", "assistant", "user"]
    assert json.loads(second["messages"][1]["content"][0].text)["edits"] == wrong
    assert "did not fix" in second["messages"][2]["content"]


def test_claude_proposer_handles_refusals_and_test_file_edits(tmp_path, monkeypatch):
    repo = create_demo_project(tmp_path / "stats", "missing-sort")
    sneaky = [{"file": "tests/test_summary.py", "search": "== 2.5", "replace": "== 2.0"}]
    fake = FakeAnthropic([fake_response(sneaky, "edit the test instead")])
    install_fake(monkeypatch, fake)
    result = SelfHealer(repo, "tests", proposers=[ClaudeProposer()]).heal()
    assert not result.success and result.attempts == 0  # test-file edits are dropped, never applied

    fake = FakeAnthropic([fake_response(stop_reason="refusal")])
    install_fake(monkeypatch, fake)
    result = SelfHealer(repo, "tests", proposers=[ClaudeProposer()]).heal()
    assert not result.success and result.state == HealState.FAILED.value
    assert any("declined" in e["message"] for e in result.trace)

    class Broken(FakeAnthropic):
        def _create(self, **kwargs):
            raise ConnectionError("network unreachable")

    install_fake(monkeypatch, Broken([]))
    result = SelfHealer(repo, "tests", proposers=[ClaudeProposer()]).heal()  # must not raise
    assert not result.success
    assert any("claude stopped: ConnectionError" in e["message"] for e in result.trace)


@pytest.mark.skipif(
    not (os.environ.get("ANTHROPIC_API_KEY") and os.environ.get("CHRONOS_LIVE_LLM") == "1"),
    reason="live API call: set ANTHROPIC_API_KEY and CHRONOS_LIVE_LLM=1 to run (costs a few cents)",
)
def test_claude_proposer_live(tmp_path):
    repo = create_demo_project(tmp_path / "stats", "missing-sort")
    result = SelfHealer(repo, "tests", proposers=[ClaudeProposer()]).heal(apply=True)
    assert result.success and result.patch["source"] == "claude"
    assert run_tests_in_sandbox(repo, "tests").passed


def _docker_ready() -> bool:
    import shutil
    import subprocess

    if shutil.which("docker") is None:
        return False
    return subprocess.run(["docker", "info"], capture_output=True).returncode == 0


@pytest.mark.skipif(not _docker_ready(), reason="docker daemon not running")
def test_docker_backend_isolates_and_heals(tmp_path):
    repo = create_demo_project(tmp_path / "demo")
    write_tree(tmp_path / "net", {"tests/test_net.py": """
        import socket
        import pytest


        def test_no_network():
            with pytest.raises(OSError):
                socket.create_connection(("1.1.1.1", 53), timeout=3)
    """})
    with Sandbox(tmp_path / "net", backend="docker") as sb:
        assert sb.run_tests("tests").passed  # the container has no network
    result = SelfHealer(repo, "tests", backend="docker").heal()
    assert result.success and "+    if quantity >= 50:" in result.diff
