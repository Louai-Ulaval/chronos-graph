"""Hybrid GraphRAG retriever.

Stage 1  vector recall: a feature-hashed TF-IDF embedding over identifier sub-tokens,
         docstrings and signatures; cosine similarity picks the seed symbols.
Stage 2  graph expansion: the C++ engine propagates seed relevance across k hops of the
         call / import / containment graph and blends it with PageRank centrality.
Stage 3  token-budget pruning: a greedy knapsack in C++ keeps the highest-value
         subgraph that fits the caller's context budget.

The embedder is pluggable: anything with ``fit_transform(texts)`` and ``transform(texts)``
returning L2-normalised rows works (a sentence-transformer wrapper, for example).
"""

from __future__ import annotations

import math
import re
import threading
import time
import zlib
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np

import chronos_cpp

from .ast_indexer import CodeIndex, Symbol, estimate_tokens

_STOPWORDS = frozenset(
    """a an the to of and or in on for is are be by with from as at it this that these those not no
    self cls def return if else elif while for try except finally raise import none true false pass
    class lambda yield async await std const auto void int bool double float char unsigned size_t
    uint32_t uint64_t int32_t static inline struct namespace public private protected using
    typename template nullptr do does how what which where when who why use used uses into its we
    you code function method""".split()
)
_SUBTOKEN = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|\d+")
_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def _stem(tok: str) -> str:
    if len(tok) > 4 and tok.endswith("ies"):
        tok = tok[:-3] + "y"
    elif len(tok) > 4 and tok.endswith(("sses", "xes", "ches", "shes")):
        tok = tok[:-2]
    elif len(tok) > 3 and tok.endswith("s") and not tok.endswith("ss"):
        tok = tok[:-1]
    for suf in ("ing", "ed", "er"):
        if len(tok) > len(suf) + 3 and tok.endswith(suf):
            tok = tok[: -len(suf)]
            break
    if len(tok) > 4 and tok.endswith("e"):
        tok = tok[:-1]
    return tok


def tokenize(text: str) -> list[str]:
    """Split identifiers on snake_case / camelCase boundaries, lowercase and stem."""
    out = []
    for ident in _IDENT.findall(text):
        for part in _SUBTOKEN.findall(ident):
            low = part.lower()
            if len(low) < 2 or low in _STOPWORDS:
                continue
            out.append(_stem(low))
    return out


class Embedder(Protocol):
    def fit_transform(self, texts: list[str]) -> np.ndarray: ...
    def transform(self, texts: list[str]) -> np.ndarray: ...


class HashingEmbedder:
    """Signed feature hashing of unigrams + bigrams with smoothed IDF and L2 normalisation."""

    def __init__(self, dim: int = 4096, bigrams: bool = True):
        self.dim = dim
        self.bigrams = bigrams
        self.idf = np.ones(dim, dtype=np.float32)

    def _features(self, text: str) -> dict[int, float]:
        toks = tokenize(text)
        feats: dict[int, float] = defaultdict(float)
        grams = toks + ([f"{a}_{b}" for a, b in zip(toks, toks[1:])] if self.bigrams else [])
        for g in grams:
            h = zlib.crc32(g.encode())
            sign = 1.0 if (h >> 31) & 1 else -1.0
            feats[h % self.dim] += sign
        return feats

    def _matrix(self, texts: list[str]) -> tuple[np.ndarray, np.ndarray]:
        mat = np.zeros((len(texts), self.dim), dtype=np.float32)
        df = np.zeros(self.dim, dtype=np.float32)
        for i, t in enumerate(texts):
            for j, v in self._features(t).items():
                # sublinear tf keeps long bodies from drowning out names
                mat[i, j] = math.copysign(1.0 + math.log(abs(v)), v) if v else 0.0
                df[j] += 1.0
        return mat, df

    @staticmethod
    def _normalise(mat: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return mat / norms

    def fit_transform(self, texts: list[str]) -> np.ndarray:
        mat, df = self._matrix(texts)
        n = max(len(texts), 1)
        self.idf = (np.log((1.0 + n) / (1.0 + df)) + 1.0).astype(np.float32)
        return self._normalise(mat * self.idf)

    def transform(self, texts: list[str]) -> np.ndarray:
        mat, _ = self._matrix(texts)
        return self._normalise(mat * self.idf)


def symbol_head(sym: Symbol) -> str:
    """Name, path, signature and docstring: what a symbol *is*."""
    qual = sym.id.replace("::", " ").replace(".", " ").replace("/", " ")
    return " ".join([sym.name] * 2 + [qual, sym.signature, sym.docstring, Path(sym.file).stem])


def symbol_body(sym: Symbol) -> str:
    """Identifiers used in the body: what a symbol *does*."""
    return " ".join(_IDENT.findall(sym.source[:8000]))


def symbol_document(sym: Symbol) -> str:
    return symbol_head(sym) + " " + symbol_body(sym)


@dataclass
class RetrievedNode:
    id: str
    name: str
    kind: str
    file: str
    line: int
    end_line: int
    score: float
    relevance: float
    centrality: float
    distance: int
    seed: bool
    token_cost: int
    vector_score: float = 0.0

    def to_dict(self) -> dict:
        return dict(self.__dict__)


@dataclass
class RetrievalResult:
    query: str
    nodes: list[RetrievedNode]
    edges: list[dict]
    tokens_used: int
    token_budget: int
    candidates: int
    timings: dict[str, float]
    seeds: list[tuple[str, float]] = field(default_factory=list)
    context: str = ""

    def ids(self) -> list[str]:
        return [n.id for n in self.nodes]

    def to_dict(self, with_context: bool = True) -> dict:
        d = {
            "query": self.query,
            "nodes": [n.to_dict() for n in self.nodes],
            "edges": self.edges,
            "tokens_used": self.tokens_used,
            "token_budget": self.token_budget,
            "candidates": self.candidates,
            "timings": self.timings,
            "seeds": [{"id": s, "score": round(v, 4)} for s, v in self.seeds],
        }
        if with_context:
            d["context"] = self.context
        return d


_DIRECTIONS = {
    "out": chronos_cpp.Direction.OUT,
    "in": chronos_cpp.Direction.IN,
    "both": chronos_cpp.Direction.BOTH,
}


class HybridRetriever:
    def __init__(self, index: CodeIndex, embedder: Embedder | None = None, head_weight: float = 0.7):
        self.index = index
        self.engine = index.engine
        self.ids = index.ids
        self.embedder = embedder or HashingEmbedder()
        self.body_embedder = HashingEmbedder(bigrams=False) if embedder is None else self.embedder
        self.head_weight = head_weight
        t0 = time.perf_counter()
        syms = [index.symbols[sid] for sid in self.ids]
        if syms:
            # Two fields, embedded separately so long bodies cannot drown out names and docs.
            self.head = self.embedder.fit_transform([symbol_head(s) for s in syms])
            self.body = self.body_embedder.fit_transform([symbol_body(s) for s in syms])
        else:
            self.head = self.body = np.zeros((0, 1), dtype=np.float32)
        self.embed_ms = (time.perf_counter() - t0) * 1000
        self._name_index: dict[str, list[int]] = defaultdict(list)
        for i, sid in enumerate(self.ids):
            self._name_index[index.symbols[sid].name.lower()].append(i)
        self._lock = threading.Lock()
        self.query_log: list[dict] = []

    # ---- stage 1 ------------------------------------------------------------
    def vector_search(self, query: str, top_k: int = 10, kinds: set[str] | None = None) -> list[tuple[str, float]]:
        if not self.ids:
            return []
        qh = self.embedder.transform([query])[0]
        qb = self.body_embedder.transform([query])[0]
        scores = self.head_weight * (self.head @ qh) + (1.0 - self.head_weight) * (self.body @ qb)
        # Lexical boost: identifiers mentioned verbatim in the query.
        for raw in set(re.findall(r"[A-Za-z_][\w]*(?:(?:::|\.)[A-Za-z_]\w*)*", query)):
            leaf = re.split(r"::|\.", raw)[-1]
            looks_like_code = "_" in raw or "::" in raw or "." in raw or re.search(r"[a-z][A-Z]|\d", raw)
            boost = 0.35 if looks_like_code else 0.03
            for i in self._name_index.get(leaf.lower(), []):
                scores[i] += boost
        if kinds:
            mask = np.array([self.index.symbols[sid].kind in kinds for sid in self.ids])
            scores = np.where(mask, scores, -np.inf)
        k = min(top_k, len(self.ids))
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        return [(self.ids[i], float(scores[i])) for i in top if np.isfinite(scores[i]) and scores[i] > 0]

    # ---- stages 2 + 3 --------------------------------------------------------
    def retrieve(
        self,
        query: str,
        token_budget: int = 4000,
        hops: int = 2,
        seeds: int = 6,
        centrality_weight: float = 0.25,
        hop_decay: float = 0.6,
        direction: str = "both",
        max_nodes: int = 0,
        render: bool = True,
    ) -> RetrievalResult:
        t0 = time.perf_counter()
        hits = self.vector_search(query, top_k=max(seeds * 3, seeds), kinds={"function", "method", "class"})
        if not hits:
            hits = self.vector_search(query, top_k=seeds)
        if hits:
            best = hits[0][1]
            hits = [h for h in hits if h[1] >= 0.35 * best][:seeds]
        t1 = time.perf_counter()

        opts = chronos_cpp.PruneOptions()
        opts.max_hops = max(0, int(hops))
        # leave room for the per-symbol headers the renderer adds
        opts.token_budget = max(1, int(token_budget * 0.92))
        opts.hop_decay = hop_decay
        opts.centrality_weight = centrality_weight
        opts.direction = _DIRECTIONS.get(direction, chronos_cpp.Direction.BOTH)
        opts.max_nodes = max_nodes
        seed_idx = [self.index.id_to_idx[sid] for sid, _ in hits]
        seed_scores = [score for _, score in hits]
        pr = self.engine.prune(seed_idx, seed_scores, opts)
        t2 = time.perf_counter()

        vec_scores = dict(hits)
        nodes = []
        for pn in pr.nodes:
            sid = self.ids[pn.node]
            s = self.index.symbols[sid]
            nodes.append(
                RetrievedNode(
                    id=sid, name=s.name, kind=s.kind, file=s.file, line=s.line, end_line=s.end_line,
                    score=round(pn.score, 5), relevance=round(pn.relevance, 5), centrality=round(pn.centrality, 5),
                    distance=pn.distance, seed=pn.seed, token_cost=pn.token_cost,
                    vector_score=round(vec_scores.get(sid, 0.0), 5),
                )
            )
        kind_names = {v: k for k, v in chronos_cpp.EdgeKind.__members__.items()}
        edges = [
            {"src": self.ids[e.src], "dst": self.ids[e.dst], "kind": kind_names[e.kind].lower()} for e in pr.edges
        ]
        result = RetrievalResult(
            query=query, nodes=nodes, edges=edges, tokens_used=int(pr.tokens_used), token_budget=token_budget,
            candidates=int(pr.candidates), seeds=hits,
            timings={
                "vector_ms": round((t1 - t0) * 1000, 3),
                "graph_ms": round((t2 - t1) * 1000, 3),
                "total_ms": 0.0,
            },
        )
        if render:
            result.context = self.render(result)
        result.timings["total_ms"] = round((time.perf_counter() - t0) * 1000, 3)
        with self._lock:
            self.query_log.append(
                {"query": query, "ms": result.timings["total_ms"], "nodes": len(nodes), "tokens": result.tokens_used, "ts": time.time()}
            )
            del self.query_log[:-500]
        return result

    def render(self, result: RetrievalResult) -> str:
        """Render the pruned subgraph as an LLM-ready context block, grouped by file."""
        by_file: dict[str, list[RetrievedNode]] = defaultdict(list)
        for n in result.nodes:
            by_file[n.file].append(n)
        out = [f"# GraphRAG context for: {result.query}", ""]
        if result.edges:
            calls = [f"{e['src']} -> {e['dst']}" for e in result.edges if e["kind"] == "call"]
            if calls:
                out.append("## Call edges")
                out.extend(f"- {c}" for c in calls[:40])
                out.append("")
        for file, nodes in sorted(by_file.items(), key=lambda kv: -max(n.score for n in kv[1])):
            out.append(f"## {file}")
            for n in sorted(nodes, key=lambda n: n.line):
                sym = self.index.symbols[n.id]
                lang = "python" if sym.language == "python" else "cpp"
                tag = "seed" if n.seed else f"hop {n.distance}"
                out.append(f"### {n.id}  ({n.kind}, line {n.line}, {tag}, score {n.score:.3f})")
                out.append(f"```{lang}\n{sym.source.rstrip()}\n```")
            out.append("")
        return "\n".join(out)

    # ---- helpers ---------------------------------------------------------------
    def neighborhood(self, sid: str, k: int = 1, direction: str = "both") -> dict[str, int]:
        i = self.index.id_to_idx[sid]
        hops = self.engine.k_hop([i], k, _DIRECTIONS.get(direction, chronos_cpp.Direction.BOTH))
        return {self.ids[n]: d for n, d in hops.items()}

    def shortest_path(self, src: str, dst: str, direction: str = "out") -> list[str]:
        a, b = self.index.id_to_idx[src], self.index.id_to_idx[dst]
        return [self.ids[n] for n in self.engine.shortest_path(a, b, _DIRECTIONS.get(direction, chronos_cpp.Direction.OUT))]

    def top_pagerank(self, n: int = 20, kinds: set[str] | None = None) -> list[tuple[str, float]]:
        pr = self.engine.pagerank()
        order = np.argsort(-pr)
        out = []
        for i in order:
            sid = self.ids[i]
            if kinds and self.index.symbols[sid].kind not in kinds:
                continue
            out.append((sid, float(pr[i])))
            if len(out) >= n:
                break
        return out


# ---------------------------------------------------------------------------
# Baseline: naive fixed-size chunking (what most RAG pipelines do)
# ---------------------------------------------------------------------------


@dataclass
class Chunk:
    file: str
    start: int  # 1-based inclusive
    end: int
    text: str
    tokens: int


class NaiveChunkRetriever:
    def __init__(self, index: CodeIndex, chunk_lines: int = 40, embedder: Embedder | None = None):
        self.index = index
        self.chunk_lines = chunk_lines
        self.embedder = embedder or HashingEmbedder()
        self.chunks: list[Chunk] = []
        root = Path(index.root)
        base = root if root.is_dir() else root.parent
        for rel in index.files:
            try:
                lines = (base / rel).read_text(encoding="utf-8", errors="replace").splitlines()
            except OSError:
                continue
            for start in range(0, len(lines), chunk_lines):
                body = "\n".join(lines[start:start + chunk_lines])
                if body.strip():
                    self.chunks.append(Chunk(rel, start + 1, min(start + chunk_lines, len(lines)), body, estimate_tokens(body)))
        self.matrix = (
            self.embedder.fit_transform([f"{Path(c.file).stem} {c.text}" for c in self.chunks])
            if self.chunks else np.zeros((0, 1), dtype=np.float32)
        )

    def retrieve(self, query: str, token_budget: int = 4000) -> tuple[list[Chunk], int]:
        if not self.chunks:
            return [], 0
        q = self.embedder.transform([query])[0]
        order = np.argsort(-(self.matrix @ q))
        picked, used = [], 0
        for i in order:
            c = self.chunks[i]
            if used + c.tokens > token_budget:
                continue
            picked.append(c)
            used += c.tokens
        return picked, used

    def covered_symbols(self, chunks: list[Chunk], min_overlap: float = 0.5) -> set[str]:
        """Symbols whose body is at least ``min_overlap`` covered by the retrieved chunks."""
        spans: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for c in chunks:
            spans[c.file].append((c.start, c.end))
        covered = set()
        for sid, s in self.index.symbols.items():
            if s.kind in ("module", "file") or s.file not in spans:
                continue
            length = max(1, s.end_line - s.line + 1)
            overlap = sum(max(0, min(e, s.end_line) - max(b, s.line) + 1) for b, e in spans[s.file])
            if overlap / length >= min_overlap:
                covered.add(sid)
        return covered
