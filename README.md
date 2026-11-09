# Chronos-Graph

A code-intelligence stack in five layers:

1. **C++20 graph engine** (`src/cpp`): a contiguous-memory call / dependency graph with k-hop BFS, bidirectional shortest path, PageRank and a token-budget subgraph pruner, exposed to Python as `chronos_cpp` through pybind11.
2. **Multi-language indexer** (`ast_indexer.py`): Python via `ast`, C/C++ via a brace-aware scanner, and **pybind11 bindings parsed as a bridge**, so a Python call like `engine.prune(...)` lands on `chronos::GraphEngine::prune`.
3. **Hybrid GraphRAG retriever** (`hybrid_retriever.py`): vector seeds, then propagation along the call graph, then PageRank blending, then a greedy knapsack that fits the result to a token budget.
4. **MCP server + self-healing agent** (`mcp_server.py`, `sandbox_healer.py`): graph tools for any MCP client, plus a repair loop that reproduces failing tests, localises the bug on the call graph, screens patches in parallel sandboxes and verifies them against the full suite.
5. **Telemetry UI** (`api_server.py`, `web/index.html`): FastAPI + WebSocket streaming into a dark force-graph view (2D and 3D) with live retrieval highlights and agent traces.

```
            +--------------------- web/index.html (force-graph 2D / 3D) ---------------------+
            |                                 WebSocket /ws                                  |
            +----------------------------------------^---------------------------------------+
                                                     |
  MCP client (stdio / SSE / HTTP) --> mcp_server.py --+--> workspace.py <-- api_server.py (REST)
                                                     |
             +---------------------------------------+------------------------------+
             |                                       |                              |
      ast_indexer.py  -------------------->  hybrid_retriever.py          sandbox_healer.py
      Python ast + C++ scanner               vector seeds + prune()        REPRODUCE > LOCALIZE >
      + pybind11 bridge                              |                     PROPOSE > APPLY > VERIFY
             |                                       |                              |
             +------------------->  chronos_cpp (C++20, CSR)  <---------------------+
                                    k-hop BFS · bidirectional BFS · PageRank · prune
```

## Quick start

```bash
make install
```

```bash
make test
```

```bash
make ui
```

Then open http://127.0.0.1:8000. `make install` creates `.venv`, installs `requirements.txt` and builds the extension through an editable install. Run `make build` alone to recompile the C++ after you edit it.

| command | what it does |
|---|---|
| `make build` | compile `chronos_cpp` in place (`-O3 -std=c++20`; `CHRONOS_NATIVE=1` adds `-mcpu=native` on arm64) |
| `make test` | build, then run the end-to-end suite (47 tests; the live-API one is skipped unless you opt in) |
| `make benchmark` | C++20 vs NetworkX at `NODES=100000`, plus GraphRAG vs naive chunking on `REPO=.` |
| `make ui` | API + UI on `PORT=8000`, indexing `REPO=.` |
| `make mcp` | MCP server over stdio (`MCP_TRANSPORT=sse` or `streamable-http` with `MCP_PORT=8765`) |
| `make demo` | self-heal the bundled demo project from the terminal |
| `make demo-llm` | heal a bug that only the Claude proposer can fix (needs `ANTHROPIC_API_KEY`; a few cents per run) |

Requires Python 3.10+ and a C++20 compiler. It was built and tested with Apple clang 17 on arm64. The flags avoid x86-only options, so the same `setup.py` also targets x86_64, Linux GCC and MSVC (`/std:c++20`). Some Python builds (pyenv) carry an `-isysroot` that no longer exists; `setup.py` swaps in the active SDK from `xcrun` automatically.

## Connect it to an MCP client

Claude Code:

```bash
claude mcp add chronos-graph -- /path/to/chronos-graph/.venv/bin/python -m chronos.mcp_server --repo /path/to/your/repo
```

Any client that takes a JSON config:

```json
{
  "mcpServers": {
    "chronos-graph": {
      "command": "/path/to/chronos-graph/.venv/bin/python",
      "args": ["-m", "chronos.mcp_server", "--repo", "/path/to/your/repo"]
    }
  }
}
```

| tool | purpose |
|---|---|
| `retrieve_context(query, token_budget, hops, centrality_weight)` | GraphRAG context as markdown: source grouped by file plus the call edges between the chosen symbols |
| `search_symbols(query, top_k)` | vector search over names, signatures, docstrings and bodies |
| `get_symbol(name)` | source, callers, callees and PageRank of a symbol (exact id, qualified suffix or short name) |
| `graph_neighbors(name, hops, direction)` | k-hop neighbourhood (`out` = callees, `in` = callers) |
| `call_path(source, target)` | shortest call chain, across the Python and C++ boundary |
| `central_symbols(limit)` | the most central functions and classes |
| `run_tests(target)` | pytest in a sandbox copy |
| `try_patch(edits, target)` | apply search/replace edits in a sandbox and run the tests; the working tree is untouched |
| `self_heal(test_target, apply, max_attempts)` | the full repair loop; writes to disk only with `apply=true` |
| `index_repository(path)` | re-index; unchanged files come from a content-hash cache |

Resources: `chronos://stats`, `chronos://symbol/{name}`. Prompt: `debug_failing_test`. Works with the MCP Python SDK 2.x (`MCPServer`) and 1.x (`FastMCP`).

## Benchmarks

Measured with `make benchmark` on Apple Silicon (arm64), Python 3.12, NetworkX 3.7 with scipy. The graph is synthetic and scale-free: 100,000 nodes and 499,322 edges, where each node calls about 5 earlier nodes, biased toward hubs. Every C++ result is checked against NetworkX's answer.

| operation | C++20 | NetworkX | speedup | same answer |
|---|---:|---:|---:|:---:|
| build graph | 15.3 ms | 854 ms | 56x | yes |
| 3-hop BFS ×200, results as numpy arrays | 37.9 ms | 936 ms | 25x | yes |
| 3-hop BFS ×200, results as a Python dict | 201 ms | 936 ms | 4.7x | yes |
| shortest path ×200 (bidirectional BFS) | 1.0 ms | 26 ms | 25x | yes |
| PageRank (same L1 stopping rule) | 3.8 ms | 614 ms | 161x | max diff 1.4e-17 |

Memory: 29.4 MB in total, or 62 bytes per edge, including both CSR directions and all scratch buffers.

How to read these numbers:

* The dict row is dominated by building about 9,000 Python objects per query. Use `k_hop_arrays` when you need speed.
* NetworkX's PageRank time includes converting the graph to a scipy sparse matrix, which is part of what a caller pays.
* The NetworkX BFS runs on an undirected copy that was built before timing started.

**GraphRAG vs naive 40-line chunking**, on this repository at a 2,000-token budget. Each query is a function's docstring. The ground truth is that function plus its direct callees, which is what an agent needs in context to change the function safely.

| metric | GraphRAG | naive chunks |
|---|---:|---:|
| dependency recall | 0.87 | 0.48 |
| target hit rate | 0.91 | 0.64 |
| share of tokens that are relevant | 0.23 | 0.10 |
| tokens used | 1,652 | 1,982 |

That is 11 queries, because only documented functions with resolved callees qualify, so treat it as indicative. Run it on a larger codebase with `make benchmark REPO=/path/to/repo`. The test suite also checks a harder case, a file that reads like the query but is not a dependency: GraphRAG returns the full dependency set at 250 tokens, while naive chunking gets 1 of 4.

## How it works

**Engine.** Nodes sit in a dense vector. Edges are staged, deduplicated per (src, dst) pair, and compacted into out- and in-CSR arrays on the first query, so every traversal walks contiguous memory. BFS scratch arrays use generation stamps, so a query never clears an O(N) visited array. Shortest path is a bidirectional BFS that always expands the smaller frontier. PageRank pulls along in-edges and spreads dangling mass over the teleport vector, the same semantics as NetworkX. Every public method takes an internal mutex, which lets the bindings release the GIL; concurrent queries are tested.

**Pruner.** Seed scores spread along edges as `relevance × hop_decay × kind_weight[edge kind]`, so calls weigh more than imports or containment. It runs as a max-product relaxation in Jacobi rounds, which means no path can exceed `max_hops`. Each candidate's score blends that relevance with normalised PageRank. A greedy knapsack then fills the budget: strong seeds (relative relevance ≥ `seed_floor`, default 0.7) go first, and everything else competes on score per token, `score / (cost / mean_cost)^cost_exponent`. Weak vector matches therefore cannot crowd out a strong match's real dependencies.

**Indexer.** Python calls resolve through:

* nested scopes, module globals and import aliases (including re-exports through `__init__`);
* `self.` / `super()` lookups that walk base classes;
* light local type inference: `x = Foo()` and `def f(x: Foo)` bind `x` to `Foo`.

C/C++ files are masked first (comments, strings, preprocessor lines), then scanned for namespace, class and function scopes, including out-of-line `Class::method` definitions and constructor initializer lists. pybind11 modules are parsed into a `module.Class.method → C++ symbol` map, which produces Python→C++ call edges.

**Healer.**

1. **REPRODUCE**: run the target in a sandbox (a temporary copy of the repo; a subprocess in its own session with CPU and file-size limits and a hard timeout).
2. **LOCALIZE**: rank suspects from traceback frames in source files, then walk the call graph out of each failing test with the C++ k-hop BFS.
3. **PROPOSE**: run a ranked mutation search over suspect functions. It flips boundary comparisons first (`<` / `<=`, the classic off-by-one), then other comparisons, `±1` constants, arithmetic, `and`/`or`, `min`/`max` and `not`. Every candidate is a single-token search/replace edit.
4. **APPLY**: screen candidates in parallel worker sandboxes, against the failing tests only.
5. **VERIFY**: run the first candidate that passes against the full suite.
6. **ROLLBACK**: restore the sandbox file whenever a candidate fails or causes a regression.

The bundled demo heals in about 1 second over 4 candidates.

An optional second proposer calls the Claude API. It turns on when the `anthropic` package and credentials are present (`ANTHROPIC_API_KEY`, or `CHRONOS_LLM=1` to use an `ant auth login` profile). It sends the failure output, the ranked suspects and a GraphRAG context, asks for search/replace edits as JSON-schema structured output, enables server-side refusal fallback, and feeds each failed attempt back for up to 3 rounds. The model defaults to `claude-opus-5-5`; set `CHRONOS_MODEL` to change it.

The healer holds a few attempts back for the Claude proposer, so a long mutation search can't use up its turn. If the API call fails (bad key, network, rate limit), the proposer reports it in the trace and the run ends normally instead of crashing.

The test suite drives this path with a simulated API client: a second demo project (`create_demo_project(path, "missing-sort")`) has a `median()` that never sorts its data. The mutation search cannot fix that, so the model has to. For a real run, set `ANTHROPIC_API_KEY` and use `make demo-llm`, or `CHRONOS_LIVE_LLM=1 make test`.

## Configuration

| variable | default | meaning |
|---|---|---|
| `CHRONOS_REPO` | current directory | repository for the MCP / API servers |
| `CHRONOS_SANDBOX` | `subprocess` | `docker` runs each test run in a throwaway container: no network, read-only root filesystem, your user id, 1 GB RAM, 1 CPU, 256 processes. On a timeout the container itself is killed. The `python:3.12-slim` + pytest image is built once, on first use |
| `CHRONOS_LLM` | auto | `0` disables the LLM proposer; `1` enables it with an `ant` profile |
| `CHRONOS_LIVE_LLM` | unset | `1` lets `make test` run the live Claude API test |
| `CHRONOS_MODEL` | `claude-opus-5-5` | model for the LLM proposer |
| `CHRONOS_WEB_DIR` | `./web` | where the API server finds `index.html` |
| `CHRONOS_NATIVE` | unset | `1` compiles with `-mcpu=native` / `-march=native` |

## API

`GET /api/health`, `GET /api/stats`, `POST /api/index {path?}`, `GET /api/graph?limit=`, `GET /api/symbol?id=`, `POST /api/retrieve {query, token_budget, hops, centrality_weight}`, `GET /api/path?src=&dst=`, `POST /api/heal {demo, test_target, apply}`, `GET /api/heal/{run_id}`, `POST /api/benchmark {nodes, queries}`, and the WebSocket at `/ws`, which carries `hello`, `telemetry` (1 Hz), `index`, `retrieval`, `path`, `heal` and `heal_done` events. Retrievals made over MCP also appear in the UI live.

## Limits

* The C/C++ front end is a scanner, not a compiler. It does not expand macros, resolve overloads, or deduce the type behind `auto x = ...; x.f()`. Calls with an unknown receiver resolve only when the method name is unique in the repo and not generic.
* The default embedder is feature-hashed TF-IDF over identifier sub-tokens: fast and dependency-free, but lexical. `HybridRetriever(index, embedder=...)` accepts any object with `fit_transform` / `transform` that returns L2-normalised rows.
* The subprocess sandbox isolates the filesystem through a copy and limits CPU and time, but it does not cut network access. Use `CHRONOS_SANDBOX=docker` for that.
* The mutation proposer fixes single-token bugs. Anything larger needs the LLM proposer, or edits passed through `try_patch`.
