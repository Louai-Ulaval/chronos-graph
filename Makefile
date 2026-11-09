PY      ?= $(if $(wildcard .venv/bin/python),.venv/bin/python,python3)
REPO    ?= .
PORT    ?= 8000
NODES   ?= 100000
QUERIES ?= 200
MCP_TRANSPORT ?= stdio
MCP_PORT ?= 8765

export PYTHONPATH := $(CURDIR)/src$(if $(PYTHONPATH),:$(PYTHONPATH))

.PHONY: help install build test benchmark ui mcp demo demo-llm clean

help:
	@echo "make install    create .venv and install requirements"
	@echo "make build      compile the C++20 extension (chronos_cpp) in place"
	@echo "make test       build, then run the end-to-end pytest suite"
	@echo "make benchmark  C++20 vs NetworkX + GraphRAG vs naive chunking (NODES=$(NODES))"
	@echo "make ui         FastAPI + WebSocket telemetry UI on http://127.0.0.1:$(PORT)"
	@echo "make mcp        MCP server over stdio (MCP_TRANSPORT=sse|streamable-http for HTTP)"
	@echo "make demo       self-heal the bundled demo project from the command line"
	@echo "make demo-llm   heal a bug only the Claude proposer can fix (needs ANTHROPIC_API_KEY, costs cents)"

install:
	python3 -m venv .venv
	.venv/bin/pip install --upgrade pip
	.venv/bin/pip install -r requirements.txt
	.venv/bin/pip install -e . --no-build-isolation

build:
	$(PY) setup.py -q build_ext --inplace

test: build
	$(PY) -m pytest -q

benchmark: build
	$(PY) -m chronos.benchmark --nodes $(NODES) --queries $(QUERIES) --repo $(REPO)

ui: build
	$(PY) -m chronos.api_server --repo $(REPO) --port $(PORT)

# Build output goes to stderr so it can never corrupt the stdio JSON-RPC stream.
mcp:
	@$(PY) setup.py -q build_ext --inplace 1>&2
	@$(PY) -m chronos.mcp_server --repo $(REPO) --transport $(MCP_TRANSPORT) --port $(MCP_PORT)

demo: build
	$(PY) -c "import tempfile, json; from chronos.sandbox_healer import create_demo_project, SelfHealer; \
	root = create_demo_project(tempfile.mkdtemp() + '/demo-inventory'); \
	r = SelfHealer(root, 'tests', on_event=lambda e: print(f\"[{e['state']:>9}] {e['message']}\")).heal(apply=True); \
	print(r.diff or r.message)"

demo-llm: build
	@test -n "$$ANTHROPIC_API_KEY" || { echo "Set ANTHROPIC_API_KEY first (this makes a few paid API calls)."; exit 1; }
	$(PY) -c "import tempfile; from chronos.sandbox_healer import create_demo_project, SelfHealer; \
	root = create_demo_project(tempfile.mkdtemp() + '/demo-stats', 'missing-sort'); \
	r = SelfHealer(root, 'tests', use_llm=True, on_event=lambda e: print(f\"[{e['state']:>9}] {e['message']}\")).heal(apply=True); \
	print(r.diff or r.message)"

clean:
	rm -rf build src/chronos_cpp*.so src/*.egg-info *.egg-info .pytest_cache benchmark_results.json
	find . -name __pycache__ -not -path "./.venv/*" -prune -exec rm -rf {} +
