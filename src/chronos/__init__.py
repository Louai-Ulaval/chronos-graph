"""Chronos-Graph: C++20-accelerated code graph engine, GraphRAG retriever, MCP server and self-healing agent."""

__version__ = "1.0.0"

try:
    import chronos_cpp  # noqa: F401  (compiled extension)
except ImportError as exc:  # pragma: no cover - only hit before `make build`
    raise ImportError(
        "The chronos_cpp extension is not built. Run `make build` (or `python setup.py build_ext --inplace`)."
    ) from exc

from .ast_indexer import CodeIndex, CodeIndexer, Symbol, index_repository
from .hybrid_retriever import HashingEmbedder, HybridRetriever, NaiveChunkRetriever, RetrievalResult
from .sandbox_healer import Patch, Edit, Sandbox, SelfHealer, create_demo_project
from .workspace import Workspace

__all__ = [
    "CodeIndex",
    "CodeIndexer",
    "Edit",
    "HashingEmbedder",
    "HybridRetriever",
    "NaiveChunkRetriever",
    "Patch",
    "RetrievalResult",
    "Sandbox",
    "SelfHealer",
    "Symbol",
    "Workspace",
    "create_demo_project",
    "index_repository",
    "__version__",
]
