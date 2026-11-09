"""Multi-language symbol indexer.

Python files are parsed with the standard-library ``ast`` module. C and C++ files
go through a brace-aware scanner that masks comments, strings and preprocessor
lines, then tracks namespace / class / function scopes. Both front-ends feed one
global resolver that turns raw call sites, imports and base classes into graph
edges between symbols.

Symbol ids
    Python  dotted module path: ``chronos.ast_indexer.CodeIndexer.index``
    C/C++   scope-qualified:    ``chronos::GraphEngine::prune`` (``::main`` for globals)
    Files   C/C++ translation units use their repo-relative path: ``src/cpp/graph_engine.cpp``
"""

from __future__ import annotations

import ast
import bisect
import builtins
import hashlib
import os
import re
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Iterable

import chronos_cpp

PY_EXTS = {".py"}
CPP_EXTS = {".c", ".h", ".cc", ".cpp", ".cxx", ".hpp", ".hh", ".hxx", ".ipp"}
IGNORED_DIRS = {
    ".git", ".hg", ".svn", ".venv", "venv", "env", "node_modules", "__pycache__", "build",
    "dist", ".eggs", ".mypy_cache", ".pytest_cache", ".ruff_cache", ".tox", "site-packages",
    ".idea", ".vscode",
}
MAX_FILE_BYTES = 1_500_000

EDGE_KINDS = {
    "call": chronos_cpp.EdgeKind.CALL,
    "import": chronos_cpp.EdgeKind.IMPORT,
    "contains": chronos_cpp.EdgeKind.CONTAINS,
    "inherits": chronos_cpp.EdgeKind.INHERITS,
    "reference": chronos_cpp.EdgeKind.REFERENCE,
}

# Method names too generic to resolve by "unique short name" fallback.
COMMON_METHODS = frozenset(
    """get set add append extend pop items keys values update copy clear close open read write
    run start stop join split strip format replace find index count sort send recv encode decode
    lower upper insert remove discard put setdefault main load dump dumps loads parse build apply
    call emit push size begin end data empty reserve resize push_back emplace_back emplace erase
    at front back swap reset release value str c_str def attr wait result submit cancel next iter
    match search sub group fill assign length name type exists mkdir unlink rename resolve info
    debug warning error exception log print render handle process execute connect fetch save""".split()
)
PY_BUILTINS = frozenset(dir(builtins))
CPP_KEYWORDS = frozenset(
    """if for while switch catch return sizeof alignof decltype do else try new delete throw case
    default static_assert using typedef static_cast dynamic_cast reinterpret_cast const_cast
    noexcept typeid alignas defined operator template typename requires co_await co_return
    co_yield""".split()
)


def estimate_tokens(text: str) -> int:
    """Cheap BPE-like token estimate (~3.8 characters per token for source code)."""
    return max(1, round(len(text) / 3.8))


@dataclass
class Symbol:
    id: str
    name: str
    kind: str  # module | file | class | function | method
    language: str  # python | cpp | c
    file: str
    line: int
    end_line: int
    signature: str = ""
    docstring: str = ""
    source: str = ""
    parent: str | None = None
    token_cost: int = 1

    def to_dict(self, with_source: bool = False) -> dict:
        d = asdict(self)
        if not with_source:
            d.pop("source", None)
        return d


@dataclass
class EdgeRec:
    src: str
    dst: str
    kind: str


@dataclass
class _RawCall:
    scope: str  # symbol id the call happens in
    target: tuple  # parsed call target, resolved later
    line: int


@dataclass
class _FileParse:
    rel: str
    language: str
    symbols: list[Symbol] = field(default_factory=list)
    calls: list[_RawCall] = field(default_factory=list)
    # Python: module id -> {local alias: qualified target}
    aliases: dict[str, dict[str, str]] = field(default_factory=dict)
    # (class id, raw base expression)
    bases: list[tuple[str, str]] = field(default_factory=list)
    imports: list[tuple[str, str]] = field(default_factory=list)  # (from symbol id, target)
    contains: list[tuple[str, str]] = field(default_factory=list)
    # C++: class qualified name -> member function names declared in the body
    members: dict[str, set[str]] = field(default_factory=dict)
    # pybind11: (python-visible qualified name, candidate C++ targets)
    bindings: list[tuple[str, list[str]]] = field(default_factory=list)
    error: str | None = None


# ---------------------------------------------------------------------------
# Python front-end
# ---------------------------------------------------------------------------


def _py_module_name(rel: str) -> str:
    parts = list(Path(rel).with_suffix("").parts)
    if parts and parts[0] in ("src", "lib"):
        parts = parts[1:]
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts) or Path(rel).stem


def _dotted(node: ast.AST) -> str | None:
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts))
    return None


def _py_call_target(func: ast.AST) -> tuple | None:
    if isinstance(func, ast.Name):
        return ("name", func.id)
    if isinstance(func, ast.Attribute):
        # super().method()
        if (
            isinstance(func.value, ast.Call)
            and isinstance(func.value.func, ast.Name)
            and func.value.func.id == "super"
        ):
            return ("super", func.attr)
        dotted = _dotted(func)
        if dotted:
            head, _, rest = dotted.partition(".")
            if head in ("self", "cls"):
                if "." in rest:
                    return ("method", rest.rsplit(".", 1)[1])
                return ("self", rest)
            return ("dotted", dotted)
        return ("method", func.attr)
    return None


class _PyParser:
    def __init__(self, rel: str, text: str):
        self.rel = rel
        self.text = text
        self.lines = text.splitlines()
        self.module = _py_module_name(rel)
        self.out = _FileParse(rel=rel, language="python")
        self.aliases: dict[str, str] = {}
        self.package = self.module if rel.endswith("__init__.py") else self.module.rpartition(".")[0]

    def _segment(self, node: ast.AST) -> str:
        start = getattr(node, "lineno", 1) - 1
        end = getattr(node, "end_lineno", start + 1)
        if getattr(node, "decorator_list", None):
            start = min(d.lineno for d in node.decorator_list) - 1
        return "\n".join(self.lines[start:end])

    def _resolve_relative(self, module: str | None, level: int) -> str:
        if level == 0:
            return module or ""
        base = self.package.split(".") if self.package else []
        if level > 1:
            base = base[: len(base) - (level - 1)] if level - 1 <= len(base) else []
        prefix = ".".join(base)
        if module:
            return f"{prefix}.{module}" if prefix else module
        return prefix

    def _record_import(self, node: ast.AST, scope: str) -> None:
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.asname:
                    self.aliases[a.asname] = a.name
                else:
                    head = a.name.split(".")[0]
                    self.aliases.setdefault(head, head)
                self.out.imports.append((scope, a.name))
        elif isinstance(node, ast.ImportFrom):
            mod = self._resolve_relative(node.module, node.level)
            for a in node.names:
                if a.name == "*":
                    continue
                target = f"{mod}.{a.name}" if mod else a.name
                self.aliases[a.asname or a.name] = target
                self.out.imports.append((scope, target))

    @staticmethod
    def _local_types(fn: ast.AST | None, body: list[ast.AST]) -> dict[str, str]:
        """Light type inference: ``x = Foo(...)`` and ``def f(x: Foo)`` bind x to Foo."""
        types: dict[str, str] = {}
        if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for a in fn.args.args + fn.args.kwonlyargs + fn.args.posonlyargs:
                ann = a.annotation
                if isinstance(ann, ast.Constant) and isinstance(ann.value, str):
                    types[a.arg] = ann.value
                elif ann is not None and (name := _dotted(ann)):
                    types[a.arg] = name
        stack = list(body)
        while stack:
            n = stack.pop()
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
                continue
            if isinstance(n, (ast.Assign, ast.AnnAssign)):
                targets = n.targets if isinstance(n, ast.Assign) else [n.target]
                value = n.value
                if len(targets) == 1 and isinstance(targets[0], ast.Name):
                    if isinstance(value, ast.Call) and (name := _dotted(value.func)):
                        if name.rsplit(".", 1)[-1][:1].isupper():  # constructor-looking call
                            types[targets[0].id] = name
                    elif isinstance(n, ast.AnnAssign) and (name := _dotted(n.annotation)):
                        types[targets[0].id] = name
            stack.extend(ast.iter_child_nodes(n))
        return types

    def _calls_in(self, nodes: Iterable[ast.AST], scope: str, fn: ast.AST | None = None) -> None:
        nodes = list(nodes)
        local_types = self._local_types(fn, nodes)
        stack = nodes
        while stack:
            n = stack.pop()
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            if isinstance(n, (ast.Import, ast.ImportFrom)):
                self._record_import(n, scope)
                continue
            if isinstance(n, ast.Call):
                tgt = _py_call_target(n.func)
                if tgt and tgt[0] == "dotted":
                    head, _, rest = tgt[1].partition(".")
                    if head in local_types and "." not in rest:
                        tgt = ("typed", local_types[head], rest)
                if tgt:
                    self.out.calls.append(_RawCall(scope, tgt, n.lineno))
            stack.extend(ast.iter_child_nodes(n))

    def parse(self) -> _FileParse:
        try:
            tree = ast.parse(self.text, filename=self.rel)
        except SyntaxError as exc:
            self.out.error = f"SyntaxError: {exc.msg} (line {exc.lineno})"
            return self.out

        doc = ast.get_docstring(tree) or ""
        top_names = [
            n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        ]
        skeleton = f'"""{doc}"""\n' if doc else ""
        skeleton += f"# module {self.module}: " + ", ".join(top_names[:40])
        mod_sym = Symbol(
            id=self.module, name=self.module.rpartition(".")[2] or self.module, kind="module",
            language="python", file=self.rel, line=1, end_line=len(self.lines), signature=f"module {self.module}",
            docstring=doc, source=skeleton, token_cost=estimate_tokens(skeleton),
        )
        self.out.symbols.append(mod_sym)
        for node in tree.body:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                self._record_import(node, self.module)
        self._walk_body(tree.body, parent=self.module, class_id=None)
        self._calls_in(
            [n for n in tree.body if not isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))],
            self.module,
        )
        self.out.aliases[self.module] = self.aliases
        return self.out

    def _walk_body(self, body: list[ast.stmt], parent: str, class_id: str | None) -> None:
        for node in body:
            if isinstance(node, ast.ClassDef):
                self._class(node, parent)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self._function(node, parent, class_id)
            elif isinstance(node, (ast.If, ast.Try, ast.With)):
                # defs guarded by `if TYPE_CHECKING:` / try-imports still count
                nested = []
                for attr in ("body", "orelse", "finalbody"):
                    nested.extend(getattr(node, attr, []) or [])
                for h in getattr(node, "handlers", []) or []:
                    nested.extend(h.body)
                self._walk_body(nested, parent, class_id)

    def _class(self, node: ast.ClassDef, parent: str) -> None:
        cid = f"{parent}.{node.name}"
        doc = ast.get_docstring(node) or ""
        bases = [b for b in (_dotted(x) for x in node.bases) if b]
        header = f"class {node.name}({', '.join(bases)}):" if bases else f"class {node.name}:"
        methods = []
        for item in node.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                prefix = "async def" if isinstance(item, ast.AsyncFunctionDef) else "def"
                methods.append(f"    {prefix} {item.name}({ast.unparse(item.args)})")
        skeleton = header + (f'\n    """{doc}"""' if doc else "") + ("\n" + "\n".join(methods) if methods else "")
        sym = Symbol(
            id=cid, name=node.name, kind="class", language="python", file=self.rel, line=node.lineno,
            end_line=node.end_lineno or node.lineno, signature=header, docstring=doc, source=skeleton,
            parent=parent, token_cost=estimate_tokens(skeleton),
        )
        self.out.symbols.append(sym)
        self.out.contains.append((parent, cid))
        for b in bases:
            self.out.bases.append((cid, b))
        self._walk_body(node.body, parent=cid, class_id=cid)
        self._calls_in(
            [n for n in node.body if not isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))],
            cid,
        )

    def _function(self, node: ast.FunctionDef | ast.AsyncFunctionDef, parent: str, class_id: str | None) -> None:
        fid = f"{parent}.{node.name}"
        src = self._segment(node)
        prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
        ret = f" -> {ast.unparse(node.returns)}" if node.returns is not None else ""
        sig = f"{prefix} {node.name}({ast.unparse(node.args)}){ret}"
        sym = Symbol(
            id=fid, name=node.name, kind="method" if class_id == parent else "function", language="python",
            file=self.rel, line=node.lineno, end_line=node.end_lineno or node.lineno, signature=sig,
            docstring=ast.get_docstring(node) or "", source=src, parent=parent, token_cost=estimate_tokens(src),
        )
        self.out.symbols.append(sym)
        self.out.contains.append((parent, fid))
        self._calls_in(node.body, fid, fn=node)
        # nested defs belong to this function's scope
        self._walk_body(node.body, parent=fid, class_id=None)


# ---------------------------------------------------------------------------
# C / C++ front-end
# ---------------------------------------------------------------------------

_CPP_MASK = re.compile(
    r'R"(?P<d>[^(\s"\\]{0,16})\((?:.|\n)*?\)(?P=d)"'  # raw strings
    r"|//[^\n]*"
    r"|/\*.*?\*/"
    r'|"(?:\\.|[^"\\\n])*"'
    r"|'(?:\\.|[^'\\\n])*'",
    re.S,
)
_CPP_PREPROC = re.compile(r"^[ \t]*#(?:[^\n]*\\\n)*[^\n]*", re.M)
_CPP_INCLUDE = re.compile(r'^[ \t]*#[ \t]*include[ \t]*[<"]([^>"]+)[>"]', re.M)
_CPP_NAMESPACE = re.compile(r"(?:^|\s)(?:inline\s+)?namespace\s*([\w:]*)\s*$")
_CPP_CLASS = re.compile(
    r"(?:^|[\s;>])(class|struct|union)\s+(?:alignas\s*\([^)]*\)\s*)?(?:\[\[[^\]]*\]\]\s*)?"
    r"(?:[A-Z_][A-Z0-9_]*\s+)?([A-Za-z_]\w*)\s*(?:<[^{]*>)?\s*(?:final\s*)?(?::\s*([^{]*))?$"
)
_CPP_FUNC_NAME = re.compile(r"(~?[A-Za-z_]\w*(?:\s*::\s*~?[A-Za-z_]\w*)*|operator\s*[^\s(]+)\s*(?:<[^()]*>)?\s*$")
_CPP_FREE_CALL = re.compile(
    r"(?<![\w.~])(?<!->)(::)?([A-Za-z_]\w*(?:::[A-Za-z_~]\w*)*)\s*(?:<[^<>;{}()]*(?:<[^<>;{}()]*>[^<>;{}()]*)*>)?\s*\("
)
_CPP_MEMBER_CALL = re.compile(r"(this\s*)?(?:\.|->)\s*([A-Za-z_]\w*)\s*(?:<[^<>;{}()]*>)?\s*\(")
_CPP_MEMBER_DECL = re.compile(r"(~?[A-Za-z_]\w*)\s*\(")
_PYBIND_MODULE = re.compile(r"PYBIND11_MODULE\s*\(\s*(\w+)\s*,\s*(\w+)\s*\)")
_PYBIND_CLASS = re.compile(r'(?:py|pybind11)::class_<\s*([\w:]+)[^>]*>\s*\(\s*\w+\s*,\s*"(\w+)"')
_PYBIND_DEF = re.compile(r'\.def(?:_static|_property_readonly|_property)?\s*\(\s*"(\w+)"\s*,\s*(&[\w:]+|\[)')


def _mask_cpp(text: str) -> str:
    def blank(m: re.Match) -> str:
        s = m.group(0)
        return "".join(c if c == "\n" else " " for c in s)

    masked = _CPP_MASK.sub(blank, text)
    return _CPP_PREPROC.sub(blank, masked)


def _matching_paren(s: str, open_idx: int) -> int:
    depth = 0
    for i in range(open_idx, len(s)):
        if s[i] == "(":
            depth += 1
        elif s[i] == ")":
            depth -= 1
            if depth == 0:
                return i
    return -1


def _cpp_function_name(header: str) -> tuple[str, str] | None:
    """Return (qualified name, signature) if the header text before ``{`` is a function definition."""
    h = header.strip()
    if not h or "(" not in h or h.startswith(("=", ",")):
        return None
    first = h.index("(")
    prefix = h[:first]
    if prefix.rstrip().endswith("operator"):  # operator()(...)
        close = _matching_paren(h, first)
        if close < 0 or close + 1 >= len(h) or "(" not in h[close + 1:]:
            return None
        first = h.index("(", close + 1)
        prefix = h[:first]
    if "=" in prefix.replace("operator=", "").replace("operator==", "") or ";" in prefix:
        return None
    m = _CPP_FUNC_NAME.search(prefix)
    if not m:
        return None
    name = re.sub(r"\s+", "", m.group(1))
    last = name.rsplit("::", 1)[-1].lstrip("~")
    if last in CPP_KEYWORDS and not name.startswith("operator"):
        return None
    lead = prefix[: m.start()].split()
    if any(tok in ("return", "if", "while", "for", "switch", "else", "case", "throw") for tok in lead):
        return None
    close = _matching_paren(h, first)
    if close < 0:
        return None
    tail = h[close + 1:]
    if "=" in tail.split(":", 1)[0] or ";" in tail:
        return None
    sig = re.sub(r"\s+", " ", h[: close + 1]).strip()
    tail_q = re.match(r"\s*((?:const|noexcept|override|final|volatile|&&?|\s)*)", tail)
    if tail_q and tail_q.group(1).strip():
        sig += " " + re.sub(r"\s+", " ", tail_q.group(1)).strip()
    return name, sig


@dataclass
class _Scope:
    kind: str  # file | namespace | linkage | class | function | block | other | initlist
    name: str = ""
    sym: Symbol | None = None
    start: int = 0


class _CppParser:
    def __init__(self, rel: str, text: str):
        self.rel = rel
        self.text = text
        self.lang = "c" if Path(rel).suffix == ".c" else "cpp"
        self.out = _FileParse(rel=rel, language=self.lang)
        self.newlines = [i for i, c in enumerate(text) if c == "\n"]
        self.seen_ids: dict[str, int] = {}

    def _line(self, pos: int) -> int:
        return bisect.bisect_right(self.newlines, pos - 1) + 1

    def _unique(self, sid: str) -> str:
        n = self.seen_ids.get(sid, 0) + 1
        self.seen_ids[sid] = n
        return sid if n == 1 else f"{sid}#{n}"

    def parse(self) -> _FileParse:
        text = self.text
        masked = _mask_cpp(text)
        file_sym = Symbol(
            id=self.rel, name=Path(self.rel).name, kind="file", language=self.lang, file=self.rel, line=1,
            end_line=len(self.newlines) + 1, signature=f"file {self.rel}",
        )
        self.out.symbols.append(file_sym)
        for inc in _CPP_INCLUDE.findall(text):
            self.out.imports.append((self.rel, inc))

        stack: list[_Scope] = [_Scope("file")]
        header_start = 0
        top_names: list[str] = []
        for m in re.finditer(r"[{};]", masked):
            ch, pos = m.group(), m.start()
            top = stack[-1]
            if top.kind in ("function", "block", "other", "initlist"):
                if ch == "{":
                    stack.append(_Scope("block", start=pos))
                elif ch == "}":
                    closed = stack.pop()
                    if closed.kind == "function":
                        self._finish_function(closed, pos, masked)
                        header_start = pos + 1
                    elif closed.kind == "other":
                        header_start = pos + 1
                    # initlist: keep accumulating the function header
                continue

            if ch == ";":
                if top.kind == "class":
                    decl = masked[header_start:pos]
                    mm = _CPP_MEMBER_DECL.search(decl)
                    if mm and mm.group(1) not in CPP_KEYWORDS:
                        self.out.members.setdefault(top.name, set()).add(mm.group(1).lstrip("~"))
                header_start = pos + 1
                continue
            if ch == "}":
                if len(stack) > 1:
                    closed = stack.pop()
                    if closed.kind == "class" and closed.sym is not None:
                        closed.sym.end_line = self._line(pos)
                        body_src = text[closed.start:pos + 1]
                        closed.sym.source = self._class_skeleton(closed.sym, body_src)
                        closed.sym.token_cost = estimate_tokens(closed.sym.source)
                header_start = pos + 1
                continue

            # ch == "{" at structural scope
            header = masked[header_start:pos]
            stripped = header.rstrip()
            if (
                stripped
                and (stripped[-1].isalnum() or stripped[-1] == "_")
                and re.search(r"\)\s*(?:const\s*|noexcept\s*)*:[^;]*$", header)
                and not _CPP_NAMESPACE.search(header)
            ):
                stack.append(_Scope("initlist", start=pos))  # Foo::Foo(int x) : a_(x), b_{x} {
                continue
            scope = self._classify(header, header_start, pos, stack, top_names)
            stack.append(scope)
            header_start = pos + 1

        doc_names = ", ".join(top_names[:40])
        file_sym.source = f"// file {self.rel}: {doc_names}"
        file_sym.token_cost = estimate_tokens(file_sym.source)
        self._pybind11(text)
        return self.out

    def _pybind11(self, text: str) -> None:
        """Map pybind11-exposed names (module.Class.method) onto the C++ symbols behind them."""
        mod = _PYBIND_MODULE.search(text)
        if not mod:
            return
        modname, mvar = mod.group(1), mod.group(2)
        self.out.bindings.append((modname, [self.rel]))
        classes = [(m.start(), m.group(1), m.group(2)) for m in _PYBIND_CLASS.finditer(text)]
        for cpp, py in ((c, p) for _, c, p in classes):
            self.out.bindings.append((f"{modname}.{py}", [cpp]))
        defs = list(_PYBIND_DEF.finditer(text))
        for i, m in enumerate(defs):
            pos, pyname, target = m.start(), m.group(1), m.group(2)
            segment = text[m.end(): defs[i + 1].start() if i + 1 < len(defs) else len(text)]
            is_module_fn = text[max(0, pos - len(mvar)):pos] == mvar
            owner = next((c for c in reversed(classes) if c[0] < pos), None)
            if is_module_fn or owner is None:
                py_name = f"{modname}.{pyname}"
                cands = [target[1:]] if target.startswith("&") else [pyname]
            else:
                _, cls_cpp, cls_py = owner
                py_name = f"{modname}.{cls_py}.{pyname}"
                if target.startswith("&"):
                    cands = [target[1:]]
                else:  # lambda: same-named method first, then whatever methods the lambda calls
                    called = re.findall(r"\b\w+\s*(?:\.|->)\s*(\w+)\s*\(", segment)
                    cands = [f"{cls_cpp}::{pyname}"] + [f"{cls_cpp}::{c}" for c in called]
            self.out.bindings.append((py_name, cands))

    def _qual(self, stack: list[_Scope], name: str) -> str:
        parts = [s.name for s in stack if s.kind in ("namespace", "class") and s.name]
        return "::".join(parts + [name])

    def _classify(self, header: str, hstart: int, pos: int, stack: list[_Scope], top_names: list[str]) -> _Scope:
        h = header.strip()
        if re.search(r"\benum\b", h):
            return _Scope("other", start=pos)
        m = _CPP_NAMESPACE.search(header)
        if m:
            return _Scope("namespace", name=m.group(1).replace(" ", ""), start=pos)
        if h.startswith("extern") and "(" not in h:
            return _Scope("linkage", start=pos)
        m = _CPP_CLASS.search(header)
        if m and "(" not in header[: m.start(1)] and "=" not in header:
            name = m.group(2)
            qual = self._qual(stack, name)
            parent = self._parent_id(stack)
            sid = self._unique(qual)
            sig_start = hstart + (len(header) - len(header.lstrip()))
            sym = Symbol(
                id=sid, name=name, kind="class", language=self.lang, file=self.rel, line=self._line(sig_start),
                end_line=self._line(pos), signature=re.sub(r"\s+", " ", h), parent=parent,
                docstring=self._leading_comment(hstart, sig_start),
            )
            self.out.symbols.append(sym)
            self.out.contains.append((parent, sid))
            if m.group(3):
                for base in m.group(3).split(","):
                    base = re.sub(r"\b(public|private|protected|virtual)\b", "", base).strip()
                    base = re.sub(r"<.*", "", base).strip()
                    if base:
                        self.out.bases.append((sid, base))
            top_names.append(name)
            return _Scope("class", name=name, sym=sym, start=sig_start)
        fn = _cpp_function_name(header)
        if fn:
            name, sig = fn
            qual = self._qual(stack, name)
            if "::" not in qual:
                qual = "::" + qual
            owner = None
            classes = [s for s in stack if s.kind == "class"]
            if classes and classes[-1].sym is not None:
                owner = classes[-1].sym.id
            elif "::" in name:
                owner = self._qual(stack, name.rsplit("::", 1)[0])
            sid = self._unique(qual)
            sig_start = hstart + (len(header) - len(header.lstrip()))
            sym = Symbol(
                id=sid, name=name.rsplit("::", 1)[-1], kind="method" if owner else "function",
                language=self.lang, file=self.rel, line=self._line(sig_start), end_line=self._line(pos),
                signature=sig, parent=owner or self._parent_id(stack),
                docstring=self._leading_comment(hstart, sig_start),
            )
            top_names.append(sym.name)
            return _Scope("function", name=owner or "", sym=sym, start=sig_start)
        return _Scope("other", start=pos)

    def _parent_id(self, stack: list[_Scope]) -> str:
        for s in reversed(stack):
            if s.kind == "class" and s.sym is not None:
                return s.sym.id
        return self.rel

    def _leading_comment(self, hstart: int, sig_start: int) -> str:
        chunk = self.text[hstart:sig_start]
        lines = [ln.strip() for ln in chunk.splitlines()]
        out = []
        for ln in lines:
            if ln.startswith(("//", "/*", "*")):
                out.append(ln.lstrip("/*! ").rstrip("*/ ").strip())
        return "\n".join(x for x in out if x)

    def _class_skeleton(self, sym: Symbol, body: str) -> str:
        decls = []
        depth = 0
        for line in body.splitlines()[1:]:
            stripped = line.strip()
            if depth == 0 and stripped and "(" in stripped and not stripped.startswith(("//", "*", "/*", "#")):
                decls.append("  " + stripped.rstrip("{").strip())
            depth += line.count("{") - line.count("}")
            depth = max(depth, 0)
        doc = f"// {sym.docstring}\n" if sym.docstring else ""
        return doc + sym.signature + " {\n" + "\n".join(decls[:60]) + "\n};"

    def _finish_function(self, scope: _Scope, close_pos: int, masked: str) -> None:
        sym = scope.sym
        if sym is None:
            return
        sym.end_line = self._line(close_pos)
        sym.source = self.text[scope.start:close_pos + 1]
        sym.token_cost = estimate_tokens(sym.source)
        self.out.symbols.append(sym)
        self.out.contains.append((sym.parent or self.rel, sym.id))
        body = masked[scope.start:close_pos + 1]
        brace = body.find("{")
        body_start = scope.start + max(brace, 0)
        body = masked[body_start:close_pos + 1]
        for m in _CPP_FREE_CALL.finditer(body):
            name = m.group(2)
            if name.rsplit("::", 1)[-1] in CPP_KEYWORDS:
                continue
            line = self._line(body_start + m.start())
            self.out.calls.append(_RawCall(sym.id, ("cpp", name, bool(m.group(1)), scope.name), line))
        for m in _CPP_MEMBER_CALL.finditer(body):
            line = self._line(body_start + m.start())
            kind = "cpp_this" if m.group(1) else "cpp_member"
            self.out.calls.append(_RawCall(sym.id, (kind, m.group(2), scope.name), line))


# ---------------------------------------------------------------------------
# Index + resolver
# ---------------------------------------------------------------------------


@dataclass
class CodeIndex:
    root: str
    symbols: dict[str, Symbol] = field(default_factory=dict)
    edges: list[EdgeRec] = field(default_factory=list)
    files: dict[str, dict] = field(default_factory=dict)
    errors: dict[str, str] = field(default_factory=dict)
    unresolved_calls: int = 0
    resolved_calls: int = 0
    elapsed_ms: float = 0.0
    _engine: chronos_cpp.GraphEngine | None = field(default=None, repr=False)
    ids: list[str] = field(default_factory=list, repr=False)
    id_to_idx: dict[str, int] = field(default_factory=dict, repr=False)

    # ---- graph ------------------------------------------------------------
    @property
    def engine(self) -> chronos_cpp.GraphEngine:
        if self._engine is None:
            self._engine = self.build_engine()
        return self._engine

    def build_engine(self) -> chronos_cpp.GraphEngine:
        import numpy as np

        g = chronos_cpp.GraphEngine()
        g.reserve(len(self.symbols), len(self.edges))
        self.ids = list(self.symbols)
        self.id_to_idx = {}
        for sid in self.ids:
            s = self.symbols[sid]
            self.id_to_idx[sid] = g.add_node(sid, s.kind, s.file, s.line, s.token_cost)
        by_kind: dict[str, tuple[list[int], list[int]]] = defaultdict(lambda: ([], []))
        for e in self.edges:
            a, b = self.id_to_idx.get(e.src), self.id_to_idx.get(e.dst)
            if a is None or b is None:
                continue
            by_kind[e.kind][0].append(a)
            by_kind[e.kind][1].append(b)
        # Lower EdgeKind values win when the same pair appears twice, so insert all kinds.
        for kind, (src, dst) in by_kind.items():
            g.add_edges(np.asarray(src, dtype=np.uint32), np.asarray(dst, dtype=np.uint32), EDGE_KINDS[kind])
        g.finalize()
        return g

    def to_networkx(self):
        import networkx as nx

        G = nx.DiGraph()
        for sid, s in self.symbols.items():
            G.add_node(sid, kind=s.kind, file=s.file, line=s.line, token_cost=s.token_cost)
        for e in self.edges:
            if not G.has_edge(e.src, e.dst):
                G.add_edge(e.src, e.dst, kind=e.kind)
        return G

    # ---- queries ----------------------------------------------------------
    def find(self, name: str, limit: int = 10) -> list[Symbol]:
        """Look a symbol up by exact id, qualified suffix, or short name."""
        if name in self.symbols:
            return [self.symbols[name]]
        n = name.strip()
        suffix = [s for sid, s in self.symbols.items() if sid.endswith("." + n) or sid.endswith("::" + n)]
        if suffix:
            return sorted(suffix, key=lambda s: (len(s.id), s.id))[:limit]
        short = [s for s in self.symbols.values() if s.name == n]
        if short:
            return sorted(short, key=lambda s: (len(s.id), s.id))[:limit]
        low = n.lower()
        fuzzy = [s for sid, s in self.symbols.items() if low in sid.lower()]
        return sorted(fuzzy, key=lambda s: (len(s.id), s.id))[:limit]

    def callees(self, sid: str) -> list[str]:
        return sorted({e.dst for e in self.edges if e.src == sid and e.kind == "call"})

    def callers(self, sid: str) -> list[str]:
        return sorted({e.src for e in self.edges if e.dst == sid and e.kind == "call"})

    def stats(self) -> dict:
        kinds: dict[str, int] = defaultdict(int)
        langs: dict[str, int] = defaultdict(int)
        for s in self.symbols.values():
            kinds[s.kind] += 1
            langs[s.language] += 1
        ekinds: dict[str, int] = defaultdict(int)
        for e in self.edges:
            ekinds[e.kind] += 1
        total = self.resolved_calls + self.unresolved_calls
        return {
            "root": self.root,
            "files": len(self.files),
            "symbols": len(self.symbols),
            "edges": len(self.edges),
            "symbols_by_kind": dict(kinds),
            "symbols_by_language": dict(langs),
            "edges_by_kind": dict(ekinds),
            "call_resolution_rate": round(self.resolved_calls / total, 4) if total else 1.0,
            "parse_errors": len(self.errors),
            "elapsed_ms": round(self.elapsed_ms, 2),
        }


class CodeIndexer:
    """Walks a repository, parses every supported file and resolves cross-file edges.

    Parsed files are cached by content hash, so re-indexing after a small patch only
    re-parses the files that changed.
    """

    def __init__(self, include_tests: bool = True):
        self.include_tests = include_tests
        self._cache: dict[str, tuple[str, _FileParse]] = {}

    # ---- discovery --------------------------------------------------------
    def discover(self, root: str | os.PathLike) -> list[Path]:
        root = Path(root)
        if root.is_file():
            return [root]
        found = []
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = sorted(d for d in dirnames if d not in IGNORED_DIRS and not d.startswith("."))
            for fn in sorted(filenames):
                p = Path(dirpath) / fn
                if p.suffix in PY_EXTS | CPP_EXTS:
                    if not self.include_tests and ("test" in p.parts[-1] and p.suffix == ".py"):
                        continue
                    try:
                        if p.stat().st_size <= MAX_FILE_BYTES:
                            found.append(p)
                    except OSError:
                        continue
        return found

    def parse_file(self, path: Path, rel: str) -> _FileParse:
        raw = path.read_bytes()
        digest = hashlib.sha1(raw).hexdigest()
        cached = self._cache.get(rel)
        if cached and cached[0] == digest:
            return cached[1]
        text = raw.decode("utf-8", errors="replace")
        parser = _PyParser(rel, text) if path.suffix in PY_EXTS else _CppParser(rel, text)
        parsed = parser.parse()
        self._cache[rel] = (digest, parsed)
        return parsed

    # ---- main entry -------------------------------------------------------
    def index(
        self,
        root: str | os.PathLike,
        progress: Callable[[int, int, str], None] | None = None,
    ) -> CodeIndex:
        t0 = time.perf_counter()
        root_path = Path(root).resolve()
        base = root_path if root_path.is_dir() else root_path.parent
        files = self.discover(root_path)
        idx = CodeIndex(root=str(root_path))
        parses: list[_FileParse] = []
        for i, p in enumerate(files):
            rel = p.relative_to(base).as_posix()
            try:
                fp = self.parse_file(p, rel)
            except OSError as exc:
                idx.errors[rel] = str(exc)
                continue
            parses.append(fp)
            if fp.error:
                idx.errors[rel] = fp.error
            idx.files[rel] = {"language": fp.language, "symbols": len(fp.symbols)}
            if progress:
                progress(i + 1, len(files), rel)
        for fp in parses:
            for s in fp.symbols:
                if s.id not in idx.symbols:
                    idx.symbols[s.id] = s
        _Resolver(idx, parses).run()
        idx.elapsed_ms = (time.perf_counter() - t0) * 1000
        return idx


class _Resolver:
    def __init__(self, idx: CodeIndex, parses: list[_FileParse]):
        self.idx = idx
        self.parses = parses
        self.syms = idx.symbols
        self.aliases: dict[str, dict[str, str]] = {}
        self.members: dict[str, set[str]] = defaultdict(set)
        for fp in parses:
            self.aliases.update(fp.aliases)
            for cls, names in fp.members.items():
                self.members[cls] |= names
        self.by_short: dict[str, list[str]] = defaultdict(list)
        self.cpp_by_short: dict[str, list[str]] = defaultdict(list)
        self.cpp_by_suffix: dict[str, list[str]] = defaultdict(list)
        for sid, s in self.syms.items():
            if s.kind in ("function", "method", "class"):
                if s.language == "python":
                    self.by_short[s.name].append(sid)
                else:
                    if "#" in sid:
                        continue
                    self.cpp_by_short[s.name].append(sid)
                    parts = sid.lstrip(":").split("::")
                    for k in range(1, len(parts) + 1):
                        self.cpp_by_suffix["::".join(parts[-k:])].append(sid)
        self.bases: dict[str, list[str]] = defaultdict(list)
        self.files_by_name: dict[str, list[str]] = defaultdict(list)
        for rel in idx.files:
            self.files_by_name[Path(rel).name].append(rel)
        self.edge_set: set[tuple[str, str, str]] = set()
        # pybind11 bridge: python-visible name -> C++ symbol id
        self.bound: dict[str, str] = {}
        self.bound_by_method: dict[str, list[str]] = defaultdict(list)
        for fp in parses:
            for py_name, cands in fp.bindings:
                for c in cands:
                    hit = c if c in self.syms else None
                    if hit is None:
                        sfx = self.cpp_by_suffix.get(c.lstrip(":"), [])
                        hit = sfx[0] if len(sfx) == 1 else None
                    if hit:
                        self.bound[py_name] = hit
                        if py_name.count(".") >= 2:
                            self.bound_by_method[py_name.rsplit(".", 1)[1]].append(hit)
                        break

    def edge(self, src: str, dst: str, kind: str) -> None:
        if src not in self.syms or dst not in self.syms:
            return
        if src == dst and kind != "call":  # only recursion may loop back
            return
        key = (src, dst, kind)
        if key not in self.edge_set:
            self.edge_set.add(key)
            self.idx.edges.append(EdgeRec(src, dst, kind))

    # ---- python helpers -----------------------------------------------------
    def _module_of(self, sid: str) -> str:
        s = self.syms.get(sid)
        while s is not None and s.kind not in ("module", "file"):
            s = self.syms.get(s.parent or "")
        return s.id if s is not None else sid

    def _class_of(self, sid: str) -> str | None:
        s = self.syms.get(sid)
        while s is not None:
            if s.kind == "class":
                return s.id
            s = self.syms.get(s.parent or "")
        return None

    def resolve_qualified(self, name: str, depth: int = 0) -> str | None:
        if name in self.syms:
            return name
        if name in self.bound:
            return self.bound[name]
        if depth > 4:
            return None
        parts = name.split(".")
        for cut in range(len(parts) - 1, 0, -1):
            mod = ".".join(parts[:cut])
            if mod in self.aliases and parts[cut] in self.aliases[mod]:
                target = self.aliases[mod][parts[cut]]
                rest = parts[cut + 1:]
                return self.resolve_qualified(".".join([target] + rest), depth + 1)
            if mod in self.syms:
                break
        return None

    def _method_in_class(self, cls: str, attr: str, depth: int = 0) -> str | None:
        cand = f"{cls}.{attr}"
        if cand in self.syms:
            return cand
        if depth > 5:
            return None
        for base in self.bases.get(cls, []):
            hit = self._method_in_class(base, attr, depth + 1)
            if hit:
                return hit
        return None

    def _unique_short(self, name: str, table: dict[str, list[str]]) -> str | None:
        if name in COMMON_METHODS or name in PY_BUILTINS:
            return None
        cands = table.get(name, [])
        return cands[0] if len(cands) == 1 else None

    def resolve_py(self, call: _RawCall) -> str | None:
        kind = call.target[0]
        scope = call.scope
        module = self._module_of(scope)
        aliases = self.aliases.get(module, {})
        if kind == "name":
            n = call.target[1]
            # nested function, then sibling in the enclosing scopes, then module level
            s = self.syms.get(scope)
            chain = []
            while s is not None:
                chain.append(s.id)
                s = self.syms.get(s.parent or "")
            for container in chain:
                if self.syms[container].kind == "class":
                    continue
                cand = f"{container}.{n}"
                if cand in self.syms:
                    return cand
            if n in aliases:
                hit = self.resolve_qualified(aliases[n])
                if hit:
                    return hit
            if n in PY_BUILTINS:
                return None
            return self._unique_short(n, self.by_short)
        if kind in ("self", "super"):
            attr = call.target[1]
            cls = self._class_of(scope)
            if cls:
                if kind == "super":
                    for base in self.bases.get(cls, []):
                        hit = self._method_in_class(base, attr)
                        if hit:
                            return hit
                else:
                    hit = self._method_in_class(cls, attr)
                    if hit:
                        return hit
            return self._unique_short(attr, self.by_short)
        if kind == "dotted":
            dotted = call.target[1]
            head, _, rest = dotted.partition(".")
            if head in aliases:
                hit = self.resolve_qualified(f"{aliases[head]}.{rest}")
                if hit:
                    return hit
            hit = self.resolve_qualified(f"{module}.{dotted}")
            if hit:
                return hit
            hit = self.resolve_qualified(dotted)
            if hit:
                return hit
            if head in aliases or head in PY_BUILTINS:
                if "." in rest:
                    return self._bound_method(rest.rsplit(".", 1)[1])
                return None  # call into an external module: leave unresolved
            return self._unique_short(dotted.rsplit(".", 1)[1], self.by_short)
        if kind == "typed":
            type_expr, attr = call.target[1], call.target[2]
            sub = "dotted" if "." in type_expr else "name"
            cls = self.resolve_py(_RawCall(scope, (sub, type_expr), call.line))
            if cls and self.syms[cls].kind == "class":
                if self.syms[cls].language != "python":  # bound C++ class
                    cand = f"{cls}::{attr}"
                    return cand if cand in self.syms else self._bound_method(attr)
                hit = self._method_in_class(cls, attr)
                if hit:
                    return hit
            return self._unique_short(attr, self.by_short)
        if kind == "method":
            attr = call.target[1]
            hit = self._unique_short(attr, self.by_short)
            if hit == scope:  # an unknown receiver calling a same-named method is almost never recursion
                hit = None
            if hit is None and set(self.by_short.get(attr, [])) <= {scope}:
                hit = self._bound_method(attr)
            return hit
        return None

    def _bound_method(self, name: str) -> str | None:
        cands = sorted(set(self.bound_by_method.get(name, [])))
        return cands[0] if len(cands) == 1 else None

    # ---- C++ helpers ----------------------------------------------------------
    def resolve_cpp(self, call: _RawCall) -> str | None:
        kind = call.target[0]
        scope_sym = self.syms.get(call.scope)
        owner = call.target[-1] or (scope_sym.parent if scope_sym else "") or ""
        if kind == "cpp":
            name, rooted = call.target[1], call.target[2]
            if rooted or "::" in name:
                key = name.lstrip(":")
                cands = self.cpp_by_suffix.get(key, [])
                if len(cands) == 1:
                    return cands[0]
                if cands:
                    return sorted(cands, key=len)[0]
                return None
            if owner and owner in self.syms:
                cand = f"{owner}::{name}"
                if cand in self.syms:
                    return cand
                for base in self.bases.get(owner, []):
                    cand = f"{base}::{name}"
                    if cand in self.syms:
                        return cand
            # enclosing namespaces, innermost first
            if scope_sym is not None:
                ns_parts = scope_sym.id.lstrip(":").split("::")[:-1]
                for k in range(len(ns_parts), -1, -1):
                    cand = "::".join(ns_parts[:k] + [name])
                    cand = cand if "::" in cand else "::" + cand
                    if cand in self.syms and self.syms[cand].kind in ("function", "method", "class"):
                        return cand
            return self._unique_short(name, self.cpp_by_short)
        if kind == "cpp_this":
            name = call.target[1]
            if owner and owner in self.syms:
                cand = f"{owner}::{name}"
                if cand in self.syms:
                    return cand
            return self._unique_short(name, self.cpp_by_short)
        if kind == "cpp_member":
            # obj.method() / ptr->method(): the receiver type is unknown, so only a
            # repo-unique, non-generic method name is trusted.
            return self._unique_short(call.target[1], self.cpp_by_short)
        return None

    # ---- run ------------------------------------------------------------------
    def run(self) -> None:
        idx = self.idx
        for fp in self.parses:
            for parent, child in fp.contains:
                self.edge(parent, child, "contains")
        # inheritance first: method lookup walks base classes
        for fp in self.parses:
            for cls, raw in fp.bases:
                target = None
                if fp.language == "python":
                    module = self._module_of(cls)
                    aliases = self.aliases.get(module, {})
                    head, _, rest = raw.partition(".")
                    if head in aliases:
                        target = self.resolve_qualified(aliases[head] + (f".{rest}" if rest else ""))
                    target = target or self.resolve_qualified(f"{module}.{raw}") or self._unique_short(
                        raw.rsplit(".", 1)[-1], self.by_short
                    )
                else:
                    cands = self.cpp_by_suffix.get(raw.lstrip(":"), [])
                    target = cands[0] if cands else None
                if target and self.syms[target].kind == "class":
                    self.bases[cls].append(target)
                    self.edge(cls, target, "inherits")
        for fp in self.parses:
            resolve = self.resolve_py if fp.language == "python" else self.resolve_cpp
            for call in fp.calls:
                target = resolve(call)
                if target and target in self.syms:
                    idx.resolved_calls += 1
                    self.edge(call.scope, target, "call")
                else:
                    idx.unresolved_calls += 1
            for src, target in fp.imports:
                if fp.language == "python":
                    hit = self.resolve_qualified(target)
                    if hit is None:
                        # `import pkg.mod` / `from pkg import mod`
                        hit = target if target in self.syms else None
                    if hit:
                        self.edge(self._module_of(src), self._module_of(hit), "import")
                else:
                    name = Path(target).name
                    for rel in self.files_by_name.get(name, []):
                        if rel.endswith(target) or len(self.files_by_name[name]) == 1:
                            self.edge(src, rel, "import")
                            break
        # drop self-import noise
        idx.edges = [e for e in idx.edges if not (e.kind == "import" and e.src == e.dst)]


def index_repository(root: str | os.PathLike, progress: Callable[[int, int, str], None] | None = None) -> CodeIndex:
    return CodeIndexer().index(root, progress=progress)
