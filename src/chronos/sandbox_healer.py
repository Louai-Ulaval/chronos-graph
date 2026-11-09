"""Self-healing debugging agent.

State machine
    IDLE -> REPRODUCE -> LOCALIZE -> PROPOSE -> APPLY -> VERIFY -> HEALED
                                       ^                   |
                                       +---- ROLLBACK <----+          (budget spent -> FAILED)

* REPRODUCE  run the target tests in a throwaway copy of the repo and record failures.
* LOCALIZE   map traceback frames onto indexed symbols, then walk the call graph out of
             each failing test (C++ k-hop BFS) to rank suspect functions.
* PROPOSE    patch proposers yield candidate edits. The built-in proposer runs a ranked
             mutation search over suspect functions (comparison, arithmetic, boolean and
             off-by-one mutations). An optional Claude-backed proposer writes search /
             replace edits from the GraphRAG context and learns from failed attempts.
* APPLY      each candidate is written into its own sandbox worker; workers screen
             candidates in parallel against the failing tests only.
* VERIFY     the first screened candidate is re-run against the full target suite, which
             catches fixes that break something else.
* ROLLBACK   the sandbox file is restored before the next candidate.

Sandboxes are temporary copies of the repository. The subprocess backend runs pytest
in a new session with CPU / file-size rlimits and a hard wall-clock timeout; the docker
backend adds ``--network none`` plus memory and CPU caps.
"""

from __future__ import annotations

import ast
import difflib
import json
import os
import queue
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import uuid
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Callable, Iterable, Iterator

from .ast_indexer import CodeIndex, CodeIndexer
from .hybrid_retriever import HashingEmbedder, HybridRetriever


class HealState(str, Enum):
    IDLE = "IDLE"
    REPRODUCE = "REPRODUCE"
    LOCALIZE = "LOCALIZE"
    PROPOSE = "PROPOSE"
    APPLY = "APPLY"
    VERIFY = "VERIFY"
    ROLLBACK = "ROLLBACK"
    HEALED = "HEALED"
    FAILED = "FAILED"


class PatchError(ValueError):
    pass


# ---------------------------------------------------------------------------
# Patches
# ---------------------------------------------------------------------------


@dataclass
class Edit:
    file: str
    search: str
    replace: str

    def apply(self, text: str) -> str:
        count = text.count(self.search)
        if count == 0:
            raise PatchError(f"{self.file}: search text not found")
        if count > 1:
            raise PatchError(f"{self.file}: search text matches {count} places; add more context")
        return text.replace(self.search, self.replace, 1)


@dataclass
class Patch:
    edits: list[Edit]
    description: str
    source: str = "manual"  # mutation | claude | manual
    target: str = ""  # suspect symbol id
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])

    def files(self) -> list[str]:
        return sorted({e.file for e in self.edits})

    def apply_to(self, texts: dict[str, str]) -> dict[str, str]:
        out = dict(texts)
        for e in self.edits:
            if e.file not in out:
                raise PatchError(f"{e.file}: file not found")
            out[e.file] = e.apply(out[e.file])
        return out

    def diff(self, before: dict[str, str]) -> str:
        after = self.apply_to(before)
        chunks = []
        for f in self.files():
            chunks.extend(
                difflib.unified_diff(
                    before[f].splitlines(keepends=True), after[f].splitlines(keepends=True),
                    fromfile=f"a/{f}", tofile=f"b/{f}",
                )
            )
        return "".join(chunks)

    def to_dict(self) -> dict:
        return {
            "id": self.id, "description": self.description, "source": self.source, "target": self.target,
            "edits": [asdict(e) for e in self.edits],
        }


# ---------------------------------------------------------------------------
# Test runs
# ---------------------------------------------------------------------------


@dataclass
class Failure:
    test_id: str
    message: str
    error_type: str
    frames: list[tuple[str, int, str]]  # (file, line, function)
    text: str = ""


@dataclass
class TestRun:
    passed: bool
    returncode: int
    duration_ms: float
    counts: dict[str, int]
    failures: list[Failure]
    output: str
    timed_out: bool = False

    def summary(self) -> str:
        if self.timed_out:
            return f"timed out after {self.duration_ms / 1000:.1f}s"
        parts = [f"{v} {k}" for k, v in self.counts.items() if v]
        return ", ".join(parts) or ("passed" if self.passed else f"exit {self.returncode}")

    def to_dict(self, with_output: bool = False) -> dict:
        d = {
            "passed": self.passed, "returncode": self.returncode, "duration_ms": round(self.duration_ms, 1),
            "counts": self.counts, "timed_out": self.timed_out, "summary": self.summary(),
            "failures": [
                {"test_id": f.test_id, "message": f.message[:500], "error_type": f.error_type,
                 "frames": [list(fr) for fr in f.frames[-8:]]}
                for f in self.failures
            ],
        }
        if with_output:
            d["output"] = self.output[-8000:]
        return d


_SUMMARY_COUNT = re.compile(r"(\d+) (passed|failed|errors?|skipped|xfailed|xpassed|deselected)")
_FAILED_LINE = re.compile(r"^(FAILED|ERROR) (\S+)(?: - (.*))?$", re.M)
_FRAME = re.compile(r"^(\S+?\.py):(\d+): (?:in (\S+)|(\w+))", re.M)

COPY_IGNORE = shutil.ignore_patterns(
    ".git", ".hg", ".svn", ".venv", "venv", "node_modules", "__pycache__", ".pytest_cache",
    ".mypy_cache", ".ruff_cache", "build", "dist", "*.egg-info", ".chronos*",
)


def _limit_resources(cpu_seconds: int) -> Callable[[], None]:
    def apply() -> None:
        os.setsid()
        try:
            import resource

            resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds + 5))
            resource.setrlimit(resource.RLIMIT_FSIZE, (256 * 2**20, 256 * 2**20))
            if sys.platform.startswith("linux"):
                resource.setrlimit(resource.RLIMIT_AS, (4 * 2**30, 4 * 2**30))
        except (ImportError, ValueError, OSError):
            pass

    return apply


class Sandbox:
    """A disposable copy of a repository in which tests can be run safely."""

    DOCKER_IMAGE = "chronos-sandbox:py312"

    def __init__(self, repo: str | os.PathLike, backend: str = "auto", timeout: float = 60.0):
        self.repo = Path(repo).resolve()
        self.timeout = timeout
        self.backend = self._pick_backend(backend)
        self.dir = Path(tempfile.mkdtemp(prefix="chronos-sandbox-")).resolve()
        shutil.copytree(self.repo, self.dir, ignore=COPY_IGNORE, dirs_exist_ok=True, symlinks=True)

    @staticmethod
    def _pick_backend(backend: str) -> str:
        if backend == "auto":
            backend = os.environ.get("CHRONOS_SANDBOX", "subprocess")
        if backend not in ("subprocess", "docker"):
            raise ValueError(f"unknown sandbox backend: {backend}")
        if backend == "docker" and shutil.which("docker") is None:
            raise RuntimeError("docker backend requested but the docker CLI is not installed")
        return backend

    # ---- files ----------------------------------------------------------------
    def path(self, rel: str) -> Path:
        p = (self.dir / rel).resolve()
        if self.dir not in p.parents and p != self.dir:
            raise PatchError(f"path escapes the sandbox: {rel}")
        return p

    def read(self, rel: str) -> str:
        return self.path(rel).read_text(encoding="utf-8")

    def write(self, rel: str, text: str) -> None:
        self.path(rel).write_text(text, encoding="utf-8")

    def close(self) -> None:
        shutil.rmtree(self.dir, ignore_errors=True)

    def __enter__(self) -> "Sandbox":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ---- running ----------------------------------------------------------------
    def _pytest_args(self, targets: list[str], stop_first: bool) -> list[str]:
        args = ["-q", "-p", "no:cacheprovider", "--tb=long", "-rfE", "--color=no",
                "--junitxml=.chronos-junit.xml", "-o", "junit_family=xunit2"]
        if stop_first:
            args.append("-x")
        return args + targets

    def _env(self) -> dict[str, str]:
        keep = ("PATH", "HOME", "LANG", "LC_ALL", "TMPDIR", "SYSTEMROOT", "VIRTUAL_ENV")
        env = {k: v for k, v in os.environ.items() if k in keep}
        src = self.dir / "src"
        env["PYTHONPATH"] = os.pathsep.join([str(self.dir)] + ([str(src)] if src.is_dir() else []))
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        env["PYTHONHASHSEED"] = "0"
        env["CHRONOS_SANDBOXED"] = "1"
        return env

    _image_lock = threading.Lock()
    _image_ready = False

    def ensure_docker_image(self) -> None:
        """Build the sandbox image once (network is only used here, never while tests run)."""
        with Sandbox._image_lock:
            if Sandbox._image_ready:
                return
            probe = subprocess.run(["docker", "image", "inspect", self.DOCKER_IMAGE], capture_output=True)
            if probe.returncode != 0:
                dockerfile = "FROM python:3.12-slim\nRUN pip install --no-cache-dir pytest\n"
                build = subprocess.run(["docker", "build", "-t", self.DOCKER_IMAGE, "-"],
                                       input=dockerfile.encode(), capture_output=True)
                if build.returncode != 0:
                    tail = build.stderr.decode(errors="replace")[-2000:]
                    raise RuntimeError(f"could not build {self.DOCKER_IMAGE}:\n{tail}")
            Sandbox._image_ready = True

    def run_tests(self, targets: Iterable[str] | str | None = None, stop_first: bool = False) -> TestRun:
        if isinstance(targets, str):
            targets = [targets]
        targets = [t for t in (targets or []) if t]
        report = self.dir / ".chronos-junit.xml"
        report.unlink(missing_ok=True)
        container = None
        if self.backend == "docker":
            self.ensure_docker_image()
            container = f"chronos-{uuid.uuid4().hex[:12]}"
            user = ["--user", f"{os.getuid()}:{os.getgid()}"] if hasattr(os, "getuid") else []
            cmd = [
                "docker", "run", "--rm", "--name", container, "--network", "none", "--memory", "1g",
                "--cpus", "1", "--pids-limit", "256", "--read-only", "--tmpfs", "/tmp", *user,
                "-v", f"{self.dir}:/work", "-w", "/work", "-e", "HOME=/tmp",
                "-e", "PYTHONPATH=/work:/work/src", "-e", "PYTHONDONTWRITEBYTECODE=1",
                self.DOCKER_IMAGE, "python", "-m", "pytest", *self._pytest_args(targets, stop_first),
            ]
            preexec = None
            env = None
        else:
            cmd = [sys.executable, "-m", "pytest", *self._pytest_args(targets, stop_first)]
            preexec = _limit_resources(int(self.timeout) + 5) if os.name == "posix" else None
            env = self._env()
        t0 = time.perf_counter()
        proc = subprocess.Popen(cmd, cwd=self.dir, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                preexec_fn=preexec, text=True)
        timed_out = False
        try:
            out, _ = proc.communicate(timeout=self.timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            if container:
                # killing the docker CLI would leave the container running; kill the container itself
                subprocess.run(["docker", "kill", container], capture_output=True, timeout=30)
            try:
                if container is None:
                    os.killpg(proc.pid, signal.SIGKILL)  # pytest and all of its children
                else:
                    proc.kill()
            except (ProcessLookupError, PermissionError, AttributeError):
                proc.kill()
            out, _ = proc.communicate()
        duration = (time.perf_counter() - t0) * 1000
        return self._parse(out or "", proc.returncode if not timed_out else -9, duration, timed_out, report)

    def _parse(self, out: str, code: int, duration: float, timed_out: bool, report: Path) -> TestRun:
        counts: dict[str, int] = {}
        tail = out.strip().splitlines()[-1] if out.strip() else ""
        for n, k in _SUMMARY_COUNT.findall(tail):
            counts["errors" if k.startswith("error") else k] = int(n)
        texts: dict[str, tuple[str, str]] = {}
        if report.exists():
            try:
                for case in ET.parse(report).getroot().iter("testcase"):
                    for tag in ("failure", "error"):
                        el = case.find(tag)
                        if el is not None:
                            texts[case.get("name", "")] = (el.get("message", ""), el.text or "")
            except ET.ParseError:
                pass
        failures = []
        for kind, node_id, msg in _FAILED_LINE.findall(out):
            name = node_id.split("::")[-1]
            message, text = texts.get(name, (msg or "", ""))
            body = text or out
            frames = [(m[0], int(m[1]), m[2] or "") for m in _FRAME.findall(body)]
            err = re.search(r"^E\s+(\w+(?:Error|Exception|Exit))", body, re.M)
            failures.append(Failure(
                test_id=node_id, message=message or msg or "", frames=frames,
                error_type=err.group(1) if err else ("AssertionError" if "assert" in (message or "") else kind),
                text=body[-4000:],
            ))
        passed = code == 0 and not timed_out
        if code == 5:  # pytest: no tests collected
            passed = False
        return TestRun(passed, code, duration, counts, failures, out, timed_out)


# ---------------------------------------------------------------------------
# Localisation
# ---------------------------------------------------------------------------


def _is_test_file(rel: str) -> bool:
    p = Path(rel)
    return p.name.startswith("test_") or p.name.endswith("_test.py") or "tests" in p.parts or p.name == "conftest.py"


@dataclass
class Suspect:
    symbol: str
    file: str
    line: int
    end_line: int
    reason: str  # "traceback" | "call graph hop N"
    rank: float
    lines_of_interest: list[int] = field(default_factory=list)

    def to_dict(self) -> dict:
        return dict(self.__dict__)


def _test_symbol(index: CodeIndex, node_id: str) -> str | None:
    path, _, rest = node_id.partition("::")
    if not rest:
        return None
    rest = re.sub(r"\[.*\]$", "", rest)  # parametrised ids
    parts = list(Path(path).with_suffix("").parts)
    if parts and parts[0] in ("src", "lib"):
        parts = parts[1:]
    cand = ".".join(parts + rest.split("::"))
    return cand if cand in index.symbols else None


def localize(index: CodeIndex, retriever: HybridRetriever, failures: list[Failure], hops: int = 3) -> list[Suspect]:
    import chronos_cpp

    suspects: dict[str, Suspect] = {}
    by_file: dict[str, list] = {}
    for s in index.symbols.values():
        if s.kind in ("function", "method"):
            by_file.setdefault(s.file, []).append(s)

    # 1. traceback frames inside non-test source files (innermost frame ranks highest)
    for f in failures:
        for depth, (file, line, _fn) in enumerate(reversed(f.frames)):
            rel = file.removeprefix("./")
            if rel not in by_file or _is_test_file(rel):
                continue
            enclosing = [s for s in by_file[rel] if s.line <= line <= s.end_line]
            if not enclosing:
                continue
            s = min(enclosing, key=lambda s: s.end_line - s.line)
            sus = suspects.get(s.id)
            if sus is None:
                suspects[s.id] = Suspect(s.id, s.file, s.line, s.end_line, "traceback", 100.0 - depth, [line])
            elif line not in sus.lines_of_interest:
                sus.lines_of_interest.append(line)

    # 2. call graph walk out of each failing test
    test_syms = [t for t in (_test_symbol(index, f.test_id) for f in failures) if t]
    if test_syms:
        seeds = [index.id_to_idx[t] for t in test_syms]
        hop_map = index.engine.k_hop(seeds, hops, chronos_cpp.Direction.OUT)
        query = " ".join(index.symbols[t].source for t in test_syms) + " " + " ".join(f.message for f in failures)
        emb = HashingEmbedder()
        cands = []
        for node, dist in hop_map.items():
            if dist == 0:
                continue
            sid = index.ids[node]
            s = index.symbols[sid]
            if s.kind not in ("function", "method") or s.language != "python" or _is_test_file(s.file):
                continue
            cands.append((sid, dist))
        if cands:
            docs = [index.symbols[sid].source for sid, _ in cands]
            mat = emb.fit_transform(docs + [query])
            sims = mat[:-1] @ mat[-1]
            for (sid, dist), sim in zip(cands, sims):
                s = index.symbols[sid]
                rank = 50.0 / dist + 10.0 * float(sim)
                if sid in suspects:
                    suspects[sid].rank += rank
                else:
                    suspects[sid] = Suspect(sid, s.file, s.line, s.end_line, f"call graph hop {dist}", rank)
    return sorted(suspects.values(), key=lambda s: -s.rank)


# ---------------------------------------------------------------------------
# Proposers
# ---------------------------------------------------------------------------


@dataclass
class HealContext:
    repo: Path
    index: CodeIndex
    retriever: HybridRetriever
    baseline: TestRun
    suspects: list[Suspect]
    read: Callable[[str], str]
    history: list[dict] = field(default_factory=list)  # [{patch, screen: TestRun, verify: TestRun|None}]


_CMP_TEXT = {
    ast.Lt: "<", ast.LtE: "<=", ast.Gt: ">", ast.GtE: ">=", ast.Eq: "==", ast.NotEq: "!=",
    ast.Is: "is", ast.IsNot: "is not", ast.In: "in", ast.NotIn: "not in",
}
_CMP_SWAPS = {
    "<": ["<=", ">"], "<=": ["<", ">="], ">": [">=", "<"], ">=": [">", "<="],
    "==": ["!="], "!=": ["=="], "is": ["is not"], "is not": ["is"], "in": ["not in"], "not in": ["in"],
}
_BIN_TEXT = {ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/", ast.FloorDiv: "//", ast.Mod: "%"}
_BIN_SWAPS = {"+": ["-"], "-": ["+"], "*": ["/", "//"], "/": ["//", "*"], "//": ["/"], "%": ["//"]}
_AUG_SWAPS = {"+": ["-"], "-": ["+"], "*": ["/"], "/": ["*"]}


def _find_op(seg: str, op: str) -> re.Match | None:
    """Locate operator text in the gap between two operands (skipping comments)."""
    if op[0].isalpha():
        return re.search(r"\b" + r"\s+".join(op.split()) + r"\b", seg)
    return re.search(re.escape(op), seg)


class MutationProposer:
    """Ranked mutation search over suspect Python functions.

    Each mutation touches a single token span (an operator, a constant, a boolean
    connective), so every candidate is a minimal, reviewable search / replace edit.
    """

    name = "mutation"

    def __init__(self, max_candidates: int = 60, per_function: int = 24):
        self.max_candidates = max_candidates
        self.per_function = per_function

    def batches(self, ctx: HealContext, batch_size: int) -> Iterator[list[Patch]]:
        cands = list(self._candidates(ctx))[: self.max_candidates]
        for i in range(0, len(cands), batch_size):
            yield cands[i:i + batch_size]

    def _candidates(self, ctx: HealContext) -> Iterator[Patch]:
        seen: set[tuple[str, str, str]] = set()
        for sus in ctx.suspects:
            if not sus.file.endswith(".py"):
                continue
            try:
                text = ctx.read(sus.file)
            except OSError:
                continue
            muts = self._mutations(text, sus)
            for prio, (start, end, new, label, line) in sorted(muts, key=lambda m: m[0])[: self.per_function]:
                edit = self._make_edit(sus.file, text, start, end, new)
                if edit is None:
                    continue
                key = (edit.file, edit.search, edit.replace)
                if key in seen:
                    continue
                seen.add(key)
                yield Patch([edit], f"{sus.symbol.rsplit('.', 1)[-1]} line {line}: {label}", "mutation", sus.symbol)

    @staticmethod
    def _offsets(text: str) -> list[int]:
        offs = [0]
        for line in text.splitlines(keepends=True):
            offs.append(offs[-1] + len(line))
        return offs

    def _mutations(self, text: str, sus: Suspect) -> list[tuple[float, tuple]]:
        try:
            tree = ast.parse(text)
        except SyntaxError:
            return []
        fn = None
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.lineno == sus.line:
                fn = node
                break
        if fn is None:
            # decorated functions report the decorator line in the index
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and sus.line <= node.lineno <= sus.end_line:
                    fn = node
                    break
        if fn is None:
            return []
        lines = text.splitlines(keepends=True)
        offs = self._offsets(text)

        def pos(lineno: int, col: int) -> int:
            line_bytes = lines[lineno - 1].encode("utf-8")
            return offs[lineno - 1] + len(line_bytes[:col].decode("utf-8", errors="ignore"))

        out: list[tuple[float, tuple]] = []
        hot = set(sus.lines_of_interest)

        def add(base: float, start: int, end: int, new: str, label: str, lineno: int) -> None:
            prio = base - (5.0 if lineno in hot else 0.0)
            out.append((prio, (start, end, new, label, lineno)))

        def between(a: ast.AST, b: ast.AST) -> tuple[int, int]:
            return pos(a.end_lineno, a.end_col_offset), pos(b.lineno, b.col_offset)

        for node in ast.walk(fn):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)) and node is not fn:
                continue
            if isinstance(node, ast.Compare):
                left = node.left
                for op_node, comp in zip(node.ops, node.comparators):
                    op = _CMP_TEXT[type(op_node)]
                    s, e = between(left, comp)
                    m = _find_op(text[s:e], op)
                    if m:
                        for r in _CMP_SWAPS[op]:
                            # boundary flips (< vs <=) are the classic off-by-one, try them first
                            base = 1.0 if r in ("<=", ">=", "<", ">") and op[0] == r[0] else 2.0
                            add(base, s + m.start(), s + m.end(), r, f"`{op}` -> `{r}`", node.lineno)
                    left = comp
            elif isinstance(node, ast.BinOp) and type(node.op) in _BIN_TEXT:
                op = _BIN_TEXT[type(node.op)]
                s, e = between(node.left, node.right)
                m = _find_op(text[s:e], op)
                if m:
                    for r in _BIN_SWAPS[op]:
                        add(3.0, s + m.start(), s + m.end(), r, f"`{op}` -> `{r}`", node.lineno)
            elif isinstance(node, ast.AugAssign) and _BIN_TEXT.get(type(node.op)) in _AUG_SWAPS:
                op = _BIN_TEXT[type(node.op)] + "="
                s, e = between(node.target, node.value)
                m = _find_op(text[s:e], op)
                if m:
                    for r in _AUG_SWAPS[op[:-1]]:
                        add(3.0, s + m.start(), s + m.end(), r + "=", f"`{op}` -> `{r}=`", node.lineno)
            elif isinstance(node, ast.BoolOp):
                word = "and" if isinstance(node.op, ast.And) else "or"
                other = "or" if word == "and" else "and"
                for a, b in zip(node.values, node.values[1:]):
                    s, e = between(a, b)
                    m = re.search(rf"\b{word}\b", text[s:e])
                    if m:
                        add(3.5, s + m.start(), s + m.end(), other, f"`{word}` -> `{other}`", node.lineno)
            elif isinstance(node, ast.AugAssign):
                s, e = between(node.target, node.value)
                seg = text[s:e]
                for op, repls in _AUG_SWAPS.items():
                    if seg.strip() == op:
                        m = seg.index(op)
                        for r in repls:
                            add(3.0, s + m, s + m + len(op), r, f"`{op}` -> `{r}`", node.lineno)
            elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
                s = pos(node.lineno, node.col_offset)
                e = pos(node.operand.lineno, node.operand.col_offset)
                add(4.0, s, e, "", "drop `not`", node.lineno)
            elif isinstance(node, ast.Constant) and type(node.value) is int and abs(node.value) <= 10_000:
                s = pos(node.lineno, node.col_offset)
                e = pos(node.end_lineno, node.end_col_offset)
                if text[s:e].lstrip("-").isdigit():
                    for delta in (1, -1):
                        add(2.5, s, e, str(node.value + delta), f"`{node.value}` -> `{node.value + delta}`", node.lineno)
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in ("min", "max"):
                s = pos(node.func.lineno, node.func.col_offset)
                e = pos(node.func.end_lineno, node.func.end_col_offset)
                other = "max" if node.func.id == "min" else "min"
                add(3.0, s, e, other, f"`{node.func.id}` -> `{other}`", node.lineno)
        return out

    @staticmethod
    def _make_edit(file: str, text: str, start: int, end: int, new: str) -> Edit | None:
        if text[start:end] == new:
            return None
        line_start = text.rfind("\n", 0, start) + 1
        line_end = text.find("\n", end)
        line_end = len(text) if line_end < 0 else line_end
        # grow context upwards until the search text is unique in the file
        while True:
            search = text[line_start:line_end]
            if text.count(search) == 1 or line_start == 0:
                break
            line_start = text.rfind("\n", 0, max(line_start - 1, 0)) + 1
        replace = text[line_start:start] + new + text[end:line_end]
        if text.count(search) != 1:
            return None
        return Edit(file, search, replace)


PATCH_SCHEMA = {
    "type": "object",
    "properties": {
        "analysis": {"type": "string"},
        "edits": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "file": {"type": "string"},
                    "search": {"type": "string"},
                    "replace": {"type": "string"},
                },
                "required": ["file", "search", "replace"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["analysis", "edits"],
    "additionalProperties": False,
}

_CLAUDE_SYSTEM = """You repair failing Python test suites by editing source code.

You get the failing test output and a GraphRAG context: the source of the symbols that
the call graph connects to the failing tests. Find the root cause and return the
smallest change to the code under test that makes the tests pass.

Rules:
- Edit source files only. Never edit test files, conftest.py, or anything under tests/.
- Each edit is an exact search/replace: `search` must be copied verbatim from the file and
  match exactly one place in it (include a neighbouring line when a line repeats).
- Keep public signatures and behaviour the passing tests rely on.
- File paths are relative to the repository root, as they appear in the context headers."""


class ClaudeProposer:
    """Optional LLM proposer backed by the Claude API (enable with ANTHROPIC_API_KEY or CHRONOS_LLM=1)."""

    name = "claude"

    def __init__(self, model: str | None = None, rounds: int = 3, effort: str = "high", token_budget: int = 6000):
        self.model = model or os.environ.get("CHRONOS_MODEL", "claude-opus-5-5")
        self.rounds = rounds
        self.effort = effort
        self.token_budget = token_budget
        self.last_analysis = ""
        self.last_error = ""
        self.reserve = rounds  # attempts the healer keeps back for this proposer

    @staticmethod
    def available() -> bool:
        if os.environ.get("CHRONOS_LLM") == "0":
            return False
        try:
            import anthropic  # noqa: F401
        except ImportError:
            return False
        return bool(
            os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN") or os.environ.get("CHRONOS_LLM") == "1"
        )

    def _prompt(self, ctx: HealContext) -> str:
        fails = ctx.baseline.failures
        query = " ".join(f"{f.test_id} {f.error_type} {f.message}" for f in fails)[:2000]
        retrieved = ctx.retriever.retrieve(query, token_budget=self.token_budget, hops=2)
        test_src = []
        for f in fails[:4]:
            sid = _test_symbol(ctx.index, f.test_id)
            if sid:
                test_src.append(f"### {f.test_id}\n```python\n{ctx.index.symbols[sid].source}\n```")
        suspects = "\n".join(f"- {s.symbol} ({s.file}:{s.line}, {s.reason})" for s in ctx.suspects[:12])
        output = ctx.baseline.output[-6000:]
        return (
            f"## Failing tests\n{chr(10).join(test_src) or '(test source not indexed)'}\n\n"
            f"## pytest output (tail)\n```\n{output}\n```\n\n"
            f"## Suspects ranked by traceback and call graph\n{suspects or '(none)'}\n\n"
            f"{retrieved.context}"
        )

    def batches(self, ctx: HealContext, batch_size: int) -> Iterator[list[Patch]]:
        import anthropic

        client = anthropic.Anthropic()
        messages: list[dict] = [{"role": "user", "content": self._prompt(ctx)}]
        for _round in range(self.rounds):
            try:
                response = self._create(client, messages)
            except Exception as exc:  # auth, network, rate limits: report and step aside
                self.last_error = f"{type(exc).__name__}: {exc}"[:300]
                return
            if response.stop_reason == "refusal":
                self.last_error = "the model declined this request"
                return
            text = next((b.text for b in response.content if b.type == "text"), "")
            try:
                data = json.loads(text)
            except json.JSONDecodeError:
                return
            self.last_analysis = data.get("analysis", "")
            edits = [
                Edit(e["file"], e["search"], e["replace"]) for e in data.get("edits", [])
                if e.get("file") and not _is_test_file(e["file"])
            ]
            if not edits:
                return
            patch = Patch(edits, self.last_analysis[:160] or "model patch", "claude")
            yield [patch]
            # feed the outcome back for the next round (append-only history)
            outcome = next((h for h in reversed(ctx.history) if h["patch"].id == patch.id), None)
            if outcome is None:
                return
            run: TestRun | None = outcome.get("verify") or outcome.get("screen")
            detail = outcome.get("error") or (run.output[-4000:] if run else "no test output")
            messages.append({"role": "assistant", "content": response.content})
            messages.append({
                "role": "user",
                "content": f"That patch did not fix the suite.\n```\n{detail}\n```\nPropose a different fix.",
            })

    def _create(self, client, messages: list[dict]):
        common = dict(
            model=self.model,
            max_tokens=16000,
            system=_CLAUDE_SYSTEM,
            messages=messages,
            output_config={"effort": self.effort, "format": {"type": "json_schema", "schema": PATCH_SCHEMA}},
        )
        import inspect

        params = inspect.signature(client.beta.messages.create).parameters
        if "fallbacks" in params or any(p.kind is p.VAR_KEYWORD for p in params.values()):
            # server-side fallback reroutes a policy decline to another model in the same call
            return client.beta.messages.create(betas=["server-side-fallback-2026-07-01"], fallbacks="default", **common)
        return client.messages.create(**common)  # SDK predates the fallbacks parameter


# ---------------------------------------------------------------------------
# Healer
# ---------------------------------------------------------------------------


@dataclass
class HealResult:
    run_id: str
    success: bool
    state: str
    attempts: int
    duration_ms: float
    patch: dict | None
    diff: str
    baseline: dict
    final: dict | None
    suspects: list[dict]
    trace: list[dict]
    applied: bool = False
    message: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


class SelfHealer:
    def __init__(
        self,
        repo: str | os.PathLike,
        test_target: str | list[str] | None = "tests",
        index: CodeIndex | None = None,
        backend: str = "auto",
        workers: int = 4,
        max_attempts: int = 40,
        timeout: float = 60.0,
        proposers: list | None = None,
        use_llm: bool | None = None,
        on_event: Callable[[dict], None] | None = None,
    ):
        self.repo = Path(repo).resolve()
        targets = [test_target] if isinstance(test_target, str) else list(test_target or [])
        self.targets = targets
        self.index = index
        self.backend = backend
        self.workers = max(1, workers)
        self.max_attempts = max_attempts
        self.timeout = timeout
        self.on_event = on_event
        if proposers is None:
            proposers = [MutationProposer()]
            if use_llm or (use_llm is None and ClaudeProposer.available()):
                proposers.append(ClaudeProposer())
        self.proposers = proposers
        self.run_id = uuid.uuid4().hex[:10]
        self.state = HealState.IDLE
        self.trace: list[dict] = []
        self._lock = threading.Lock()

    def _emit(self, state: HealState, message: str, **extra) -> None:
        with self._lock:
            self.state = state
            event = {"type": "heal", "run_id": self.run_id, "state": state.value, "message": message,
                     "ts": time.time(), **extra}
            self.trace.append(event)
        if self.on_event:
            try:
                self.on_event(event)
            except Exception:  # telemetry must never break the healer
                pass

    def heal(self, apply: bool = False) -> HealResult:
        t0 = time.perf_counter()
        pool: queue.Queue[Sandbox] = queue.Queue()
        sandboxes: list[Sandbox] = []
        attempts = 0
        try:
            self._emit(HealState.REPRODUCE, f"Running {' '.join(self.targets) or 'all tests'} in a sandbox")
            primary = Sandbox(self.repo, self.backend, self.timeout)
            sandboxes.append(primary)
            baseline = primary.run_tests(self.targets)
            self._emit(HealState.REPRODUCE, f"Baseline: {baseline.summary()}", test=baseline.to_dict())
            if baseline.passed:
                return self._finish(t0, True, HealState.HEALED, attempts, None, "", baseline, baseline, [],
                                    "All tests already pass; nothing to heal.")
            if not baseline.failures:
                return self._finish(t0, False, HealState.FAILED, attempts, None, "", baseline, None, [],
                                    "Tests fail without reporting a test failure (collection or config error).")

            self._emit(HealState.LOCALIZE, "Mapping failures onto the code graph")
            index = self.index or CodeIndexer().index(self.repo)
            retriever = HybridRetriever(index)
            suspects = localize(index, retriever, baseline.failures)
            self._emit(HealState.LOCALIZE, f"{len(suspects)} suspect symbols",
                       suspects=[s.to_dict() for s in suspects[:12]])
            if not suspects:
                return self._finish(t0, False, HealState.FAILED, attempts, None, "", baseline, None, [],
                                    "Could not localise the failure to any indexed source symbol.")

            for _ in range(self.workers - 1):
                sandboxes.append(Sandbox(self.repo, self.backend, self.timeout))
            for sb in sandboxes:
                pool.put(sb)
            ctx = HealContext(self.repo, index, retriever, baseline, suspects, read=primary.read)
            failing_ids = [f.test_id for f in baseline.failures]

            for i, proposer in enumerate(self.proposers):
                # keep attempts back for later proposers so a long mutation search can't starve them
                held_back = sum(getattr(p, "reserve", 0) for p in self.proposers[i + 1:])
                limit = self.max_attempts - held_back
                if attempts >= self.max_attempts:
                    break
                self._emit(HealState.PROPOSE, f"Proposer: {proposer.name}")
                for batch in proposer.batches(ctx, self.workers):
                    if attempts >= limit:
                        break
                    batch = batch[: limit - attempts]
                    for p in batch:
                        attempts += 1
                        self._emit(HealState.APPLY, p.description, attempt=attempts, patch=p.to_dict())
                    with ThreadPoolExecutor(max_workers=self.workers) as ex:
                        results = list(ex.map(lambda p: self._screen(pool, p, failing_ids), batch))
                    for patch, screen, error in results:
                        record = {"patch": patch, "screen": screen, "error": error}
                        ctx.history.append(record)
                        if error or screen is None or not screen.passed:
                            why = error or (screen.summary() if screen else "no result")
                            self._emit(HealState.ROLLBACK, f"{patch.description}: {why}", patch_id=patch.id)
                            continue
                        self._emit(HealState.VERIFY, f"{patch.description}: failing tests pass, running full suite",
                                   patch_id=patch.id)
                        full = self._run_with(pool, patch, self.targets)
                        record["verify"] = full
                        if full.passed:
                            before = {f: primary.read(f) for f in patch.files()}
                            diff = patch.diff(before)
                            applied = False
                            if apply:
                                self._apply_to_workspace(patch)
                                applied = True
                            self._emit(HealState.HEALED, f"Healed with {patch.source} patch: {patch.description}",
                                       patch=patch.to_dict(), diff=diff, test=full.to_dict())
                            return self._finish(t0, True, HealState.HEALED, attempts, patch, diff, baseline, full,
                                                suspects, "Healed.", applied)
                        self._emit(HealState.ROLLBACK, f"{patch.description}: regression ({full.summary()})",
                                   patch_id=patch.id, test=full.to_dict())
                if getattr(proposer, "last_error", ""):
                    self._emit(HealState.PROPOSE, f"{proposer.name} stopped: {proposer.last_error}")
            return self._finish(t0, False, HealState.FAILED, attempts, None, "", baseline, None, suspects,
                                f"No patch fixed the suite within {attempts} attempts.")
        finally:
            for sb in sandboxes:
                sb.close()

    def _screen(self, pool: queue.Queue, patch: Patch, failing_ids: list[str]):
        try:
            return patch, self._run_with(pool, patch, failing_ids, stop_first=True), None
        except PatchError as exc:
            return patch, None, str(exc)

    def _run_with(self, pool: queue.Queue, patch: Patch, targets: list[str], stop_first: bool = False) -> TestRun:
        sb = pool.get()
        originals: dict[str, str] = {}
        try:
            for f in patch.files():
                originals[f] = sb.read(f)
            patched = patch.apply_to(originals)
            for f, text in patched.items():
                sb.write(f, text)
            return sb.run_tests(targets, stop_first=stop_first)
        finally:
            for f, text in originals.items():  # rollback
                sb.write(f, text)
            pool.put(sb)

    def _apply_to_workspace(self, patch: Patch) -> None:
        for f in patch.files():
            path = self.repo / f
            text = path.read_text(encoding="utf-8")
            new = text
            for e in patch.edits:
                if e.file == f:
                    new = e.apply(new)
            path.write_text(new, encoding="utf-8")

    def _finish(self, t0, success, state, attempts, patch, diff, baseline, final, suspects, message, applied=False):
        if state in (HealState.FAILED,) or (state == HealState.HEALED and patch is None):
            self._emit(state, message)
        return HealResult(
            run_id=self.run_id, success=success, state=state.value, attempts=attempts,
            duration_ms=round((time.perf_counter() - t0) * 1000, 1), patch=patch.to_dict() if patch else None,
            diff=diff, baseline=baseline.to_dict(), final=final.to_dict() if final else None,
            suspects=[s.to_dict() for s in suspects[:12]], trace=list(self.trace), applied=applied, message=message,
        )


# ---------------------------------------------------------------------------
# Convenience entry points (used by the MCP and API servers)
# ---------------------------------------------------------------------------


def run_tests_in_sandbox(repo: str | os.PathLike, target: str | None = None, backend: str = "auto",
                         timeout: float = 120.0) -> TestRun:
    with Sandbox(repo, backend, timeout) as sb:
        return sb.run_tests([target] if target else [])


def try_patch_in_sandbox(repo: str | os.PathLike, edits: list[dict], target: str | None = None,
                         backend: str = "auto", timeout: float = 120.0) -> dict:
    patch = Patch([Edit(e["file"], e["search"], e["replace"]) for e in edits], "manual patch", "manual")
    with Sandbox(repo, backend, timeout) as sb:
        before = {f: sb.read(f) for f in patch.files()}
        after = patch.apply_to(before)
        for f, text in after.items():
            sb.write(f, text)
        run = sb.run_tests([target] if target else [])
        return {"diff": patch.diff(before), "test": run.to_dict(with_output=True)}


DEMO_PROJECT: dict[str, str] = {
    "README.md": "# demo-inventory\n\nA tiny cart service with a planted off-by-one bug for the self-healing demo.\n",
    "inventory/__init__.py": '"""Demo inventory service."""\n',
    "inventory/pricing.py": '''"""Pricing rules for the demo inventory service."""


def bulk_discount_percent(quantity: int) -> int:
    """Volume discount tiers: 100+ units get 15%, 50+ units get 10%, 10+ units get 5%."""
    if quantity >= 100:
        return 15
    if quantity > 50:
        return 10
    if quantity >= 10:
        return 5
    return 0


def apply_discount(amount: float, percent: int) -> float:
    """Return ``amount`` reduced by ``percent`` percent, rounded to cents."""
    return round(amount * (100 - percent) / 100, 2)


def sales_tax(amount: float, rate: float = 0.15) -> float:
    """Sales tax on an amount, rounded to cents."""
    return round(amount * rate, 2)
''',
    "inventory/cart.py": '''"""Shopping cart that applies volume discounts and tax."""

from .pricing import apply_discount, bulk_discount_percent, sales_tax


class Cart:
    def __init__(self) -> None:
        self.lines: list[tuple[str, float, int]] = []

    def add(self, sku: str, unit_price: float, quantity: int = 1) -> None:
        if quantity <= 0:
            raise ValueError("quantity must be positive")
        self.lines.append((sku, unit_price, quantity))

    def units(self) -> int:
        return sum(q for _, _, q in self.lines)

    def subtotal(self) -> float:
        return round(sum(price * q for _, price, q in self.lines), 2)

    def total(self, with_tax: bool = False) -> float:
        """Subtotal after the volume discount, optionally with sales tax."""
        discounted = apply_discount(self.subtotal(), bulk_discount_percent(self.units()))
        if with_tax:
            return round(discounted + sales_tax(discounted), 2)
        return discounted
''',
    "tests/test_cart.py": '''import pytest

from inventory.cart import Cart
from inventory.pricing import bulk_discount_percent


def test_small_order_has_no_discount():
    cart = Cart()
    cart.add("mug", 10.0, 3)
    assert cart.total() == 30.0


def test_ten_units_get_five_percent():
    cart = Cart()
    cart.add("mug", 10.0, 10)
    assert cart.total() == 95.0


def test_fifty_units_get_ten_percent():
    cart = Cart()
    cart.add("mug", 10.0, 30)
    cart.add("plate", 10.0, 20)
    assert cart.total() == 450.0


def test_hundred_units_get_fifteen_percent():
    cart = Cart()
    cart.add("mug", 10.0, 100)
    assert cart.total() == 850.0


def test_tax_is_charged_on_the_discounted_total():
    cart = Cart()
    cart.add("bowl", 5.0, 20)
    assert cart.total(with_tax=True) == 109.25


def test_rejects_non_positive_quantity():
    with pytest.raises(ValueError):
        Cart().add("mug", 10.0, 0)


def test_tiers_are_monotonic():
    tiers = [bulk_discount_percent(q) for q in (1, 9, 10, 49, 50, 99, 100)]
    assert tiers == [0, 0, 5, 5, 10, 10, 15]
''',
}


# A bug no single-token mutation can fix (the data is never sorted), so the
# mutation search exhausts and the Claude proposer has to write the patch.
MISSING_SORT_PROJECT: dict[str, str] = {
    "README.md": "# demo-stats\n\nSummary statistics with a planted bug for the LLM proposer demo.\n",
    "stats/__init__.py": '"""Demo statistics package."""\n',
    "stats/summary.py": '''"""Summary statistics."""


def mean(values):
    """Arithmetic mean of a non-empty list."""
    return sum(values) / len(values)


def median(values):
    """Middle value of the data (mean of the two middle values when the length is even)."""
    n = len(values)
    if n == 0:
        raise ValueError("median of empty data")
    mid = n // 2
    if n % 2:
        return values[mid]
    return (values[mid - 1] + values[mid]) / 2


def spread(values):
    """Distance between the median and the mean."""
    return abs(median(values) - mean(values))
''',
    "tests/test_summary.py": '''import pytest

from stats.summary import mean, median, spread


def test_mean():
    assert mean([1, 2, 3, 6]) == 3


def test_median_of_sorted_odd():
    assert median([1, 2, 9]) == 2


def test_median_of_unsorted_odd():
    assert median([9, 1, 2]) == 2


def test_median_of_unsorted_even():
    assert median([4, 1, 3, 2]) == 2.5


def test_median_rejects_empty():
    with pytest.raises(ValueError):
        median([])


def test_spread():
    assert spread([3, 1, 2, 10]) == 1.5
''',
}

DEMO_PROJECTS = {"off-by-one": DEMO_PROJECT, "missing-sort": MISSING_SORT_PROJECT}


def create_demo_project(path: str | os.PathLike, name: str = "off-by-one") -> Path:
    """Write a demo project with a planted bug to ``path`` and return it.

    ``off-by-one`` is fixed by the mutation search; ``missing-sort`` needs the Claude proposer."""
    root = Path(path)
    for rel, text in DEMO_PROJECTS[name].items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
    return root
