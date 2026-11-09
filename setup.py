"""Build the chronos_cpp extension. Project metadata lives in pyproject.toml."""

import os
import platform
import subprocess
import sys

from pybind11.setup_helpers import Pybind11Extension
from pybind11.setup_helpers import build_ext as _build_ext
from setuptools import setup

extra_compile_args = ["-O3", "-fno-math-errno"]
if sys.platform == "win32":
    extra_compile_args = ["/O2", "/std:c++20"]
elif os.environ.get("CHRONOS_NATIVE") == "1":
    # Opt-in: tune for the build machine (Apple Silicon: -mcpu=native).
    extra_compile_args.append(
        "-mcpu=native" if platform.machine() in ("arm64", "aarch64") else "-march=native"
    )


def _macos_sdk() -> str | None:
    try:
        return subprocess.check_output(["xcrun", "--show-sdk-path"], text=True).strip() or None
    except (OSError, subprocess.CalledProcessError):
        return None


class build_ext(_build_ext):
    """Repairs a stale -isysroot baked into some Python builds (pyenv, old CLT installs).

    When the interpreter was compiled against an SDK path that no longer exists, the
    linker cannot find libc++. Swap any missing sysroot for the active one from xcrun.
    """

    def build_extensions(self):
        if sys.platform == "darwin":
            sdk = _macos_sdk()
            for attr in list(getattr(self.compiler, "executables", {})):
                cmd = getattr(self.compiler, attr, None)
                if not isinstance(cmd, list):
                    continue
                fixed = []
                skip = False
                for i, arg in enumerate(cmd):
                    if skip:
                        skip = False
                        continue
                    if arg == "-isysroot" and i + 1 < len(cmd) and not os.path.isdir(cmd[i + 1]):
                        skip = True
                        if sdk:
                            fixed += ["-isysroot", sdk]
                        continue
                    fixed.append(arg)
                self.compiler.set_executable(attr, fixed)
        super().build_extensions()


ext_modules = [
    Pybind11Extension(
        "chronos_cpp",
        sources=["src/cpp/graph_engine.cpp", "src/cpp/bindings.cpp"],
        include_dirs=["src/cpp"],
        depends=["src/cpp/graph_engine.hpp"],  # rebuild when only the header changes
        cxx_std=20,
        extra_compile_args=extra_compile_args,
        define_macros=[("NDEBUG", "1")],
    )
]

setup(ext_modules=ext_modules, cmdclass={"build_ext": build_ext})
