"""AgentJIT FASM Hardware Assembly Acceleration Backend.

Direct bare-metal AVX2+FMA (x86-64) and SSE2 (x86 32-bit) SIMD acceleration
for speculative guards, trajectory hashing, and semantic router evaluation.
Copyright (c) 2026 eminsk (M_N_Nik@yahoo.com)
Apache-2.0 License
"""

from __future__ import annotations

import ctypes
import math
import os
import sys
from pathlib import Path
from typing import Any, List, Optional, Sequence, Union

_FASM_LIB: Optional[ctypes.CDLL] = None
_FASM_ISA: str = "Pure Python (Hardware FASM Engine Unavailable)"


def _bind_fasm_library(lib: ctypes.CDLL) -> ctypes.CDLL:
    """Bind CTypes signatures to the loaded FASM shared library."""
    if hasattr(lib, "agentjit_version"):
        lib.agentjit_version.argtypes = []
        lib.agentjit_version.restype = ctypes.c_int

    if hasattr(lib, "agentjit_simd_isa"):
        lib.agentjit_simd_isa.argtypes = []
        lib.agentjit_simd_isa.restype = ctypes.c_char_p

    if hasattr(lib, "agentjit_fast_hash"):
        lib.agentjit_fast_hash.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint64]
        lib.agentjit_fast_hash.restype = ctypes.c_uint64

    if hasattr(lib, "agentjit_eval_null_guards"):
        lib.agentjit_eval_null_guards.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        lib.agentjit_eval_null_guards.restype = ctypes.c_int

    if hasattr(lib, "agentjit_eval_range_guards_f64"):
        lib.agentjit_eval_range_guards_f64.argtypes = [
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_size_t,
        ]
        lib.agentjit_eval_range_guards_f64.restype = ctypes.c_int

    if hasattr(lib, "agentjit_vector_dot"):
        lib.agentjit_vector_dot.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_size_t,
        ]
        lib.agentjit_vector_dot.restype = ctypes.c_float

    if hasattr(lib, "agentjit_cosine_similarity"):
        lib.agentjit_cosine_similarity.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_size_t,
        ]
        lib.agentjit_cosine_similarity.restype = ctypes.c_float

    if hasattr(lib, "agentjit_semantic_guard_cosine"):
        lib.agentjit_semantic_guard_cosine.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_size_t,
            ctypes.c_float,
        ]
        lib.agentjit_semantic_guard_cosine.restype = ctypes.c_int

    if hasattr(lib, "agentjit_batch_arithmetic_f64"):
        lib.agentjit_batch_arithmetic_f64.argtypes = [
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_size_t,
            ctypes.c_int,
        ]
        lib.agentjit_batch_arithmetic_f64.restype = ctypes.c_int

    if hasattr(lib, "agentjit_batch_lookup_hash"):
        lib.agentjit_batch_lookup_hash.argtypes = [
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.c_size_t,
            ctypes.c_uint64,
        ]
        lib.agentjit_batch_lookup_hash.restype = ctypes.c_int64

    return lib


def _find_and_load_fasm_lib() -> Optional[ctypes.CDLL]:
    """Locates and loads the native hardware SIMD/FASM library (Windows/Linux/macOS)."""
    global _FASM_ISA

    env_path = os.environ.get("AGENTJIT_SIMD_LIB") or os.environ.get("AGENTJIT_FASM_LIB")
    if env_path and Path(env_path).exists():
        try:
            lib = ctypes.CDLL(env_path)
            bound = _bind_fasm_library(lib)
            if hasattr(bound, "agentjit_simd_isa"):
                _FASM_ISA = bound.agentjit_simd_isa().decode("utf-8", errors="replace")
            return bound
        except Exception:
            pass

    is_64bit = sys.maxsize > 2**32
    if sys.platform.startswith("win"):
        target_names = ["agentjit64.dll"] if is_64bit else ["agentjit32.dll"]
    elif sys.platform.startswith("darwin"):
        target_names = ["libagentjit.dylib", "libagentjit64.dylib"]
    else:
        # Linux / POSIX
        target_names = (
            ["libagentjit64.so", "agentjit64.so", "libagentjit.so"]
            if is_64bit
            else ["libagentjit32.so", "agentjit32.so", "libagentjit.so"]
        )

    pkg_dir = Path(__file__).resolve().parent
    search_dirs = [
        pkg_dir,
        pkg_dir.parent,
        pkg_dir.parent / "asm",
        pkg_dir.parent.parent / "asm",
        Path(r"C:\proekts\agentjit\asm"),
        Path(r"C:\proekts\agentjit\src\agentjit"),
        Path("/usr/local/lib"),
        Path("/tmp"),
        Path.cwd(),
    ]

    for d in search_dirs:
        for name in target_names:
            cand = d / name
            if cand.exists():
                try:
                    lib = ctypes.CDLL(str(cand))
                    bound_lib = _bind_fasm_library(lib)
                    if hasattr(bound_lib, "agentjit_simd_isa"):
                        _FASM_ISA = bound_lib.agentjit_simd_isa().decode("utf-8", errors="replace")
                    return bound_lib
                except Exception:
                    continue

    # On-demand compilation on Linux/macOS if gcc/clang and kernel source are available
    if not sys.platform.startswith("win"):
        try:
            import shutil
            import subprocess

            cc = shutil.which("gcc") or shutil.which("clang")
            if cc:
                c_candidates = [
                    pkg_dir / "agentjit_kernel.c",
                    pkg_dir.parent / "asm" / "agentjit_kernel.c",
                    pkg_dir.parent.parent / "asm" / "agentjit_kernel.c",
                    Path(r"/content/agentjit/asm/agentjit_kernel.c"),
                ]
                src_c = next((c for c in c_candidates if c.exists()), None)
                if src_c:
                    out_so = Path("/tmp") / target_names[0]
                    cmd = [cc, "-O3", "-shared", "-fPIC", "-mavx2", "-mfma", "-ffast-math", str(src_c), "-o", str(out_so), "-lm"]
                    res = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    if res.returncode == 0 and out_so.exists():
                        lib = ctypes.CDLL(str(out_so))
                        bound_lib = _bind_fasm_library(lib)
                        if hasattr(bound_lib, "agentjit_simd_isa"):
                            _FASM_ISA = bound_lib.agentjit_simd_isa().decode("utf-8", errors="replace")
                        return bound_lib
        except Exception:
            pass

    return None


def get_fasm_library() -> Optional[ctypes.CDLL]:
    """Return the cached FASM shared library instance, or None if unavailable."""
    global _FASM_LIB
    if _FASM_LIB is None:
        _FASM_LIB = _find_and_load_fasm_lib()
    return _FASM_LIB


def is_fasm_available() -> bool:
    """Return True if hardware FASM microkernel engine is available and loaded."""
    return get_fasm_library() is not None


def simd_backend() -> str:
    """Return the active hardware acceleration backend string."""
    get_fasm_library()
    return _FASM_ISA


class FASMHardwareEngine:
    """High-level hardware-accelerated guard, routing, and arithmetic execution engine."""

    def __init__(self) -> None:
        self.lib = get_fasm_library()

    @property
    def is_available(self) -> bool:
        return self.lib is not None

    @property
    def simd_backend(self) -> str:
        return simd_backend()

    def fast_hash(self, data: Union[str, bytes], seed: int = 0) -> int:
        """Compute an ultra-fast 64-bit unrolled hash for a string or bytes buffer."""
        raw = data.encode("utf-8") if isinstance(data, str) else bytes(data)
        if self.lib is not None:
            buf = (ctypes.c_char * len(raw)).from_buffer_copy(raw)
            return int(self.lib.agentjit_fast_hash(buf, len(raw), seed))

        # Pure-Python fallback (FNV-1a 64-bit)
        h = 0xCBF29CE484222325 if seed == 0 else seed
        prime = 0x100000001B3
        for b in raw:
            h = (h ^ b) * prime & 0xFFFFFFFFFFFFFFFF
        return h

    def eval_null_guards(self, pointers: Sequence[Any]) -> bool:
        """Vectorized evaluation of non-null parameters. Returns True if all are non-null."""
        if not pointers:
            return True

        if self.lib is not None:
            c_ptrs = (ctypes.c_void_p * len(pointers))()
            for idx, item in enumerate(pointers):
                c_ptrs[idx] = 1 if item is not None else 0
            return bool(self.lib.agentjit_eval_null_guards(c_ptrs, len(pointers)))

        return all(p is not None for p in pointers)

    def eval_range_guards(
        self,
        values: Sequence[float],
        mins: Sequence[float],
        maxs: Sequence[float],
    ) -> bool:
        """Vectorized parallel check that mins[i] <= values[i] <= maxs[i]."""
        n = min(len(values), len(mins), len(maxs))
        if n == 0:
            return True

        if self.lib is not None:
            c_vals = (ctypes.c_double * n)(*values[:n])
            c_mins = (ctypes.c_double * n)(*mins[:n])
            c_maxs = (ctypes.c_double * n)(*maxs[:n])
            return bool(self.lib.agentjit_eval_range_guards_f64(c_vals, c_mins, c_maxs, n))

        for i in range(n):
            if not (mins[i] <= values[i] <= maxs[i]):
                return False
        return True

    def vector_dot(self, a: Sequence[float], b: Sequence[float]) -> float:
        """Unrolled hardware SIMD dot product of two float arrays."""
        dim = min(len(a), len(b))
        if dim == 0:
            return 0.0

        if self.lib is not None:
            c_a = (ctypes.c_float * dim)(*a[:dim])
            c_b = (ctypes.c_float * dim)(*b[:dim])
            return float(self.lib.agentjit_vector_dot(c_a, c_b, dim))

        return float(sum(x * y for x, y in zip(a[:dim], b[:dim])))

    def cosine_similarity(self, a: Sequence[float], b: Sequence[float]) -> float:
        """Compute normalized cosine similarity between two float vectors."""
        dim = min(len(a), len(b))
        if dim == 0:
            return 0.0

        if self.lib is not None:
            c_a = (ctypes.c_float * dim)(*a[:dim])
            c_b = (ctypes.c_float * dim)(*b[:dim])
            return float(self.lib.agentjit_cosine_similarity(c_a, c_b, dim))

        dot = sum(x * y for x, y in zip(a[:dim], b[:dim]))
        na = sum(x * x for x in a[:dim])
        nb = sum(y * y for y in b[:dim])
        denom = math.sqrt(na * nb)
        return float(dot / denom) if denom > 1e-12 else 0.0

    def eval_semantic_guard(
        self,
        prompt_vec: Sequence[float],
        trajectory_vec: Sequence[float],
        threshold: float = 0.85,
    ) -> bool:
        """Check if incoming prompt embedding matches trajectory embedding within threshold."""
        dim = min(len(prompt_vec), len(trajectory_vec))
        if dim == 0:
            return False

        if self.lib is not None:
            c_a = (ctypes.c_float * dim)(*prompt_vec[:dim])
            c_b = (ctypes.c_float * dim)(*trajectory_vec[:dim])
            return bool(
                self.lib.agentjit_semantic_guard_cosine(
                    c_a, c_b, dim, ctypes.c_float(threshold)
                )
            )

        sim = self.cosine_similarity(prompt_vec, trajectory_vec)
        return sim >= threshold

    def batch_arithmetic(
        self,
        a: Sequence[float],
        b: Sequence[float],
        op: str = "add",
    ) -> List[float]:
        """Hardware SIMD vectorized arithmetic for synthesized data-flow nodes."""
        n = min(len(a), len(b))
        if n == 0:
            return []

        op_map = {"add": 0, "sub": 1, "mul": 2, "div": 3}
        op_code = op_map.get(op.lower(), 0)

        if self.lib is not None:
            c_a = (ctypes.c_double * n)(*a[:n])
            c_b = (ctypes.c_double * n)(*b[:n])
            c_out = (ctypes.c_double * n)()
            self.lib.agentjit_batch_arithmetic_f64(c_a, c_b, c_out, n, op_code)
            return list(c_out)

        if op_code == 0:
            return [x + y for x, y in zip(a[:n], b[:n])]
        elif op_code == 1:
            return [x - y for x, y in zip(a[:n], b[:n])]
        elif op_code == 2:
            return [x * y for x, y in zip(a[:n], b[:n])]
        elif op_code == 3:
            return [(x / y) if y != 0 else 0.0 for x, y in zip(a[:n], b[:n])]
        return [x + y for x, y in zip(a[:n], b[:n])]

    def lookup_hash(self, table: Sequence[int], key: int) -> int:
        """Scan a table of 64-bit trajectory hashes using SIMD vector comparison."""
        n = len(table)
        if n == 0:
            return -1

        if self.lib is not None and sys.maxsize > 2**32:
            c_tbl = (ctypes.c_uint64 * n)(*table)
            return int(self.lib.agentjit_batch_lookup_hash(c_tbl, n, key))

        try:
            return table.index(key)
        except ValueError:
            return -1


_DEFAULT_ENGINE: Optional[FASMHardwareEngine] = None


def get_fasm_engine() -> FASMHardwareEngine:
    """Return the global singleton FASMHardwareEngine instance."""
    global _DEFAULT_ENGINE
    if _DEFAULT_ENGINE is None:
        _DEFAULT_ENGINE = FASMHardwareEngine()
    return _DEFAULT_ENGINE
