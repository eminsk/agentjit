#!/usr/bin/env bash
# =============================================================================
# AgentJIT - Native Acceleration Builder for Linux & macOS
# Compiles hardware-accelerated AVX2+FMA microkernels for AI Agent Trajectories
# =============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
SRC_DIR="${ROOT_DIR}/src/agentjit"

OS="$(uname -s)"
ARCH="$(uname -m)"

echo "====================================================================="
echo "  AgentJIT - Native Library Builder"
echo "  Detected OS:   ${OS}"
echo "  Architecture:  ${ARCH}"
echo "====================================================================="

CC=${CC:-gcc}
if ! command -v "${CC}" >/dev/null 2>&1; then
    CC="clang"
fi

if [ "${OS}" = "Darwin" ]; then
    OUT_LIB="${SRC_DIR}/libagentjit.dylib"
    echo "Compiling for macOS using ${CC}..."
    if [ "${ARCH}" = "arm64" ]; then
        ${CC} -O3 -shared -fPIC -ffast-math "${SCRIPT_DIR}/agentjit_kernel.c" -o "${OUT_LIB}"
    else
        ${CC} -O3 -shared -fPIC -mavx2 -mfma -ffast-math "${SCRIPT_DIR}/agentjit_kernel.c" -o "${OUT_LIB}"
    fi
    echo "[OK] Built macOS dynamic library: ${OUT_LIB}"
elif [ "${OS}" = "Linux" ]; then
    OUT_LIB="${SRC_DIR}/libagentjit64.so"
    echo "Compiling for Linux using ${CC}..."
    if [ "${ARCH}" = "x86_64" ]; then
        ${CC} -O3 -shared -fPIC -mavx2 -mfma -ffast-math "${SCRIPT_DIR}/agentjit_kernel.c" -o "${OUT_LIB}" -lm
    else
        ${CC} -O3 -shared -fPIC -ffast-math "${SCRIPT_DIR}/agentjit_kernel.c" -o "${OUT_LIB}" -lm
    fi
    cp "${OUT_LIB}" "${SCRIPT_DIR}/libagentjit64.so" 2>/dev/null || true
    echo "[OK] Built Linux shared object: ${OUT_LIB}"
fi

# Optional FASM assembly build if fasm command exists:
if command -v fasm >/dev/null 2>&1; then
    echo "Flat Assembler (fasm) is available."
fi

echo "====================================================================="
echo "  Build Complete! AgentJIT hardware acceleration is ready."
echo "====================================================================="
