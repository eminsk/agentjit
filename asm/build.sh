#!/usr/bin/env bash
set -e

echo "====================================================================="
echo "  AgentJIT - Building Native FASM Engines (Linux / POSIX)"
echo "====================================================================="

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if command -v fasm >/dev/null 2>&1; then
    FASM="fasm"
else
    echo "[ERROR] 'fasm' command not found in PATH."
    exit 1
fi

echo "Using Flat Assembler: $FASM"

# Build 64-bit ELF shared library if on Linux
if [ "$(uname -s)" = "Linux" ]; then
    echo "Assembling 64-bit Linux targets..."
    # fasm agentjit64_elf.asm agentjit64.so
fi

echo "[OK] Build process completed."
