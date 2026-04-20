#!/bin/bash
# Build and run corvid_p3 (alpha screen demo).
# Run from anywhere: bash MAT201B_projects/corvid/run_demo.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build"

# Number of parallel jobs
JOBS=$(nproc 2>/dev/null || echo 4)

# --------------------------------------------------------------------------
# Pick cmake — VS-bundled cmake handles MSVC correctly on Windows
# --------------------------------------------------------------------------
VS_CMAKE="C:/Program Files/Microsoft Visual Studio/2022/Community/Common7/IDE/CommonExtensions/Microsoft/CMake/CMake/bin/cmake.exe"
if [ -f "${VS_CMAKE}" ]; then
  CMAKE_BIN="${VS_CMAKE}"
  GENERATOR="Visual Studio 17 2022"
  PLATFORM_FLAG="-A x64"
else
  CMAKE_BIN="cmake"
  GENERATOR="Unix Makefiles"
  PLATFORM_FLAG=""
fi

# --------------------------------------------------------------------------
# Configure if cache is missing
# --------------------------------------------------------------------------
if [ ! -f "${BUILD_DIR}/CMakeCache.txt" ]; then
  echo "[corvid] configuring with generator: ${GENERATOR}"
  "${CMAKE_BIN}" \
    -S "${SCRIPT_DIR}" \
    -B "${BUILD_DIR}" \
    -G "${GENERATOR}" \
    ${PLATFORM_FLAG} \
    -DCMAKE_BUILD_TYPE=Release \
    -DCORVID_USE_LLM=ON \
    -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
    || exit 1
fi

# --------------------------------------------------------------------------
# Build corvid_p3
# --------------------------------------------------------------------------
echo "[corvid] building corvid_p3..."
"${CMAKE_BIN}" --build "${BUILD_DIR}" --target corvid_p3 --config Release -j "${JOBS}" || exit 1

# --------------------------------------------------------------------------
# Run
# --------------------------------------------------------------------------
MODEL="C:/Users/lpfre/allolib_playground/MAT201B_projects/assets/models/gemma-4-E2B-it-Q4_K_M.gguf"

echo "[corvid] launching screen demo..."
# VS puts the binary in a config subdirectory
BIN="${BUILD_DIR}/Release/corvid_p3.exe"
[ -f "${BIN}" ] || BIN="${BUILD_DIR}/corvid_p3.exe"
[ -f "${BIN}" ] || BIN="${BUILD_DIR}/corvid_p3"

if [ -f "${MODEL}" ]; then
  echo "[corvid] model found, LLM overlay enabled"
  "${BIN}" --model "${MODEL}"
else
  echo "[corvid] WARNING: model not found at ${MODEL}, running without LLM"
  "${BIN}"
fi
