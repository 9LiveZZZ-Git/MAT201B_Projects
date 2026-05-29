#!/bin/bash
# Build and run corvid_m1 (allolib-native alpha demo).
# Run from anywhere: bash MAT201B_Projects/corvid/run_demo.sh [target]
#   target defaults to corvid_m1 (try corvid_p3 for the distributed demo).

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build"
TARGET="${1:-corvid_m1}"

JOBS=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)

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
# Configure if cache is missing. allolib needs the policy-min shim.
# --------------------------------------------------------------------------
if [ ! -f "${BUILD_DIR}/CMakeCache.txt" ]; then
  echo "[corvid] configuring with generator: ${GENERATOR}"
  "${CMAKE_BIN}" \
    -S "${SCRIPT_DIR}" \
    -B "${BUILD_DIR}" \
    -G "${GENERATOR}" \
    ${PLATFORM_FLAG} \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
    || exit 1
fi

# --------------------------------------------------------------------------
# Build + run
# --------------------------------------------------------------------------
echo "[corvid] building ${TARGET}..."
"${CMAKE_BIN}" --build "${BUILD_DIR}" --target "${TARGET}" --config Release -j "${JOBS}" || exit 1

BIN="${BUILD_DIR}/Release/${TARGET}.exe"
[ -f "${BIN}" ] || BIN="${BUILD_DIR}/${TARGET}.exe"
[ -f "${BIN}" ] || BIN="${BUILD_DIR}/bin/${TARGET}"
[ -f "${BIN}" ] || BIN="${BUILD_DIR}/${TARGET}"

echo "[corvid] launching ${TARGET}..."
"${BIN}"
