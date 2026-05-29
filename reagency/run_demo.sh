#!/bin/bash
# Build and run "World of Shadow Work" (allolib-native).
# Run from anywhere:  bash MAT201B_Projects/reagency/run_demo.sh [target]
#   target defaults to reagency.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build"
TARGET="${1:-reagency}"
JOBS=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)

# Pick cmake (VS-bundled on Windows, else system cmake).
VS_CMAKE="C:/Program Files/Microsoft Visual Studio/2022/Community/Common7/IDE/CommonExtensions/Microsoft/CMake/CMake/bin/cmake.exe"
if [ -f "${VS_CMAKE}" ]; then
  CMAKE_BIN="${VS_CMAKE}"; GENERATOR="Visual Studio 17 2022"; PLATFORM_FLAG="-A x64"
else
  CMAKE_BIN="cmake"; GENERATOR="Unix Makefiles"; PLATFORM_FLAG=""
fi

if [ ! -f "${BUILD_DIR}/CMakeCache.txt" ]; then
  echo "[wosw] configuring with generator: ${GENERATOR}"
  "${CMAKE_BIN}" \
    -S "${SCRIPT_DIR}" \
    -B "${BUILD_DIR}" \
    -G "${GENERATOR}" \
    ${PLATFORM_FLAG} \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
    || exit 1
fi

echo "[wosw] building ${TARGET}..."
"${CMAKE_BIN}" --build "${BUILD_DIR}" --target "${TARGET}" --config Release -j "${JOBS}" || exit 1

BIN="${BUILD_DIR}/Release/${TARGET}.exe"
[ -f "${BIN}" ] || BIN="${BUILD_DIR}/${TARGET}.exe"
[ -f "${BIN}" ] || BIN="${BUILD_DIR}/bin/${TARGET}"
[ -f "${BIN}" ] || BIN="${BUILD_DIR}/${TARGET}"

echo "[wosw] launching ${TARGET}..."
"${BIN}" "${SCRIPT_DIR}/assets"
