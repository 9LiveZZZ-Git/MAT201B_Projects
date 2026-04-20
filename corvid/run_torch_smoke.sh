#!/bin/bash
# Build and run corvid_torch_smoke (Week 2 libtorch plumbing check).
# Run from anywhere: bash MAT201B_projects/corvid/run_torch_smoke.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Required by CUDA 13.2.targets (MSBuild) to locate the toolkit
export CUDA_PATH="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.2"
export CUDA_PATH_V13_2="${CUDA_PATH}"
BUILD_DIR="${SCRIPT_DIR}/build"

VS_CMAKE="C:/Program Files/Microsoft Visual Studio/2022/Community/Common7/IDE/CommonExtensions/Microsoft/CMake/CMake/bin/cmake.exe"
CMAKE_BIN="${VS_CMAKE}"

if [ ! -f "${BUILD_DIR}/CMakeCache.txt" ]; then
  echo "[corvid] configuring..."
  "${CMAKE_BIN}" \
    -S "${SCRIPT_DIR}" \
    -B "${BUILD_DIR}" \
    -G "Visual Studio 17 2022" -A x64 \
    -DCMAKE_BUILD_TYPE=Release \
    -DCORVID_USE_LLM=OFF \
    -DCORVID_USE_TORCH=ON \
    -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
    -DCUDA_TOOLKIT_ROOT_DIR="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.2" \
    || exit 1
fi

echo "[corvid] building corvid_torch_smoke..."
"${CMAKE_BIN}" --build "${BUILD_DIR}" --target corvid_torch_smoke --config Release -j8 || exit 1

BIN="${BUILD_DIR}/Release/corvid_torch_smoke.exe"
[ -f "${BIN}" ] || BIN="${BUILD_DIR}/corvid_torch_smoke.exe"

echo "[corvid] running torch smoke test..."
"${BIN}"
