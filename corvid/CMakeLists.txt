cmake_minimum_required(VERSION 3.24)
project(corvid CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --------------------------------------------------------------------------
# Allolib — reuse playground's submodule
# --------------------------------------------------------------------------
set(ALLOLIB_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../allolib")
add_subdirectory(${ALLOLIB_DIR} allolib_build)

# Common include paths for all targets
set(CORVID_INCLUDE_DIRS
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${ALLOLIB_DIR}/include"
    "${ALLOLIB_DIR}/external/Gamma"
)

# --------------------------------------------------------------------------
# LLM option — llama.cpp
# Uses vendor/llama.cpp if present (git clone --depth 1 --branch b5575),
# otherwise falls back to FetchContent for CI.
# --------------------------------------------------------------------------
option(CORVID_USE_LLM "Enable llama.cpp LLM smoke test" ON)

if(CORVID_USE_LLM)
  set(LLAMA_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
  set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
  set(GGML_CUDA            OFF CACHE BOOL "" FORCE)  # CPU-only; see pins.toml
  set(GGML_NATIVE          ON  CACHE BOOL "" FORCE)

  set(CORVID_LLAMA_VENDOR "${CMAKE_CURRENT_SOURCE_DIR}/vendor/llama.cpp")
  if(EXISTS "${CORVID_LLAMA_VENDOR}/CMakeLists.txt")
    add_subdirectory(${CORVID_LLAMA_VENDOR} llama_cpp_build)
    set(llama_cpp_SOURCE_DIR "${CORVID_LLAMA_VENDOR}")
  else()
    include(FetchContent)
    FetchContent_Declare(
      llama_cpp
      GIT_REPOSITORY https://github.com/ggerganov/llama.cpp
      GIT_TAG        b5575
      GIT_SHALLOW    ON
    )
    FetchContent_MakeAvailable(llama_cpp)
  endif()
endif()

# --------------------------------------------------------------------------
# Torch — prefer vendor/libtorch standalone zip (no Python DLL init issues),
# fall back to pip-installed PyTorch for cmake path detection.
#
# To use standalone libtorch:
#   1. Unzip libtorch-win-shared-with-deps-2.9.0+cu126.zip to vendor/libtorch/
#   2. Re-run cmake (CORVID_TORCH_PREFIX auto-detected)
# --------------------------------------------------------------------------
option(CORVID_USE_TORCH "Enable libtorch smoke test" ON)

if(CORVID_USE_TORCH)
  if(NOT CORVID_TORCH_PREFIX)
    # 1. Prefer standalone libtorch zip (vendor/libtorch)
    set(_vendor_libtorch "${CMAKE_CURRENT_SOURCE_DIR}/vendor/libtorch/share/cmake/Torch")
    if(EXISTS "${_vendor_libtorch}")
      set(CORVID_TORCH_PREFIX "${_vendor_libtorch}")
      message(STATUS "Torch: using standalone vendor/libtorch")
    else()
      # 2. Fall back to pip-installed (Windows-only caveat: crashes pre-main in standalone exes)
      execute_process(
        COMMAND python -c
          "import torch, os; print(os.path.join(os.path.dirname(torch.__file__), 'share', 'cmake', 'Torch'))"
        OUTPUT_VARIABLE _torch_cmake_dir
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
      )
      if(_torch_cmake_dir AND EXISTS "${_torch_cmake_dir}")
        set(CORVID_TORCH_PREFIX "${_torch_cmake_dir}")
        message(STATUS "Torch: using pip-installed (runtime may crash on Windows standalone exe)")
      endif()
    endif()
  endif()

  if(CORVID_TORCH_PREFIX)
    set(CAFFE2_USE_CUDA OFF CACHE BOOL "" FORCE)
    find_package(Torch REQUIRED PATHS "${CORVID_TORCH_PREFIX}" NO_DEFAULT_PATH)
    message(STATUS "Torch found: ${Torch_VERSION} at ${CORVID_TORCH_PREFIX}")
    set(CMAKE_CXX_STANDARD 17)
  else()
    message(WARNING "CORVID_USE_TORCH=ON but libtorch not found — unzip vendor/libtorch or install torch via pip")
    set(CORVID_USE_TORCH OFF)
  endif()
endif()

# --------------------------------------------------------------------------
# Helper: apply shared compiler flags
# --------------------------------------------------------------------------
function(corvid_target_flags tgt)
  if(MSVC)
    target_compile_options(${tgt} PRIVATE /W3 /WX-)
  else()
    target_compile_options(${tgt} PRIVATE -Wall -Wextra -fno-omit-frame-pointer)
  endif()
  target_include_directories(${tgt} PRIVATE ${CORVID_INCLUDE_DIRS})
endfunction()

# --------------------------------------------------------------------------
# corvid_p1 — love-chase agents
# --------------------------------------------------------------------------
add_executable(corvid_p1 src/main_p1.cpp)
target_link_libraries(corvid_p1 PRIVATE al)
corvid_target_flags(corvid_p1)

# --------------------------------------------------------------------------
# corvid_p2 — Reynolds flocking + spatial hash + GUI
# --------------------------------------------------------------------------
add_executable(corvid_p2 src/main_p2.cpp)
target_link_libraries(corvid_p2 PRIVATE al)
corvid_target_flags(corvid_p2)

# --------------------------------------------------------------------------
# corvid_p3 — DistributedAppWithState + PolySynth + alpha demo
# --------------------------------------------------------------------------
if(CORVID_USE_LLM)
  add_executable(corvid_p3 src/main_p3.cpp cognition/LlmSmoke.cpp)
  target_link_libraries(corvid_p3 PRIVATE al llama)
  target_compile_definitions(corvid_p3 PRIVATE CORVID_LLM)
  target_include_directories(corvid_p3 PRIVATE
    ${CORVID_INCLUDE_DIRS}
    ${llama_cpp_SOURCE_DIR}
    ${llama_cpp_SOURCE_DIR}/include
    ${llama_cpp_SOURCE_DIR}/ggml/include
  )
  if(MSVC)
    target_compile_options(corvid_p3 PRIVATE /W3 /WX-)
  else()
    target_compile_options(corvid_p3 PRIVATE -Wall -Wextra -fno-omit-frame-pointer)
  endif()
else()
  add_executable(corvid_p3 src/main_p3.cpp)
  target_link_libraries(corvid_p3 PRIVATE al)
  corvid_target_flags(corvid_p3)
endif()

# --------------------------------------------------------------------------
# corvid_llm_smoke — llama.cpp C API wrapper
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# corvid_torch_smoke — libtorch + RavenNet stub + LoRA forward pass
# --------------------------------------------------------------------------
if(CORVID_USE_TORCH)
  add_executable(corvid_torch_smoke src/main_torch_smoke.cpp)
  target_link_libraries(corvid_torch_smoke PRIVATE "${TORCH_LIBRARIES}")
  target_include_directories(corvid_torch_smoke PRIVATE ${TORCH_INCLUDE_DIRS})
  set_property(TARGET corvid_torch_smoke PROPERTY CXX_STANDARD 17)
  if(MSVC)
    # Torch requires /bigobj and disables some warnings
    target_compile_options(corvid_torch_smoke PRIVATE /W3 /WX- /bigobj)
    # Delay-load core torch DLLs so they init after CRT is fully set up,
    # avoiding the FAST_FAIL_INCORRECT_STACK crash in ucrtbase.dll DllMain.
    target_link_options(corvid_torch_smoke PRIVATE
      /DELAYLOAD:torch.dll /DELAYLOAD:torch_cpu.dll /DELAYLOAD:torch_cuda.dll)
    target_link_libraries(corvid_torch_smoke PRIVATE delayimp.lib)
    # Copy torch DLLs next to the exe post-build (exclude torch_python.dll — it
    # requires a live Python interpreter and crashes DllMain in standalone exes)
    file(GLOB _torch_dlls "${CORVID_TORCH_PREFIX}/../../../lib/*.dll")
    list(FILTER _torch_dlls EXCLUDE REGEX "torch_python\\.dll$")
    add_custom_command(TARGET corvid_torch_smoke POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${_torch_dlls} $<TARGET_FILE_DIR:corvid_torch_smoke>)
  else()
    target_compile_options(corvid_torch_smoke PRIVATE -Wall -Wextra)
  endif()
endif()

# --------------------------------------------------------------------------
# corvid_m1 — M1 alpha demo (energy · reproduction · death · place grid · entities)
# --------------------------------------------------------------------------
add_executable(corvid_m1 src/main_corvid.cpp)
target_link_libraries(corvid_m1 PRIVATE al)
corvid_target_flags(corvid_m1)

if(CORVID_USE_LLM)
  add_executable(corvid_llm_smoke
    src/main_llm_smoke.cpp
    cognition/LlmSmoke.cpp
  )
  target_link_libraries(corvid_llm_smoke PRIVATE al llama)
  target_include_directories(corvid_llm_smoke PRIVATE
    ${CORVID_INCLUDE_DIRS}
    ${llama_cpp_SOURCE_DIR}
    ${llama_cpp_SOURCE_DIR}/include
    ${llama_cpp_SOURCE_DIR}/ggml/include
  )
  if(MSVC)
    target_compile_options(corvid_llm_smoke PRIVATE /W3 /WX-)
  else()
    target_compile_options(corvid_llm_smoke PRIVATE -Wall -Wextra -fno-omit-frame-pointer)
  endif()
endif()
