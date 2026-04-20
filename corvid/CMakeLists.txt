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
