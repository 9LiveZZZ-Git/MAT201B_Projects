cmake_minimum_required(VERSION 3.24)
project(corvid CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --------------------------------------------------------------------------
# Allolib-native corvid.
#
# This project is intentionally allolib + plain-C++ only. The former libtorch
# (RavenNet) and llama.cpp (LLM reflection) dependencies have been removed:
#   - RavenNet  -> hand-rolled MLP + LoRA + PPO in cognition/RavenBrain.cpp
#   - reflection-> procedural reflector in cognition/Reflection.cpp, run on a
#                  background thread by cognition/LlmReflection.cpp
# No torch / llama / ggml / Python anywhere. See docs/REWRITE_PLAN.md.
# --------------------------------------------------------------------------

# Allolib — reuse playground's submodule
set(ALLOLIB_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../allolib")
add_subdirectory(${ALLOLIB_DIR} allolib_build)

find_package(Threads REQUIRED)

set(CORVID_INCLUDE_DIRS
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${ALLOLIB_DIR}/include"
    "${ALLOLIB_DIR}/external/Gamma"
)

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
# corvid_p3 — DistributedAppWithState + PolySynth + alpha demo (no LLM overlay)
# --------------------------------------------------------------------------
add_executable(corvid_p3 src/main_p3.cpp)
target_link_libraries(corvid_p3 PRIVATE al)
corvid_target_flags(corvid_p3)

# --------------------------------------------------------------------------
# corvid_m1 — M1/M2/M3 alpha demo (native RavenNet + procedural reflection)
# --------------------------------------------------------------------------
add_executable(corvid_m1
    src/main_corvid.cpp
    cognition/Perception.cpp
    cognition/Reflection.cpp
    cognition/RavenBrain.cpp
    cognition/LlmReflection.cpp)
target_link_libraries(corvid_m1 PRIVATE al Threads::Threads)
# RavenNet (hand-rolled) and threaded procedural reflection are always on now.
target_compile_definitions(corvid_m1 PRIVATE CORVID_USE_RAVENNET CORVID_USE_LLM)
corvid_target_flags(corvid_m1)

# --------------------------------------------------------------------------
# corvid_m2_test — Kahan EMA unit tests (M2 acceptance, spec §3.10.6)
# --------------------------------------------------------------------------
add_executable(corvid_m2_test test/test_place_kahan.cpp)
target_include_directories(corvid_m2_test PRIVATE ${CORVID_INCLUDE_DIRS})
if(MSVC)
    target_compile_options(corvid_m2_test PRIVATE /W3 /WX-)
else()
    target_compile_options(corvid_m2_test PRIVATE -Wall -Wextra)
endif()

# --------------------------------------------------------------------------
# corvid_brain_smoke — native RavenNet numeric check (no allolib needed)
# --------------------------------------------------------------------------
add_executable(corvid_brain_smoke test/test_ravenbrain.cpp cognition/RavenBrain.cpp)
target_include_directories(corvid_brain_smoke PRIVATE ${CORVID_INCLUDE_DIRS})
if(MSVC)
    target_compile_options(corvid_brain_smoke PRIVATE /W3 /WX-)
else()
    target_compile_options(corvid_brain_smoke PRIVATE -Wall -Wextra)
endif()

# --------------------------------------------------------------------------
# corvid_lens_smoke — tuned-lens readout check (Part B Phase 6, no allolib)
# --------------------------------------------------------------------------
add_executable(corvid_lens_smoke
    test/test_lens.cpp cognition/Lens.cpp cognition/RavenBrain.cpp)
target_include_directories(corvid_lens_smoke PRIVATE ${CORVID_INCLUDE_DIRS})
if(MSVC)
    target_compile_options(corvid_lens_smoke PRIVATE /W3 /WX-)
else()
    target_compile_options(corvid_lens_smoke PRIVATE -Wall -Wextra)
endif()
