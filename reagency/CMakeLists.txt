cmake_minimum_required(VERSION 3.24)
project(reagency CXX)
set(CMAKE_CXX_STANDARD 14)   # AlloSphere toolchain does NOT support C++17 — whole project is C++14 (matches allolib)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --------------------------------------------------------------------------
# World of Shadow Work — allolib + plain-C++ ONLY runtime (no ML, no raw GL).
# All ML is offline (factory/, Python); the app loads plain data assets.
# Mirrors corvid's build: reuse the playground's allolib submodule.
# --------------------------------------------------------------------------
set(ALLOLIB_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../allolib")
add_subdirectory(${ALLOLIB_DIR} allolib_build)

# --- cuttlebone state distribution (al_ext) — Linux/Darwin only --------------
# The dome's reliable fragmented-UDP-broadcast transport. al_ext/statedistribution auto-enables
# cuttlebone on Linux/Darwin (sets PUBLIC AL_USE_CUTTLEBONE + the cuttlebone/ include dir on the
# al_statedistribution target). On Windows / if the extension is absent we silently keep the OSC
# StateDistributionDomain that DistributedAppWithState installs. (The run.sh route gets this
# transitively via the playground's ${AL_EXT_LIBRARIES}; run_demo.sh — this file — wires it here.)
set(AL_EXT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../al_ext")
set(WOSW_STATEDIST_LIB "")
if((CMAKE_SYSTEM_NAME MATCHES "Linux" OR CMAKE_SYSTEM_NAME MATCHES "Darwin")
   AND EXISTS "${AL_EXT_DIR}/statedistribution/CMakeLists.txt")
  add_subdirectory("${AL_EXT_DIR}/statedistribution"
                   "${CMAKE_CURRENT_BINARY_DIR}/al_statedistribution_build")
  if(TARGET al_statedistribution)
    set(WOSW_STATEDIST_LIB al_statedistribution)
  endif()
endif()

find_package(Threads REQUIRED)

set(REAGENCY_INCLUDE_DIRS
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${ALLOLIB_DIR}/include"
    "${ALLOLIB_DIR}/external/Gamma")

# --------------------------------------------------------------------------
# reagency — the installation (M0: distributed galaxy point cloud)
# --------------------------------------------------------------------------
add_executable(reagency
    src/main_reagency.cpp
    viz/ParticleField.cpp
    viz/WebRenderer.cpp
    viz/VesselSplats.cpp
    viz/HumanTrace.cpp
    viz/LabelLayer.cpp
    viz/CaptionLayer.cpp
    viz/StoryLayer.cpp
    viz/CreditLayer.cpp
    viz/DreamLayer.cpp
    viz/EmergencePlayer.cpp
    viz/DetectionHUD.cpp
    audio/AudioEngine.cpp
    core/Conductor.cpp)
target_link_libraries(reagency PRIVATE al Threads::Threads ${WOSW_STATEDIST_LIB})
target_include_directories(reagency PRIVATE ${REAGENCY_INCLUDE_DIRS} "${AL_EXT_DIR}/..")
if(WOSW_STATEDIST_LIB)
  target_compile_definitions(reagency PRIVATE WOSW_HAVE_CUTTLEBONE)   # gates the cuttlebone #include/onInit
endif()
if(MSVC)
  target_compile_options(reagency PRIVATE /W3 /WX-)
else()
  target_compile_options(reagency PRIVATE -Wall -Wextra -fno-omit-frame-pointer)
endif()
