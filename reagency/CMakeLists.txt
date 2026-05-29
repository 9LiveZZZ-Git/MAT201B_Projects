cmake_minimum_required(VERSION 3.24)
project(reagency CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --------------------------------------------------------------------------
# World of Shadow Work — allolib + plain-C++ ONLY runtime (no ML, no raw GL).
# All ML is offline (factory/, Python); the app loads plain data assets.
# Mirrors corvid's build: reuse the playground's allolib submodule.
# --------------------------------------------------------------------------
set(ALLOLIB_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../allolib")
add_subdirectory(${ALLOLIB_DIR} allolib_build)

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
    core/Conductor.cpp)
target_link_libraries(reagency PRIVATE al Threads::Threads)
target_include_directories(reagency PRIVATE ${REAGENCY_INCLUDE_DIRS})
if(MSVC)
  target_compile_options(reagency PRIVATE /W3 /WX-)
else()
  target_compile_options(reagency PRIVATE -Wall -Wextra -fno-omit-frame-pointer)
endif()
