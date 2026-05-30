# flags.cmake — lets the playground single-file builder (run.sh) build
# main_reagency.cpp together with its companion sources, WITHOUT touching run.sh.
#
#   ./run.sh MAT201B_Projects/reagency/src/main_reagency.cpp
#
# The single-file builder (allolib_playground/CMakeLists.txt) `include()`s this file
# from the app's directory and reads app_include_dirs / app_link_libs / app_compile_flags.
# Since it compiles only the one app .cpp, companion units are built into a static lib
# and linked (same pattern as corvid/src/flags.cmake). allolib + plain C++ only.

# reagency/ root is one level up from this src/ directory.
get_filename_component(REAGENCY_ROOT "${app_path}/.." ABSOLUTE)

# Headers: reagency/ root (for "core/..." and "viz/...") and Gamma.
set(app_include_dirs
    "${REAGENCY_ROOT}"
    "${al_path}/external/Gamma"
)

# Companion translation units main_reagency.cpp depends on.
set(_reagency_lib_srcs
    "${REAGENCY_ROOT}/viz/ParticleField.cpp"
    "${REAGENCY_ROOT}/viz/WebRenderer.cpp"
    "${REAGENCY_ROOT}/viz/VesselSplats.cpp"
    "${REAGENCY_ROOT}/viz/HumanTrace.cpp"
    "${REAGENCY_ROOT}/viz/LabelLayer.cpp"
    "${REAGENCY_ROOT}/viz/CaptionLayer.cpp"
    "${REAGENCY_ROOT}/viz/StoryLayer.cpp"
    "${REAGENCY_ROOT}/core/Conductor.cpp"
    "${REAGENCY_ROOT}/audio/AudioEngine.cpp"
)

if(NOT TARGET reagency_support)
    add_library(reagency_support STATIC ${_reagency_lib_srcs})
    target_include_directories(reagency_support PRIVATE
        "${REAGENCY_ROOT}"
        "${al_path}/include"
        "${al_path}/external/Gamma"
    )
    set_target_properties(reagency_support PROPERTIES
        CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON)
    target_link_libraries(reagency_support PUBLIC al)
endif()

set(app_link_libs reagency_support)
set(app_compile_flags -std=c++17)
