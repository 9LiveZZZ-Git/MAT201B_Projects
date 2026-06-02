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
    "${REAGENCY_ROOT}/viz/CreditLayer.cpp"
    "${REAGENCY_ROOT}/viz/DreamLayer.cpp"
    "${REAGENCY_ROOT}/viz/EmergencePlayer.cpp"
    "${REAGENCY_ROOT}/viz/DetectionHUD.cpp"
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
        CXX_STANDARD 14 CXX_STANDARD_REQUIRED ON)
    target_link_libraries(reagency_support PUBLIC al)
    # Stage 2 (decorrelated bed): al_ext/spatialaudio builds as `al_spatialaudio` ONLY when FFTW is
    # found (Linux/Darwin). When present, give AudioEngine.cpp the al_ext include path + WOSW_HAVE_DECORR
    # so it pulls al::Decorrelation; the lib itself reaches the app via the playground's ${AL_EXT_LIBRARIES}.
    # Absent FFTW -> macro undefined everywhere -> the decorr code compiles out (no link error).
    if(TARGET al_spatialaudio)
        target_compile_definitions(reagency_support PRIVATE WOSW_HAVE_DECORR)
        target_include_directories(reagency_support PRIVATE "${al_path}/..")   # for "al_ext/spatialaudio/..."
    endif()
endif()

set(app_link_libs reagency_support)
set(app_compile_flags -std=c++14)   # AlloSphere toolchain has no C++17; build the whole project as C++14 (allolib is C++14)

# Cuttlebone state distribution: the playground top-level CMake already links ${AL_EXT_LIBRARIES}
# into every app, which includes al_statedistribution (carrying PUBLIC AL_USE_CUTTLEBONE + the
# cuttlebone/ include dir) on Linux/Darwin. So linking is free on this route; we only need to set
# our include-guard macro so main_reagency.cpp pulls the cuttlebone header. Windows stays OSC-only.
if(CMAKE_SYSTEM_NAME MATCHES "Linux" OR CMAKE_SYSTEM_NAME MATCHES "Darwin")
  set(app_definitions ${app_definitions} WOSW_HAVE_CUTTLEBONE)
endif()

# Stage 2: define WOSW_HAVE_DECORR for the APP TU too (main_reagency.cpp includes AudioEngine.hpp),
# so the app + reagency_support see an IDENTICAL AudioEngine layout (decorr members are #if-guarded) —
# otherwise an ODR mismatch. Only when al_spatialaudio actually built (FFTW present).
if(TARGET al_spatialaudio)
  set(app_definitions ${app_definitions} WOSW_HAVE_DECORR)
endif()
