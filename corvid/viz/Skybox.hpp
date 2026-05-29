#pragma once
// Generative "thinking" skybox (Part B, Phase 7) — allolib-native.
//
// A large inward-facing sphere centered on the viewer, shaded by a custom
// al::ShaderProgram fragment shader that paints a domain-warped fbm field over
// the world-space view direction. The field is driven by:
//   - a crow-image-derived color palette (built from assets/crows/*.jpg), and
//   - the live ThoughtVector from the tuned lens (Phase 6): entropy controls
//     turbulence/incoherence ("hallucination"), confidence/value control
//     brightness, the action distribution shifts hue.
//
// Omni/dome-safe by construction: it is REAL 3D geometry shaded via
// al_ModelViewMatrix / al_ProjectionMatrix, so allolib's per-projection capture
// renders + warps it correctly on the sphere (NOT a clip-space fullscreen quad,
// which would not warp). Draw it first each frame with depth-write off.
//
// Sphere distribution: draw() takes the ThoughtVector + time explicitly, so on
// the dome the primary's broadcast values drive every render node identically.
#include "al/graphics/al_Graphics.hpp"
#include "al/graphics/al_Shapes.hpp"
#include "al/graphics/al_Texture.hpp"
#include "al/graphics/al_VAOMesh.hpp"
#include "al/graphics/al_Shader.hpp"
#include "al/math/al_Vec.hpp"
#include "cognition/Lens.hpp"   // ThoughtVector
#include <string>

namespace corvid {

struct Skybox {
    // Compile the shader and build the crow-image palette.
    // crow_dir: directory of crow .jpg images (palette source); if empty/missing,
    // a neutral palette is used. radius: skybox sphere radius (world units).
    // Returns false if the shader failed to compile.
    bool init(const std::string& crow_dir, float radius = 60.f);

    // Draw the skybox centered at eyePos. Call FIRST in onDraw, before agents.
    // tv: current thought vector (drives the field). t: animation time (seconds).
    void draw(al::Graphics& g, const al::Vec3d& eyePos,
              const ThoughtVector& tv, float t);

    bool ready() const { return shader_ok_; }
    int  paletteSize() const { return palette_n_; }

private:
    al::VAOMesh      sphere_;
    al::ShaderProgram shader_;
    al::Texture      palette_;
    bool             shader_ok_ = false;
    int              palette_n_ = 0;
    float            radius_    = 60.f;
};

} // namespace corvid
