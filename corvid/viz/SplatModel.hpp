#pragma once
// Distilled-student crow splat cloud (Part B, Phase 8) — allolib-native runtime.
//
// A Gaussian-splat point field "hallucinated" from the user's crow images and
// evolved live by an agent's tuned-lens ThoughtVector.
//
// Two evolution sources, same renderer:
//   1. DISTILLED STUDENT (preferred): a tiny per-point MLP whose weights are
//      produced OFFLINE by distilling a large diffusion/3DGS teacher on the crow
//      images, exported as a plain float file `student.bin`. At runtime the
//      hand-rolled forward (no torch) maps (base_pos, cond) -> point delta in a
//      few steps. See docs/REWRITE_PLAN.md Phase 8a for the offline recipe + the
//      student.bin format documented in SplatModel.cpp.
//   2. PROCEDURAL FALLBACK (no student.bin): the field is seeded by CPU-sampling
//      a crow image into a 3D point cloud and animated with value noise. Same
//      look, no learned weights — so the app always runs.
//
// Rendering: al::ShaderProgram point sprites with a radial Gaussian falloff and
// additive blending (the "splat" look), drawn from a VAOMesh of POINTS. All
// evolution is deterministic from (ThoughtVector, time, seed) so it stays
// consistent across dome render nodes (Phase 10).
#include "al/graphics/al_Graphics.hpp"
#include "al/graphics/al_VAOMesh.hpp"
#include "al/graphics/al_Shader.hpp"
#include "al/math/al_Vec.hpp"
#include "cognition/Lens.hpp"   // ThoughtVector
#include <string>
#include <vector>

namespace corvid {

struct SplatModel {
    // crow_path: a crow image file to seed the cloud (silhouette/color source).
    // student_path: optional distilled weights; if missing/invalid -> procedural.
    // n_points: target splat count (kept modest for dome fill-rate).
    bool init(const std::string& crow_path,
              const std::string& student_path,
              int n_points = 4000);

    // Advance the field one frame for the focused agent's thought.
    void update(const ThoughtVector& tv, float t, float dt);

    // Draw the cloud centered at `center` (e.g. the focused corvid's position).
    void draw(al::Graphics& g, const al::Vec3f& center, float worldScale = 1.f);

    bool ready()       const { return shader_ok_; }
    bool usingStudent() const { return have_student_; }
    int  count()       const { return n_; }

private:
    struct Splat { al::Vec3f base, pos, color; float sigma, alpha; };

    std::vector<Splat> splats_;
    int  n_ = 0;

    al::VAOMesh       mesh_;
    al::ShaderProgram shader_;
    bool              shader_ok_   = false;
    bool             have_student_ = false;

    // Distilled student (tiny per-point MLP): in = base_pos(3)+cond(COND) ->
    // hidden -> out = dpos(3)+dcol(3)+dsigma(1). Plain floats, no deps.
    static constexpr int COND = 8;   // ThoughtVector compact conditioning
    int sd_in_ = 0, sd_hid_ = 0, sd_out_ = 0;
    std::vector<float> sd_W1, sd_b1, sd_W2, sd_b2;

    bool loadStudent(const std::string& path);
    void studentForward(const float* in, float* out) const;
    void rebuildMesh();
};

} // namespace corvid
