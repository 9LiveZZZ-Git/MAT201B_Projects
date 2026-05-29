#pragma once
// VesselSplats — the outer "vessel": the machine's image-to-3D Gaussian-splat hallucination
// of the fixated cluster's representative photo, MELTED between concept keyframes live.
//
// Loads vessel.wswv (WSWV: K keyframes x G gaussians x [pos3,rgb3,opacity1,sigma1] float16,
// from factory/stage_d_vessel.py). If absent, synthesizes a procedural vessel so the app
// always runs. Morph is CPU-side index-lerp between two keyframes (the offline Morton
// ordering makes that a real morph, not swimming) at the dome-safe G — cheaper and more
// allolib-native than packing dual keyframes into custom vertex attributes. The smoky melt
// (curl-noise spread + opacity dip + size swell near the crossfade midpoint) gives the
// SplatFlow/Anadol register. Rendered as additive bloom point-sprites via al::ShaderProgram.
#include "al/graphics/al_Graphics.hpp"
#include "al/graphics/al_Shader.hpp"
#include "al/graphics/al_VAOMesh.hpp"
#include "al/math/al_Vec.hpp"
#include <string>
#include <vector>

namespace wosw {

struct VesselSplats {
  bool init(const std::string& assetDir);

  // kfFloat: continuous keyframe position. A = floor(kfFloat) % K, B = (A+1) % K,
  // blend = frac(kfFloat). Morphs A->B and applies the melt. Call once per frame on
  // every node (driven by synced state, so it stays identical across the dome).
  void update(float kfFloat, float time);

  // Additive bloom point-sprites at `center`, spread by `scale`, faded by `alpha`
  // (the galaxy<->vessel depth crossfade).
  void draw(al::Graphics& g, const al::Vec3f& center, float scale, float alpha);

  int  keyframeCount() const { return K_; }
  bool loadedReal()    const { return loaded_real_; }

 private:
  struct Splat { al::Vec3f pos, rgb; float opacity = 0.6f, sigma = 0.5f; };

  std::vector<std::vector<Splat>> kf_;   // K keyframes x G splats
  int  K_ = 0, G_ = 0;
  bool loaded_real_ = false;

  al::VAOMesh       mesh_;
  al::ShaderProgram shader_;
  bool              shader_ok_ = false;

  // bloom tunables (match ParticleField; vessel runs a touch bigger/brighter)
  float point_size_    = 18.f;
  float core_sigma_    = 0.20f;
  float halo_sigma_    = 0.55f;
  float halo_strength_ = 0.55f;
  float intensity_     = 1.70f;

  bool loadWSWV(const std::string& path);
  void synthesize(int K, int G);
};

}  // namespace wosw
