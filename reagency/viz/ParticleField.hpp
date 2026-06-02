#pragma once
// ParticleField — THE one conserved particle buffer (every grain = one corpus
// image or word). Rendered as additive Gaussian point-sprites through a custom
// al::ShaderProgram (allolib-native; no raw GL).
//
// M0: galaxy layout only (the loaded UMAP positions). Later milestones re-lay-out
// the SAME buffer per shell (smoke / vessel) by changing vertex-shader uniforms;
// the mesh is built ONCE in init() and never rebuilt per frame.
//
// Loads assets/points.bin ("WSWP": 10 float32/point = xyz, rgb, density, cluster,
// type, atlas_idx). If absent, a procedural galaxy is synthesized so the app always
// runs (mirrors corvid's no-student.bin fallback).
#include "al/graphics/al_Graphics.hpp"
#include "al/graphics/al_Shader.hpp"
#include "al/graphics/al_VAOMesh.hpp"
#include "al/math/al_Vec.hpp"
#include <string>
#include <vector>

namespace wosw {

struct ParticleField {
  // assetDir: directory containing points.bin (tries a few candidates; falls back
  // to a procedural galaxy if none found). Returns false only if the shader fails.
  bool init(const std::string& assetDir);

  // pointScale scales sprite size; bright scales overall intensity (the galaxy
  // dims toward the vessel end of the depth crossfade). posterize (v2 T10) quantizes
  // the color — the Act-V haunt residue that never fully clears once the turn is reached.
  void draw(al::Graphics& g, float pointScale = 1.f, float bright = 1.f, float posterize = 0.f);

  int  count()      const { return n_; }
  bool loadedReal() const { return loaded_real_; }

  // Galaxy points for WebRenderer / Conductor (copied once at init).
  std::vector<al::Vec3f> positions() const;
  std::vector<al::Vec3f> colors() const;

  // Per-node lookups (HumanTrace / Conductor).
  al::Vec3f posOf(int i)    const { return (i >= 0 && i < int(pts_.size())) ? pts_[i].pos : al::Vec3f(0, 0, 0); }
  int       typeOf(int i)   const { return (i >= 0 && i < int(pts_.size())) ? pts_[i].type : -1; }
  int       atlasOf(int i)  const { return (i >= 0 && i < int(pts_.size())) ? pts_[i].atlas : -1; }
  float     densityOf(int i) const { return (i >= 0 && i < int(pts_.size())) ? pts_[i].density : 0.5f; }
  int       clusterOf(int i) const { return (i >= 0 && i < int(pts_.size())) ? pts_[i].cluster : -1; }

 private:
  struct P {
    al::Vec3f pos, col;
    float density = 0.5f, sigma = 0.5f;
    int   cluster = 0, type = 0, atlas = -1;
  };

  std::vector<P>    pts_;
  int               n_ = 0;
  bool              loaded_real_ = false;

  al::VAOMesh       mesh_;
  al::ShaderProgram shader_;
  bool              shader_ok_ = false;

  // Look / cheap per-sprite "bloom" (no post-process; dome-bright). All one-line tunables.
  float point_size_    = 14.f;   // sprite quad size (19->14: ~45% less fill/sprite -> faster on weak dome projectors; also tones the bloom)
  float core_sigma_    = 0.22f;  // bright CORE — kept at the original ~size-7 apparent radius
  float halo_sigma_    = 0.55f;  // soft glow radius
  float halo_strength_ = 0.28f;  // glow intensity relative to the core (0.50 -> 0.35 -> 0.28; less overdraw-y bloom)
  float intensity_     = 1.70f;  // overall brightness (AlloSphere projector washout headroom)

  bool loadPoints(const std::string& path);
  void synthesize(int n);   // procedural fallback galaxy
  void buildMesh();         // ONCE
};

}  // namespace wosw
