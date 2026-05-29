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

  void draw(al::Graphics& g, float pointScale = 1.f);

  int  count()      const { return n_; }
  bool loadedReal() const { return loaded_real_; }

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

  bool loadPoints(const std::string& path);
  void synthesize(int n);   // procedural fallback galaxy
  void buildMesh();         // ONCE
};

}  // namespace wosw
