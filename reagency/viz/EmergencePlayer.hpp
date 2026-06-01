#pragma once
// EmergencePlayer — the "watch it think" layer (v2 Phase-2). When the machine ATTENDS a dream node
// (the Conductor's current node is type==2), this animates that dream FORMING FROM NOISE — the
// decoded intermediate diffusion latents the Qwen run captured, baked by v2/bake_emergence.py into
// emergence_atlas.png + emergence_index.txt. Billboarded at the dream's node, facing the viewer.
// allolib-only: custom GLSL through al::ShaderProgram.
#include "al/graphics/al_Graphics.hpp"
#include "al/graphics/al_Shader.hpp"
#include "al/graphics/al_Texture.hpp"
#include "al/graphics/al_VAOMesh.hpp"
#include "al/math/al_Vec.hpp"
#include <string>
#include <unordered_map>

namespace wosw {

struct EmergencePlayer {
  bool init(const std::string& assetDir);
  bool ready() const { return shader_ok_ && tex_ok_; }
  bool has(int node) const { return seq_.find(node) != seq_.end(); }
  // Draw dream `node` forming at `pos`, at formation phase [0,1] (0 = first captured latent, 1 = formed).
  void draw(al::Graphics& g, const al::Vec3f& pos, const al::Vec3f& camRight,
            const al::Vec3f& camUp, int node, float phase, float alpha = 1.f);

 private:
  struct Seq { int start, num; };
  std::unordered_map<int, Seq> seq_;   // dream node -> [startCell, numFrames] in the atlas

  al::Texture       tex_;
  al::ShaderProgram shader_;
  al::VAOMesh       mesh_;
  bool tex_ok_ = false, shader_ok_ = false;
  int  atlasW_ = 1, atlasH_ = 1, cell_ = 160, cols_ = 20;   // must match bake_emergence.py CELL/COLS

  bool loadAtlas(const std::string& path);
  void loadIndex(const std::string& assetDir);
};

}  // namespace wosw
