#pragma once
// DreamLayer — the diffusion DREAMS as billboards AT their galaxy nodes (v2 Phase-2). Each dream is
// a type=2 node the ML concatenated into the web (indices 50661..), with real CLIP edges to its
// archive neighbours. Here its image surfaces at that node, revealed by PROXIMITY as you fly near —
// so the dreams read as the machine's confabulations wired INTO the archive (the active-web trace
// lights their edges as the Conductor passes). Loads dream_atlas.png + dream_pos.txt (baked by
// v2/bake_dreams.py). allolib-only: custom GLSL through al::ShaderProgram.
#include "al/graphics/al_Graphics.hpp"
#include "al/graphics/al_Shader.hpp"
#include "al/graphics/al_Texture.hpp"
#include "al/graphics/al_VAOMesh.hpp"
#include "al/math/al_Vec.hpp"
#include <string>
#include <vector>

namespace wosw {

struct DreamLayer {
  bool init(const std::string& assetDir);
  bool ready() const { return shader_ok_ && tex_ok_; }
  int  count() const { return int(dreams_.size()); }
  // Billboard every dream near the camera (proximity-faded), facing the viewer. globalAlpha lets the
  // conductor/act state lift them (e.g. brighten in Act IV as the confabulation surfaces).
  void draw(al::Graphics& g, const al::Vec3f& camPos, const al::Vec3f& camRight,
            const al::Vec3f& camUp, float globalAlpha = 1.f, float time = 0.f);

 private:
  struct Dream { al::Vec3f pos; int col, row, op; };
  std::vector<Dream> dreams_;

  al::Texture       tex_;
  al::ShaderProgram shader_;
  al::VAOMesh       mesh_;
  bool tex_ok_ = false, shader_ok_ = false;
  int  atlasW_ = 1, atlasH_ = 1, cell_ = 384, cols_ = 6;   // must match bake_dreams.py CELL/COLS

  bool loadAtlas(const std::string& path);
  void loadPos(const std::string& assetDir);
};

}  // namespace wosw
