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

struct EmergencePlayer;   // fwd-decl: DreamLayer drives the per-dream forming overlay (see .cpp)

struct DreamLayer {
  bool init(const std::string& assetDir);
  bool ready() const { return shader_ok_ && tex_ok_; }
  int  count() const { return int(dreams_.size()); }
  al::Vec3f posAt(int i) const { return dreams_[i].pos; }   // for the autopilot dream-tour

  // The EmergencePlayer that owns the "forming from noise" atlas. When set, each dream that is
  // within the formation band is drawn FORMING (the emergence frames) and crossfades into its
  // resolved image — so all dreams "watch it think" by proximity, with zero new synced state.
  void setEmergence(EmergencePlayer* e) { emerge_ = e; }

  // Billboard the near dreams (proximity-faded + forming), facing the viewer. globalAlpha lets the
  // conductor/act state lift them (e.g. brighten in Act IV as the confabulation surfaces).
  void draw(al::Graphics& g, const al::Vec3f& camPos, const al::Vec3f& camRight,
            const al::Vec3f& camUp, float globalAlpha = 1.f, float time = 0.f);

 private:
  struct Dream { al::Vec3f pos; int node, col, row, op; };
  std::vector<Dream> dreams_;
  EmergencePlayer* emerge_ = nullptr;

  al::Texture       tex_;
  al::ShaderProgram shader_;
  al::VAOMesh       mesh_;
  bool tex_ok_ = false, shader_ok_ = false;
  int  atlasW_ = 1, atlasH_ = 1, cell_ = 384, cols_ = 14;  // cols auto-derived from atlas; match bake_dreams.py

  bool loadAtlas(const std::string& path);
  void loadPos(const std::string& assetDir);
};

}  // namespace wosw
