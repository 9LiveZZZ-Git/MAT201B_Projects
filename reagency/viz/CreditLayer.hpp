#pragma once
// CreditLayer — the persistent THEM credit-envelope (v2 T8). The real, cited, harmed workers AI is
// built on (v2/them.txt, baked by v2/bake_credits.py into credits_atlas.png + credits_index.txt)
// are held OUTSIDE "we": their credits accrue in a fixed head-locked corner and NEVER fully fade
// (alpha floor ~0.12) for the rest of a session. This is the differential-persistence guard — the
// warm US/voice dissolves while THEM stays. (Necessary but NOT sufficient: only an honest guard
// PAIRED with the Act-V comfort-fail cap + the Act-IV collapse; alone it is a donor wall.)
//
// A galaxy node consumed on trace-expiry is only the TOKEN that lights a slot; the slot stores a
// WORKER INDEX cycling through them.txt, so all rows surface over a session. allolib-only: custom
// GLSL through al::ShaderProgram.
#include "al/graphics/al_Graphics.hpp"
#include "al/graphics/al_Shader.hpp"
#include "al/graphics/al_Texture.hpp"
#include "al/graphics/al_VAOMesh.hpp"
#include "al/math/al_Vec.hpp"
#include <string>
#include <vector>

namespace wosw {

struct CreditLayer {
  bool init(const std::string& assetDir);
  bool ready() const { return shader_ok_ && tex_ok_; }
  int  count() const { return int(credits_.size()); }   // # of them.txt worker rows

  // Draw up to nSlots live credit slots in a fixed head-locked corner (lower/upper left). For each
  // slot, slotWorker[i] is the worker-row index (or <0 for empty); slotAge[i] is seconds since the
  // slot was lit (fresh = bright, decays toward the floor by creditLife, never below 0.12).
  void draw(al::Graphics& g, const al::Vec3f& camPos, const al::Vec3f& camFwd,
            const al::Vec3f& camRight, const al::Vec3f& camUp,
            const int* slotWorker, const float* slotAge, int nSlots,
            float creditLife, float globalAlpha = 1.f);

 private:
  struct Credit { int startRow, numLines; };
  std::vector<Credit> credits_;

  al::Texture       tex_;
  al::ShaderProgram shader_;
  al::VAOMesh       mesh_;
  bool tex_ok_ = false, shader_ok_ = false;
  int  atlasW_ = 1, atlasH_ = 1, ch_ = 64;

  bool loadAtlas(const std::string& path);
  void loadIndex(const std::string& assetDir);
};

}  // namespace wosw
