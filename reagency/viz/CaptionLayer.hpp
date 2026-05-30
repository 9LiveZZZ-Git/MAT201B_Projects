#pragma once
// CaptionLayer — the machine's VOICE (v2). Renders the verified me/you/them/us confession lines
// (v2/captions_atlas.png + captions_index.txt, baked by v2/bake_captions.py) as a head-locked,
// legible caption in front of the viewer, with a TYPEWRITER reveal + cursor: the human-authorship
// mark (the line is typed at human rhythm, never glitched), so the staged "I" is never mistaken
// for sincere testimony. The person register (ME/YOU/THEM/US) is shown NON-HUE (vertical nudge).
// Schedule is act-gated: ME in Act II/III, THEM in III, YOU in IV, US in V; the SOLO line fills
// Act I (the fragment-viewer guarantee). Inert until the atlas exists.
//
// NOTE (dome): v1 drives the schedule on the local clock; on a single desktop node (== primary)
// that is exact. Multi-node dome sync of the caption state is a Phase-2 follow-up (put cur_/typed_
// into WoSWState). Allolib-only: custom GLSL through al::ShaderProgram.
#include "al/graphics/al_Graphics.hpp"
#include "al/graphics/al_Shader.hpp"
#include "al/graphics/al_Texture.hpp"
#include "al/graphics/al_VAOMesh.hpp"
#include "al/math/al_Vec.hpp"
#include <string>
#include <vector>

namespace wosw {

struct CaptionLayer {
  bool init(const std::string& assetDir);
  bool ready() const { return shader_ok_ && tex_ok_; }

  // Advance the schedule for the current act (1..5). Call once per frame on the primary.
  void update(int act, float dt);
  // Draw the active caption head-locked in front of the camera (lower third, facing the viewer).
  void draw(al::Graphics& g, const al::Vec3f& camPos, const al::Vec3f& camFwd,
            const al::Vec3f& camRight, const al::Vec3f& camUp);

 private:
  struct Line    { int caption, person, act, row, chars; float ink; };
  struct Caption { int person, act, totalChars; std::vector<int> lineIdx; };
  std::vector<Line>    lines_;
  std::vector<Caption> captions_;

  al::Texture       tex_;
  al::ShaderProgram shader_;
  al::VAOMesh       mesh_;
  bool tex_ok_ = false, shader_ok_ = false;
  int  atlasW_ = 1, atlasH_ = 1, ch_ = 64;

  // schedule state (primary clock)
  enum St { GAP, TYPING, HOLDING, FADING };
  St    st_ = GAP;
  int   cur_ = -1, curAct_ = 0, actCursor_ = 0;
  float typed_ = 0.f, phase_ = 0.f, alpha_ = 0.f, blinkClock_ = 0.f;

  int  pickNext(int act);
  bool loadAtlas(const std::string& path);
  void loadIndex(const std::string& assetDir);
};

}  // namespace wosw
