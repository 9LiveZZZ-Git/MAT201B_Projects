#pragma once
// HumanTrace — the thesis gesture. When the machine fixates an image, that image's actual
// photograph SURFACES in front of the camera, holds, then DISSOLVES INTO GRAINS as the
// wander moves on: the visible human->machine "cede" (human meaning handed to, and consumed
// by, the machine).
//
// Loads atlas_0.png (an 8x8 grid of 256px hero thumbnails packed by stage_b_layout.py). The
// fixated node's atlas slot (from points.bin) selects the sub-rect; the dissolve is a
// per-pixel grain threshold driven by the alpha envelope, with a warm glow at the frontier.
// allolib-native: al::Texture + a custom al::ShaderProgram quad (no raw GL).
#include "al/graphics/al_Graphics.hpp"
#include "al/graphics/al_Shader.hpp"
#include "al/graphics/al_Texture.hpp"
#include "al/graphics/al_VAOMesh.hpp"
#include "al/math/al_Vec.hpp"
#include <string>

namespace wosw {

struct HumanTrace {
  bool init(const std::string& assetDir);
  bool ready() const { return shader_ok_ && tex_ok_; }

  // Draw the photo for atlas `slot` as a camera-facing quad at `center` (use the camera's
  // right/up axes), faded + dissolved by `alpha` (1 = surfaced, 0 = fully grains/gone).
  void draw(al::Graphics& g, const al::Vec3f& center, const al::Vec3f& right,
            const al::Vec3f& up, int slot, float alpha, float size);

 private:
  al::Texture       tex_;
  al::ShaderProgram shader_;
  al::VAOMesh       mesh_;
  bool tex_ok_ = false, shader_ok_ = false;
  int  grid_ = 8;   // atlas is grid_ x grid_ thumbnails (stage_b: 2048/256 = 8)

  bool loadAtlas(const std::string& path);
};

}  // namespace wosw
