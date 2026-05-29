#pragma once
// HumanTrace — the thesis gesture, sphere edition. When the machine fixates an image, that
// photograph SURFACES at its node position in the surrounding galaxy (multiple can be alive
// at once), holds, then DISSOLVES — the photo disintegrates while a burst of grains (its own
// colour) streams OUTWARD INTO the galaxy point cloud. The visible human->machine "cede":
// human meaning handed to, and dispersed by, the machine. (The splat-vessel middle —
// photo->splat->galaxy — slots in here once vessel.wswv exists.)
//
// Loads atlas_0.png (8x8 grid of 256px hero thumbnails, stage_b). Grain motion is a pure
// function of (slot, grain index, age), so every dome node computes it identically from the
// synced trace ring — no per-grain broadcast. allolib-native (al::Texture + al::ShaderProgram).
#include "al/graphics/al_Graphics.hpp"
#include "al/graphics/al_Shader.hpp"
#include "al/graphics/al_Texture.hpp"
#include "al/graphics/al_VAOMesh.hpp"
#include "al/math/al_Vec.hpp"
#include <string>
#include <vector>

namespace wosw {

struct HumanTrace {
  bool init(const std::string& assetDir);
  bool ready() const { return shader_ok_ && tex_ok_; }

  // Draw ONE active trace: the photo for atlas `slot` as a camera-facing quad at `nodePos`,
  // fading + dissolving by `age` (0..lifetime), with grains streaming out into the galaxy.
  void draw(al::Graphics& g, const al::Vec3f& nodePos, const al::Vec3f& camRight,
            const al::Vec3f& camUp, int slot, float age, float lifetime);

 private:
  al::Texture       tex_;
  al::ShaderProgram photoShader_;   // textured quad + grain dissolve
  al::ShaderProgram grainShader_;   // additive point grains
  al::VAOMesh       quad_, grains_;
  bool tex_ok_ = false, shader_ok_ = false;
  int  grid_ = 8;                   // atlas is grid_ x grid_ thumbnails (stage_b: 2048/256 = 8)
  std::vector<al::Vec3f> slotAvg_;  // average colour per atlas slot (for the grains)

  bool loadAtlas(const std::string& path);
};

}  // namespace wosw
