#pragma once
// StoryLayer — the THEM register, IN THE GALAXY (v2). Verified labor stories (one-liners with real
// attribution) placed at points in the cloud and billboarded facing the viewer, revealed by
// PROXIMITY: you read the ones you fly near. Renders on ALL dome nodes (static baked data + the
// synced camera) — no primary-only limit, unlike a head-locked caption. Baked by v2/bake_stories.py
// (stories_atlas.png + stories_pos.txt). allolib-only: custom GLSL through al::ShaderProgram.
#include "al/graphics/al_Graphics.hpp"
#include "al/graphics/al_Shader.hpp"
#include "al/graphics/al_Texture.hpp"
#include "al/graphics/al_VAOMesh.hpp"
#include "al/math/al_Vec.hpp"
#include <string>
#include <vector>

namespace wosw {

struct StoryLayer {
  bool init(const std::string& assetDir);
  bool ready() const { return shader_ok_ && tex_ok_; }
  int  count() const { return int(stories_.size()); }
  // Billboard every story near the camera (proximity-faded), facing the viewer. Safe on all nodes.
  void draw(al::Graphics& g, const al::Vec3f& camPos, const al::Vec3f& camRight,
            const al::Vec3f& camUp, float globalAlpha = 1.f);

 private:
  struct Story { al::Vec3f pos; int era, startRow, numLines; };
  std::vector<Story> stories_;

  al::Texture       tex_;
  al::ShaderProgram shader_;
  al::VAOMesh       mesh_;
  bool tex_ok_ = false, shader_ok_ = false;
  int  atlasW_ = 1, atlasH_ = 1, ch_ = 56, cw_ = 980, cols_ = 1;  // grid atlas (cw_ must match bake)

  bool loadAtlas(const std::string& path);
  void loadPos(const std::string& assetDir);
};

}  // namespace wosw
