#include "viz/StoryLayer.hpp"
#include "al/graphics/al_Image.hpp"
#include <cmath>
#include <cstdio>

namespace wosw {

using namespace al;

// proximity reveal: full text within NEAR, faded out by FAR (read what you fly past)
static constexpr float NEAR = 2.4f, FAR = 7.5f;  // full text held across a wide band so you can read the whole passage as you pass
static constexpr float WHALF = 0.20f, LINEH = 0.046f;  // world size of a story line-cell (smaller — whole passage fits in frame)

static const char* kVert = R"GLSL(
#version 330
uniform mat4 al_ModelViewMatrix;
uniform mat4 al_ProjectionMatrix;
layout (location = 0) in vec3 vertexPosition;
layout (location = 2) in vec2 vertexTexCoord;
out vec2 vuv;
void main() {
  vuv = vertexTexCoord;
  gl_Position = al_ProjectionMatrix * al_ModelViewMatrix * vec4(vertexPosition, 1.0);
}
)GLSL";

// warm archival ink; a subtle era tint (contemporary cool-white / historical sepia), NON-HUE-heavy.
static const char* kFrag = R"GLSL(
#version 330
uniform sampler2D tex;
uniform float uAlpha;
uniform vec4  uUV;
uniform float uEra;     // 0 contemporary, 1 historical
in vec2 vuv;
out vec4 fragColor;
void main() {
  vec2 t = mix(uUV.xy, uUV.zw, vuv);
  vec4 c = texture(tex, t);
  vec3 tint = mix(vec3(0.95, 0.96, 1.0), vec3(1.0, 0.92, 0.78), uEra);  // cool vs sepia
  float a = c.a * uAlpha;
  if (a <= 0.003) discard;
  fragColor = vec4(c.rgb * tint, a);
}
)GLSL";

bool StoryLayer::loadAtlas(const std::string& path) {
  Image im;
  if (!im.load(path)) return false;
  auto& a = im.array();
  if (a.empty()) return false;
  atlasW_ = int(im.width()); atlasH_ = int(im.height());
  tex_.create2D(atlasW_, atlasH_, Texture::RGBA8, Texture::RGBA, Texture::UBYTE);
  tex_.submit(&a[0], Texture::RGBA, Texture::UBYTE);
  tex_.filter(Texture::LINEAR);
  tex_.wrap(Texture::CLAMP_TO_EDGE, Texture::CLAMP_TO_EDGE, Texture::CLAMP_TO_EDGE);
  return true;
}

void StoryLayer::loadPos(const std::string& assetDir) {
  const std::string bases[] = { assetDir + "/../v2", "../../v2", "reagency/v2",
                                "MAT201B_Projects/reagency/v2", assetDir };
  for (const auto& b : bases) {
    FILE* f = std::fopen((b + "/stories_pos.txt").c_str(), "r");
    if (!f) continue;
    char line[256];
    int sid, era, sr, nl; float x, y, z;
    while (std::fgets(line, sizeof(line), f)) {
      if (line[0] == '#') continue;
      if (std::sscanf(line, "%d %d %f %f %f %d %d", &sid, &era, &x, &y, &z, &sr, &nl) == 7)
        stories_.push_back({Vec3f(x, y, z), era, sr, nl});
    }
    std::fclose(f);
    break;
  }
}

bool StoryLayer::init(const std::string& assetDir) {
  const std::string candidates[] = {
    assetDir + "/../v2/stories_atlas.png", "../../v2/stories_atlas.png",
    "reagency/v2/stories_atlas.png", "MAT201B_Projects/reagency/v2/stories_atlas.png",
    assetDir + "/stories_atlas.png",
  };
  tex_ok_ = false;
  for (const auto& c : candidates)
    if (loadAtlas(c)) { tex_ok_ = true;
      std::fprintf(stderr, "[wosw] story atlas loaded from %s\n", c.c_str()); break; }
  if (tex_ok_) { cols_ = atlasW_ / cw_; if (cols_ < 1) cols_ = 1; loadPos(assetDir); }
  else std::fprintf(stderr, "[wosw] stories_atlas.png not found — galaxy stories off (bake v2/bake_stories.py)\n");
  shader_ok_ = shader_.compile(kVert, kFrag);
  std::fprintf(stderr, "[wosw] galaxy stories: %zu in the cloud\n", stories_.size());
  return ready();
}

void StoryLayer::draw(Graphics& g, const Vec3f& camPos, const Vec3f& camRight,
                      const Vec3f& camUp, float globalAlpha) {
  if (!ready() || globalAlpha <= 0.003f) return;
  g.depthTesting(false); g.blending(true); g.blendTrans();
  g.shader(shader_);
  tex_.bind(0);
  g.shader().uniform("tex", 0);
  const Vec3f rx = camRight * WHALF, uy = camUp * (LINEH * 0.5f);

  for (const auto& s : stories_) {
    const float dist = (s.pos - camPos).mag();
    // smoothstep proximity reveal: 1 near, 0 far
    float a = (FAR - dist) / (FAR - NEAR);
    a = a < 0.f ? 0.f : (a > 1.f ? 1.f : a);
    a = a * a * (3.f - 2.f * a) * globalAlpha;
    if (a <= 0.01f) continue;

    g.shader().uniform("uEra", float(s.era));
    for (int i = 0; i < s.numLines; ++i) {
      const int idx = s.startRow + i;                  // flat cell index -> grid (col,row)
      const int col = idx % cols_, row = idx / cols_;
      // stack lines above the point (top line first), so the block reads downward
      const float yo = (float(s.numLines - 1) * 0.5f - float(i)) * LINEH;
      const Vec3f c  = s.pos + camUp * yo;
      const Vec3f BL = c - rx - uy, BR = c + rx - uy, TL = c - rx + uy, TR = c + rx + uy;
      mesh_.reset(); mesh_.primitive(Mesh::TRIANGLE_STRIP);
      mesh_.vertex(BL.x, BL.y, BL.z); mesh_.texCoord(0.f, 1.f);
      mesh_.vertex(BR.x, BR.y, BR.z); mesh_.texCoord(1.f, 1.f);
      mesh_.vertex(TL.x, TL.y, TL.z); mesh_.texCoord(0.f, 0.f);
      mesh_.vertex(TR.x, TR.y, TR.z); mesh_.texCoord(1.f, 0.f);
      mesh_.update();
      const float u0 = float(col * cw_) / atlasW_, u1 = float((col + 1) * cw_) / atlasW_;
      const float v0 = 1.f - float((row + 1) * ch_) / atlasH_;
      const float v1 = 1.f - float(row * ch_) / atlasH_;
      g.shader().uniform("uUV", Vec4f(u0, v0, u1, v1));
      g.shader().uniform("uAlpha", a);
      g.draw(mesh_);
    }
  }
  tex_.unbind(0);
  g.depthTesting(true);
}

}  // namespace wosw
