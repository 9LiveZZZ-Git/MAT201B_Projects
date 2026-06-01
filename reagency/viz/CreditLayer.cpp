#include "viz/CreditLayer.hpp"
#include "al/graphics/al_Image.hpp"
#include <cstdio>

namespace wosw {

using namespace al;

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

// Pale, cool ink — THEM is held APART from the warm dissolving US/voice. Persistent (floor alpha
// set by the caller), never the warm sepia of the stories.
static const char* kFrag = R"GLSL(
#version 330
uniform sampler2D tex;
uniform float uAlpha;
uniform vec4  uUV;
in vec2 vuv;
out vec4 fragColor;
void main() {
  vec2 t = mix(uUV.xy, uUV.zw, vuv);
  vec4 c = texture(tex, t);
  vec3 tint = vec3(0.86, 0.90, 0.97);
  float a = c.a * uAlpha;
  if (a <= 0.003) discard;
  fragColor = vec4(c.rgb * tint, a);
}
)GLSL";

bool CreditLayer::loadAtlas(const std::string& path) {
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

void CreditLayer::loadIndex(const std::string& assetDir) {
  const std::string bases[] = { assetDir + "/../v2", "../../v2", "reagency/v2",
                                "MAT201B_Projects/reagency/v2", assetDir };
  for (const auto& b : bases) {
    FILE* f = std::fopen((b + "/credits_index.txt").c_str(), "r");
    if (!f) continue;
    char line[256];
    int cid, sr, nl;
    while (std::fgets(line, sizeof(line), f)) {
      if (line[0] == '#') continue;
      if (std::sscanf(line, "%d %d %d", &cid, &sr, &nl) == 3)
        credits_.push_back({sr, nl});
    }
    std::fclose(f);
    break;
  }
}

bool CreditLayer::init(const std::string& assetDir) {
  const std::string candidates[] = {
    assetDir + "/../v2/credits_atlas.png", "../../v2/credits_atlas.png",
    "reagency/v2/credits_atlas.png", "MAT201B_Projects/reagency/v2/credits_atlas.png",
    assetDir + "/credits_atlas.png",
  };
  tex_ok_ = false;
  for (const auto& c : candidates)
    if (loadAtlas(c)) { tex_ok_ = true;
      std::fprintf(stderr, "[wosw] credit atlas loaded from %s\n", c.c_str()); break; }
  if (tex_ok_) loadIndex(assetDir);
  else std::fprintf(stderr, "[wosw] credits_atlas.png not found — THEM credit-envelope off (bake v2/bake_credits.py)\n");
  shader_ok_ = shader_.compile(kVert, kFrag);
  std::fprintf(stderr, "[wosw] THEM credits: %zu worker rows\n", credits_.size());
  return ready();
}

void CreditLayer::draw(Graphics& g, const Vec3f& camPos, const Vec3f& camFwd,
                       const Vec3f& camRight, const Vec3f& camUp,
                       const int* slotWorker, const float* slotAge, int nSlots,
                       float creditLife, float globalAlpha) {
  if (!ready() || globalAlpha <= 0.003f || credits_.empty()) return;
  // a fixed head-locked stack down the upper-LEFT (clear of the lower-centre voice caption)
  const float DIST = 5.0f, CX = -2.15f, CWHALF = 1.15f, LINEH = 0.20f, SLOTGAP = 0.10f;
  g.depthTesting(false); g.blending(true); g.blendTrans();
  g.shader(shader_);
  tex_.bind(0);
  g.shader().uniform("tex", 0);
  const Vec3f rx = camRight * CWHALF, uy = camUp * (LINEH * 0.5f);

  float y = 1.15f;   // top of the stack; grows downward as slots/lines are drawn
  for (int sidx = 0; sidx < nSlots; ++sidx) {
    const int w = slotWorker[sidx];
    if (w < 0 || w >= int(credits_.size())) continue;
    const Credit& cr = credits_[w];
    // fresh = bright; decays toward the floor by creditLife; NEVER below ~0.12 (differential persistence)
    float k = 1.f - slotAge[sidx] / creditLife;
    k = k < 0.f ? 0.f : (k > 1.f ? 1.f : k);
    const float alpha = (0.12f + 0.88f * k) * globalAlpha;
    for (int i = 0; i < cr.numLines; ++i) {
      const int row = cr.startRow + i;
      const Vec3f c  = camPos + camFwd * DIST + camRight * CX + camUp * y;
      const Vec3f BL = c - rx - uy, BR = c + rx - uy, TL = c - rx + uy, TR = c + rx + uy;
      mesh_.reset(); mesh_.primitive(Mesh::TRIANGLE_STRIP);
      mesh_.vertex(BL.x, BL.y, BL.z); mesh_.texCoord(0.f, 1.f);
      mesh_.vertex(BR.x, BR.y, BR.z); mesh_.texCoord(1.f, 1.f);
      mesh_.vertex(TL.x, TL.y, TL.z); mesh_.texCoord(0.f, 0.f);
      mesh_.vertex(TR.x, TR.y, TR.z); mesh_.texCoord(1.f, 0.f);
      mesh_.update();
      const float v0 = 1.f - float((row + 1) * ch_) / atlasH_;
      const float v1 = 1.f - float(row * ch_) / atlasH_;
      g.shader().uniform("uUV", Vec4f(0.f, v0, 1.f, v1));
      g.shader().uniform("uAlpha", alpha);
      g.draw(mesh_);
      y -= LINEH;
    }
    y -= SLOTGAP;
  }
  tex_.unbind(0);
  g.depthTesting(true);
}

}  // namespace wosw
