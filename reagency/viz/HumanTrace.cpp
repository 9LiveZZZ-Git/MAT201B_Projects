#include "viz/HumanTrace.hpp"
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

// Dissolve: as uAlpha drops, more grains fall away (discard), with a warm glow at the
// dissolving frontier — the photo disintegrates back into machine "grains".
static const char* kFrag = R"GLSL(
#version 330
uniform sampler2D tex;
uniform float uAlpha;
uniform vec4  uUV;        // (u0, v0, u1, v1) atlas sub-rect
in vec2 vuv;
out vec4 fragColor;
float hash(vec2 p){ return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453); }
void main() {
  vec2 t = mix(uUV.xy, uUV.zw, vuv);
  vec4 c = texture(tex, t);
  float n = hash(floor(vuv * 160.0));      // static grain field over the quad
  float thresh = 1.0 - uAlpha;             // dissolve frontier rises as alpha falls
  if (n < thresh) discard;                 // grains already dissolved away
  float edge = smoothstep(thresh, thresh + 0.10, n);
  vec3 rgb = mix(vec3(1.0, 0.85, 0.6), c.rgb, edge);   // warm glow at the frontier
  fragColor = vec4(rgb, c.a * uAlpha * 0.85);
}
)GLSL";

bool HumanTrace::loadAtlas(const std::string& path) {
  Image im;
  if (!im.load(path)) return false;
  auto& a = im.array();
  if (a.empty()) return false;
  tex_.create2D(im.width(), im.height(), Texture::RGBA8, Texture::RGBA, Texture::UBYTE);
  tex_.submit(&a[0], Texture::RGBA, Texture::UBYTE);
  tex_.filter(Texture::LINEAR);
  tex_.wrap(Texture::CLAMP_TO_EDGE, Texture::CLAMP_TO_EDGE, Texture::CLAMP_TO_EDGE);
  return true;
}

bool HumanTrace::init(const std::string& assetDir) {
  const std::string candidates[] = {
    assetDir + "/atlas_0.png",
    "assets/atlas_0.png",
    "../../assets/atlas_0.png",
    "reagency/assets/atlas_0.png",
    "MAT201B_Projects/reagency/assets/atlas_0.png",
  };
  tex_ok_ = false;
  for (const auto& c : candidates) {
    if (loadAtlas(c)) { tex_ok_ = true;
      std::fprintf(stderr, "[wosw] human-trace atlas loaded from %s\n", c.c_str());
      break;
    }
  }
  if (!tex_ok_)
    std::fprintf(stderr, "[wosw] atlas_0.png not found — human trace disabled until assets land\n");
  shader_ok_ = shader_.compile(kVert, kFrag);
  return ready();
}

void HumanTrace::draw(Graphics& g, const Vec3f& center, const Vec3f& right,
                      const Vec3f& up, int slot, float alpha, float size) {
  if (!ready() || slot < 0 || alpha <= 0.f) return;
  const float h = size * 0.5f;
  const Vec3f r = right * h, u = up * h;
  const Vec3f BL = center - r - u, BR = center + r - u,
              TL = center - r + u, TR = center + r + u;
  mesh_.reset();
  mesh_.primitive(Mesh::TRIANGLE_STRIP);
  mesh_.vertex(BL.x, BL.y, BL.z); mesh_.texCoord(0.f, 0.f);
  mesh_.vertex(BR.x, BR.y, BR.z); mesh_.texCoord(1.f, 0.f);
  mesh_.vertex(TL.x, TL.y, TL.z); mesh_.texCoord(0.f, 1.f);
  mesh_.vertex(TR.x, TR.y, TR.z); mesh_.texCoord(1.f, 1.f);
  mesh_.update();

  // atlas sub-rect for this slot (PIL packs top-left origin; flip v for GL sampling)
  const int col = slot % grid_, row = slot / grid_;
  const float u0 = float(col) / grid_, u1 = float(col + 1) / grid_;
  const float v0 = 1.f - float(row + 1) / grid_, v1 = 1.f - float(row) / grid_;

  g.depthTesting(false);
  g.blending(true);
  g.blendTrans();
  g.shader(shader_);
  tex_.bind(0);
  g.shader().uniform("tex", 0);
  g.shader().uniform("uAlpha", alpha);
  g.shader().uniform("uUV", Vec4f(u0, v0, u1, v1));
  g.draw(mesh_);
  tex_.unbind(0);
  g.depthTesting(true);
}

}  // namespace wosw
