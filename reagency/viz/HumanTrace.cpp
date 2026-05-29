#include "viz/HumanTrace.hpp"
#include "al/graphics/al_Image.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>

namespace wosw {

using namespace al;

// --- photo billboard with a grain-threshold dissolve ------------------------------
static const char* kPhotoVert = R"GLSL(
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

static const char* kPhotoFrag = R"GLSL(
#version 330
uniform sampler2D tex;
uniform float uAlpha;
uniform vec4  uUV;
in vec2 vuv;
out vec4 fragColor;
float hash(vec2 p){ return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453); }
void main() {
  vec2 t = mix(uUV.xy, uUV.zw, vuv);
  vec4 c = texture(tex, t);
  float n = hash(floor(vuv * 160.0));
  float thresh = 1.0 - uAlpha;
  if (n < thresh) discard;                 // grains already dissolved away
  float edge = smoothstep(thresh, thresh + 0.10, n);
  vec3 rgb = mix(vec3(1.0, 0.85, 0.6), c.rgb, edge);
  fragColor = vec4(rgb, c.a * uAlpha * 0.9);
}
)GLSL";

// --- additive point grains (the photo streaming into the galaxy) ------------------
static const char* kGrainVert = R"GLSL(
#version 330
uniform mat4 al_ModelViewMatrix;
uniform mat4 al_ProjectionMatrix;
uniform float uPointSize;
layout (location = 0) in vec3 vertexPosition;
layout (location = 1) in vec4 vertexColor;
out vec4 vcol;
void main() {
  vcol = vertexColor;
  gl_Position = al_ProjectionMatrix * al_ModelViewMatrix * vec4(vertexPosition, 1.0);
  gl_PointSize = uPointSize;
}
)GLSL";

static const char* kGrainFrag = R"GLSL(
#version 330
in vec4 vcol;
out vec4 fragColor;
void main() {
  vec2 c = gl_PointCoord * 2.0 - 1.0;
  float r2 = dot(c, c);
  if (r2 > 1.0) discard;
  fragColor = vec4(vcol.rgb, vcol.a * exp(-r2 * 2.5));
}
)GLSL";

static float hashf(int n) {
  float x = std::sin(float(n) * 12.9898f) * 43758.5453f;
  return x - std::floor(x);
}

bool HumanTrace::loadAtlas(const std::string& path) {
  Image im;
  if (!im.load(path)) return false;
  auto& a = im.array();
  if (a.empty()) return false;
  const int w = int(im.width()), h = int(im.height());
  tex_.create2D(w, h, Texture::RGBA8, Texture::RGBA, Texture::UBYTE);
  tex_.submit(&a[0], Texture::RGBA, Texture::UBYTE);
  tex_.filter(Texture::LINEAR);
  tex_.wrap(Texture::CLAMP_TO_EDGE, Texture::CLAMP_TO_EDGE, Texture::CLAMP_TO_EDGE);

  // average colour of each atlas slot (used to tint that trace's grains)
  const int cw = w / grid_, ch = h / grid_;
  slotAvg_.assign(grid_ * grid_, Vec3f(0.7f, 0.7f, 0.7f));
  for (int s = 0; s < grid_ * grid_; ++s) {
    const int col = s % grid_, row = s / grid_;
    double sr = 0, sg = 0, sb = 0; long n = 0;
    for (int y = row * ch; y < (row + 1) * ch; y += 8)
      for (int x = col * cw; x < (col + 1) * cw; x += 8) {
        const size_t i = (size_t(y) * w + x) * 4;
        sr += a[i]; sg += a[i + 1]; sb += a[i + 2]; ++n;
      }
    if (n) slotAvg_[s] = Vec3f(float(sr / n) / 255.f, float(sg / n) / 255.f, float(sb / n) / 255.f);
  }
  return true;
}

bool HumanTrace::init(const std::string& assetDir) {
  const std::string candidates[] = {
    assetDir + "/atlas_0.png", "assets/atlas_0.png", "../../assets/atlas_0.png",
    "reagency/assets/atlas_0.png", "MAT201B_Projects/reagency/assets/atlas_0.png",
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
  shader_ok_ = photoShader_.compile(kPhotoVert, kPhotoFrag)
            && grainShader_.compile(kGrainVert, kGrainFrag);
  return ready();
}

void HumanTrace::draw(Graphics& g, const Vec3f& nodePos, const Vec3f& camRight,
                      const Vec3f& camUp, int slot, float age, float lifetime) {
  if (!ready() || slot < 0) return;
  const float t01 = (lifetime > 0.f) ? std::min(1.f, age / lifetime) : 1.f;

  // photo envelope: fade in fast, hold, then dissolve out over the back half
  float photoA = (t01 < 0.12f) ? (t01 / 0.12f)
               : (t01 < 0.5f) ? 1.f : (1.f - (t01 - 0.5f) / 0.5f);
  if (photoA < 0.f) photoA = 0.f;

  // --- the photo, camera-facing, at its node ---
  if (photoA > 0.003f) {
    const float h = 1.2f;     // ~2.4-unit quad (readable from across the cloud)
    const Vec3f r = camRight * h, u = camUp * h;
    const Vec3f BL = nodePos - r - u, BR = nodePos + r - u,
                TL = nodePos - r + u, TR = nodePos + r + u;
    quad_.reset();
    quad_.primitive(Mesh::TRIANGLE_STRIP);
    quad_.vertex(BL.x, BL.y, BL.z); quad_.texCoord(0.f, 1.f);   // v flipped (images were upside down)
    quad_.vertex(BR.x, BR.y, BR.z); quad_.texCoord(1.f, 1.f);
    quad_.vertex(TL.x, TL.y, TL.z); quad_.texCoord(0.f, 0.f);
    quad_.vertex(TR.x, TR.y, TR.z); quad_.texCoord(1.f, 0.f);
    quad_.update();
    const int col = slot % grid_, row = slot / grid_;
    const float u0 = float(col) / grid_, u1 = float(col + 1) / grid_;
    const float v0 = 1.f - float(row + 1) / grid_, v1 = 1.f - float(row) / grid_;
    g.depthTesting(false);
    g.blending(true);
    g.blendTrans();
    g.shader(photoShader_);
    tex_.bind(0);
    g.shader().uniform("tex", 0);
    g.shader().uniform("uAlpha", photoA);
    g.shader().uniform("uUV", Vec4f(u0, v0, u1, v1));
    g.draw(quad_);
    tex_.unbind(0);
  }

  // --- grains streaming outward into the galaxy (the photo dispersing) ---
  float gp = (t01 - 0.30f) / 0.70f;   // emerge during the dissolve
  if (gp < 0.f) gp = 0.f; if (gp > 1.f) gp = 1.f;
  if (gp > 0.003f) {
    Vec3f avg = (slot < int(slotAvg_.size())) ? slotAvg_[slot] : Vec3f(1.f, 0.85f, 0.6f);
    avg *= 1.5f;   // grains are additive; brighten
    const int NG = 300; const float SPREAD = 6.5f;
    grains_.reset();
    grains_.primitive(Mesh::POINTS);
    for (int i = 0; i < NG; ++i) {
      const float h1 = hashf(slot * 977 + i * 131);
      const float h2 = hashf(slot * 331 + i * 17 + 5);
      const float h3 = hashf(slot * 53 + i * 7 + 3);
      const float th = h1 * 6.2831853f, z = h2 * 2.f - 1.f, rr = std::sqrt(std::max(0.f, 1.f - z * z));
      const Vec3f dir(rr * std::cos(th), rr * std::sin(th), z);
      const Vec3f p = nodePos + dir * (gp * SPREAD * (0.4f + 0.8f * h3));
      const float ga = gp * (1.3f - gp) * (0.5f + 0.5f * h3);   // peak mid-flight, fade at end
      grains_.vertex(p.x, p.y, p.z);
      grains_.color(avg.x, avg.y, avg.z, ga);
    }
    grains_.update();
    g.depthTesting(false);
    g.blending(true);
    g.blendAdd();
    g.shader(grainShader_);
    g.shader().uniform("uPointSize", 8.f);
    g.pointSize(8.f);
    g.draw(grains_);
    g.blendTrans();
    g.depthTesting(true);
  }
}

}  // namespace wosw
