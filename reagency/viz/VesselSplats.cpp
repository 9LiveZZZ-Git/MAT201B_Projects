#include "viz/VesselSplats.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>

namespace wosw {

using namespace al;

// Additive Gaussian point-sprite with a tight core + soft halo (bloom), an overall
// alpha for the depth crossfade, and a per-vertex size factor (texCoord.x) so the melt
// can swell each splat.
static const char* kVert = R"GLSL(
#version 330
uniform mat4 al_ModelViewMatrix;
uniform mat4 al_ProjectionMatrix;
uniform float uPointSize;
layout (location = 0) in vec3 vertexPosition;
layout (location = 1) in vec4 vertexColor;
layout (location = 2) in vec2 vertexTexCoord;   // x = size factor
out vec4 vcol;
void main() {
  vcol = vertexColor;
  gl_Position = al_ProjectionMatrix * al_ModelViewMatrix * vec4(vertexPosition, 1.0);
  gl_PointSize = uPointSize * clamp(vertexTexCoord.x, 0.2, 3.0);
}
)GLSL";

static const char* kFrag = R"GLSL(
#version 330
in vec4 vcol;
out vec4 fragColor;
uniform float uCore;
uniform float uHalo;
uniform float uHaloStr;
uniform float uIntensity;
uniform float uAlpha;
void main() {
  vec2 c = gl_PointCoord * 2.0 - 1.0;
  float r2 = dot(c, c);
  if (r2 > 1.0) discard;
  float core = exp(-r2 / (2.0 * uCore * uCore));
  float halo = exp(-r2 / (2.0 * uHalo * uHalo));
  float i = core + uHaloStr * halo;
  fragColor = vec4(vcol.rgb * uIntensity, vcol.a * i * uAlpha);
}
)GLSL";

// --- 3D value noise (same family as SplatModel.cpp) for the smoky melt -------------
static float hash3(float x, float y, float z) {
  float h = std::sin(x * 12.9898f + y * 78.233f + z * 37.719f) * 43758.5453f;
  return h - std::floor(h);
}
static float vnoise(const Vec3f& p) {
  Vec3f i(std::floor(p.x), std::floor(p.y), std::floor(p.z));
  Vec3f f(p.x - i.x, p.y - i.y, p.z - i.z);
  Vec3f u(f.x*f.x*(3-2*f.x), f.y*f.y*(3-2*f.y), f.z*f.z*(3-2*f.z));
  auto L = [](float a, float b, float t){ return a + (b-a)*t; };
  float n000=hash3(i.x,i.y,i.z),     n100=hash3(i.x+1,i.y,i.z);
  float n010=hash3(i.x,i.y+1,i.z),   n110=hash3(i.x+1,i.y+1,i.z);
  float n001=hash3(i.x,i.y,i.z+1),   n101=hash3(i.x+1,i.y,i.z+1);
  float n011=hash3(i.x,i.y+1,i.z+1), n111=hash3(i.x+1,i.y+1,i.z+1);
  return L(L(L(n000,n100,u.x), L(n010,n110,u.x), u.y),
           L(L(n001,n101,u.x), L(n011,n111,u.x), u.y), u.z);
}

// --- dependency-free IEEE-754 binary16 -> float32 ---------------------------------
static float half2float(uint16_t h) {
  uint32_t sign = uint32_t(h & 0x8000) << 16;
  uint32_t exp  = (h >> 10) & 0x1F;
  uint32_t mant = h & 0x3FF;
  uint32_t f;
  if (exp == 0) {
    if (mant == 0) { f = sign; }
    else {                                  // subnormal
      exp = 1;
      while (!(mant & 0x400)) { mant <<= 1; --exp; }
      mant &= 0x3FF;
      f = sign | ((exp + (127 - 15)) << 23) | (mant << 13);
    }
  } else if (exp == 0x1F) {
    f = sign | 0x7F800000u | (mant << 13);  // inf / nan
  } else {
    f = sign | ((exp + (127 - 15)) << 23) | (mant << 13);
  }
  float out;
  std::memcpy(&out, &f, 4);
  return out;
}

bool VesselSplats::loadWSWV(const std::string& path) {
  FILE* f = std::fopen(path.c_str(), "rb");
  if (!f) return false;
  char magic[4] = {};
  int32_t ver = 0, K = 0, G = 0, stride = 0;
  bool ok = std::fread(magic, 1, 4, f) == 4
         && std::fread(&ver, 4, 1, f) == 1
         && std::fread(&K, 4, 1, f) == 1
         && std::fread(&G, 4, 1, f) == 1
         && std::fread(&stride, 4, 1, f) == 1;
  if (!ok || magic[0]!='W'||magic[1]!='S'||magic[2]!='W'||magic[3]!='V'
      || ver != 1 || K <= 0 || G <= 0 || stride != 8 || K > 4096 || G > 5000000) {
    std::fclose(f); return false;
  }
  float aabb[6];
  if (std::fread(aabb, 4, 6, f) != 6) { std::fclose(f); return false; }
  (void)aabb;                               // positions are pre-normalized; draw() scales
  kf_.assign(K, std::vector<Splat>(G));
  std::vector<uint16_t> row(stride);
  for (int k = 0; k < K; ++k) {
    for (int i = 0; i < G; ++i) {
      if (std::fread(row.data(), 2, stride, f) != size_t(stride)) { std::fclose(f); return false; }
      Splat s;
      s.pos     = Vec3f(half2float(row[0]), half2float(row[1]), half2float(row[2]));
      s.rgb     = Vec3f(half2float(row[3]), half2float(row[4]), half2float(row[5]));
      s.opacity = half2float(row[6]);
      s.sigma   = half2float(row[7]);
      kf_[k][i] = s;
    }
  }
  std::fclose(f);
  K_ = K; G_ = G;
  return true;
}

void VesselSplats::synthesize(int K, int G) {
  K_ = K; G_ = G;
  kf_.assign(K, std::vector<Splat>(G));
  for (int k = 0; k < K; ++k) {
    std::mt19937 rng(0x9E37u + uint32_t(k) * 2654435761u);
    std::uniform_real_distribution<float> uni(-1.f, 1.f);
    float hue = float(k) / float(std::max(1, K));
    Vec3f col(0.5f + 0.5f*std::cos(6.2831f*hue),
              0.5f + 0.5f*std::cos(6.2831f*(hue+0.33f)),
              0.5f + 0.5f*std::cos(6.2831f*(hue+0.66f)));
    for (int i = 0; i < G; ++i) {
      Vec3f d(uni(rng), uni(rng), uni(rng));
      float len = d.mag(); if (len < 1e-4f) { d = Vec3f(0,1,0); len = 1; }
      d /= len;
      float r = 0.6f + 0.4f * vnoise(d * 2.0f + Vec3f(float(k)*3.1f, 0, 0));
      Vec3f p = d * r;
      p += Vec3f(vnoise(p*3+Vec3f(5,0,0))-0.5f, vnoise(p*3+Vec3f(0,9,0))-0.5f,
                 vnoise(p*3+Vec3f(0,0,13))-0.5f) * 0.3f;
      Splat s;
      s.pos = p;
      s.rgb = col * (0.5f + 0.5f*r);
      s.opacity = 0.6f;
      s.sigma = 0.5f;
      kf_[k][i] = s;
    }
  }
}

void VesselSplats::update(float kfFloat, float time) {
  if (K_ == 0 || G_ == 0) return;
  int A = int(std::floor(kfFloat)) % K_; if (A < 0) A += K_;
  int B = (A + 1) % K_;
  float t  = kfFloat - std::floor(kfFloat);
  float ts = t * t * (3.f - 2.f * t);            // smoothstep ease for position
  float melt = 1.f - std::abs(2.f * t - 1.f);    // tent, peaks at t=0.5
  const auto& a = kf_[A];
  const auto& b = kf_[B];
  mesh_.reset();
  mesh_.primitive(Mesh::POINTS);
  for (int i = 0; i < G_; ++i) {
    Vec3f p = a[i].pos + (b[i].pos - a[i].pos) * ts;
    // smoky melt: curl-ish value-noise displacement, strongest mid-morph
    Vec3f q = p * 3.5f + Vec3f(0, 0, time * 0.5f);
    Vec3f disp(vnoise(q + Vec3f(11,0,0)) - 0.5f,
               vnoise(q + Vec3f(0,17,0)) - 0.5f,
               vnoise(q + Vec3f(0,0,23)) - 0.5f);
    p += disp * (melt * 0.35f);
    Vec3f col = a[i].rgb + (b[i].rgb - a[i].rgb) * t;
    float op  = (a[i].opacity + (b[i].opacity - a[i].opacity) * t) * (1.f - 0.5f * melt);
    float sg  = (a[i].sigma   + (b[i].sigma   - a[i].sigma)   * t) * (1.f + 1.2f * melt);
    mesh_.vertex(p.x, p.y, p.z);
    mesh_.color(col.x, col.y, col.z, op);
    mesh_.texCoord(std::min(3.f, sg / 0.5f), 0.f);   // size factor (melt swells the splats)
  }
  mesh_.update();
}

bool VesselSplats::init(const std::string& assetDir) {
  const std::string candidates[] = {
    assetDir + "/vessel.wswv",
    "assets/vessel.wswv",
    "../../assets/vessel.wswv",
    "reagency/assets/vessel.wswv",
    "MAT201B_Projects/reagency/assets/vessel.wswv",
  };
  loaded_real_ = false;
  for (const auto& c : candidates) {
    if (loadWSWV(c)) { loaded_real_ = true;
      std::fprintf(stderr, "[wosw] vessel: %d keyframes x %d gaussians from %s\n", K_, G_, c.c_str());
      break;
    }
  }
  if (!loaded_real_) {
    synthesize(4, 4000);
    std::fprintf(stderr, "[wosw] vessel.wswv not found — procedural vessel (%d x %d)\n", K_, G_);
  }
  mesh_.primitive(Mesh::POINTS);
  update(0.f, 0.f);
  shader_ok_ = shader_.compile(kVert, kFrag);
  if (!shader_ok_) std::fprintf(stderr, "[wosw] ERROR: vessel shader failed to compile\n");
  return shader_ok_;
}

void VesselSplats::draw(Graphics& g, const Vec3f& center, float scale, float alpha) {
  if (!shader_ok_ || alpha <= 0.001f) return;
  g.depthTesting(false);
  g.blending(true);
  g.blendAdd();
  g.shader(shader_);
  g.shader().uniform("uPointSize", point_size_);
  g.shader().uniform("uCore",      core_sigma_);
  g.shader().uniform("uHalo",      halo_sigma_);
  g.shader().uniform("uHaloStr",   halo_strength_);
  g.shader().uniform("uIntensity", intensity_);
  g.shader().uniform("uAlpha",     alpha);
  g.pushMatrix();
  g.translate(center.x, center.y, center.z);
  g.scale(scale);
  g.pointSize(point_size_);
  g.draw(mesh_);
  g.popMatrix();
  g.blendTrans();
  g.depthTesting(true);
}

}  // namespace wosw
