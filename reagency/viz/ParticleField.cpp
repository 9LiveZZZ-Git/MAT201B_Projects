#include "viz/ParticleField.hpp"
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>

namespace wosw {

using namespace al;

// Additive Gaussian point-sprite (same sanctioned path as corvid/viz/SplatModel.cpp:
// custom GLSL through al::ShaderProgram, gl_PointSize from a uniform, radial falloff).
static const char* kVert = R"GLSL(
#version 330
uniform mat4 al_ModelViewMatrix;
uniform mat4 al_ProjectionMatrix;
uniform float uPointSize;
layout (location = 0) in vec3 vertexPosition;
layout (location = 1) in vec4 vertexColor;   // rgb + alpha
out vec4 vcol;
void main() {
  vcol = vertexColor;
  gl_Position = al_ProjectionMatrix * al_ModelViewMatrix * vec4(vertexPosition, 1.0);
  gl_PointSize = uPointSize;
}
)GLSL";

// Per-sprite "bloom": a tight bright CORE plus a soft wide HALO, summed. Dense
// regions bloom for free via additive overlap. No FBO / no post-process pass, so
// it composes with omni/dome rendering and costs only a little fragment work.
static const char* kFrag = R"GLSL(
#version 330
in vec4 vcol;
out vec4 fragColor;
uniform float uCore;       // bright-center sigma (small)
uniform float uHalo;       // glow sigma (wide)
uniform float uHaloStr;    // glow strength
uniform float uIntensity;  // overall brightness
void main() {
  vec2 c = gl_PointCoord * 2.0 - 1.0;
  float r2 = dot(c, c);
  if (r2 > 1.0) discard;
  float core = exp(-r2 / (2.0 * uCore * uCore));
  float halo = exp(-r2 / (2.0 * uHalo * uHalo));
  float i = core + uHaloStr * halo;
  fragColor = vec4(vcol.rgb * uIntensity, vcol.a * i);   // additive blend in draw()
}
)GLSL";

bool ParticleField::loadPoints(const std::string& path) {
  FILE* f = std::fopen(path.c_str(), "rb");
  if (!f) return false;
  char magic[4] = {};
  int32_t ver = 0, N = 0, stride = 0;
  bool ok = std::fread(magic, 1, 4, f) == 4
         && std::fread(&ver, 4, 1, f) == 1
         && std::fread(&N, 4, 1, f) == 1
         && std::fread(&stride, 4, 1, f) == 1;
  if (!ok || magic[0]!='W'||magic[1]!='S'||magic[2]!='W'||magic[3]!='P'
      || ver != 1 || stride != 10 || N <= 0 || N > 5000000) {
    std::fclose(f); return false;
  }
  pts_.clear();
  pts_.reserve(N);
  float r[10];
  for (int i = 0; i < N; ++i) {
    if (std::fread(r, 4, 10, f) != 10) break;
    P p;
    p.pos = Vec3f(r[0], r[1], r[2]);
    p.col = Vec3f(r[3], r[4], r[5]);
    p.density = r[6];
    p.cluster = int(r[7]);
    p.type    = int(r[8]);
    p.atlas   = int(r[9]);
    p.sigma   = 0.5f;
    pts_.push_back(p);
  }
  std::fclose(f);
  n_ = int(pts_.size());
  return n_ > 0;
}

void ParticleField::synthesize(int n) {
  // A procedural multi-cluster "galaxy" so the app is visibly alive before the
  // factory produces points.bin. Themed-ish palette (cool/warm clusters).
  pts_.clear();
  pts_.reserve(n);
  std::mt19937 rng(0x5EED);
  std::uniform_real_distribution<float> uni(0.f, 1.f);
  std::normal_distribution<float> gauss(0.f, 1.f);
  const int kClusters = 12;
  Vec3f centers[kClusters];
  Vec3f hues[kClusters];
  for (int c = 0; c < kClusters; ++c) {
    centers[c] = Vec3f(gauss(rng), gauss(rng), gauss(rng)) * 3.2f;
    float h = float(c) * 0.61803398875f;  // golden angle
    h -= std::floor(h);
    hues[c] = Vec3f(0.5f + 0.5f*std::cos(6.2831f*(h+0.0f)),
                    0.5f + 0.5f*std::cos(6.2831f*(h+0.33f)),
                    0.5f + 0.5f*std::cos(6.2831f*(h+0.66f)));
  }
  for (int i = 0; i < n; ++i) {
    int c = i % kClusters;
    Vec3f p = centers[c] + Vec3f(gauss(rng), gauss(rng), gauss(rng)) * 1.1f;
    P s;
    s.pos = p;
    s.density = std::max(0.f, std::min(1.f, 0.4f + 0.4f*gauss(rng)));
    s.col = hues[c] * (0.4f + 0.5f*s.density);
    s.cluster = c;
    s.type = (uni(rng) < 0.2f) ? 1 : 0;   // ~20% "words"
    if (s.type == 1) s.col = Vec3f(0.85f, 0.86f, 0.9f);  // words read pale
    s.sigma = 0.5f;
    pts_.push_back(s);
  }
  n_ = int(pts_.size());
}

void ParticleField::buildMesh() {
  mesh_.reset();
  mesh_.primitive(Mesh::POINTS);
  for (const auto& s : pts_) {
    float a = 0.30f + 0.65f * s.density;  // brighter base for AlloSphere washout
    mesh_.vertex(s.pos.x, s.pos.y, s.pos.z);
    mesh_.color(s.col.x, s.col.y, s.col.z, a);
    mesh_.texCoord(s.sigma, 0.f);
  }
  mesh_.update();
}

bool ParticleField::init(const std::string& assetDir) {
  const std::string candidates[] = {
    assetDir + "/points.bin",
    "assets/points.bin",
    "../../assets/points.bin",            // when launched from reagency/src/bin (run.sh)
    "reagency/assets/points.bin",
    "MAT201B_Projects/reagency/assets/points.bin",
  };
  loaded_real_ = false;
  for (const auto& c : candidates) {
    if (loadPoints(c)) { loaded_real_ = true;
      std::fprintf(stderr, "[wosw] loaded %d points from %s\n", n_, c.c_str());
      break;
    }
  }
  if (!loaded_real_) {
    synthesize(5000);
    std::fprintf(stderr, "[wosw] points.bin not found — procedural galaxy (%d points)\n", n_);
  }
  mesh_.primitive(Mesh::POINTS);
  buildMesh();
  shader_ok_ = shader_.compile(kVert, kFrag);
  if (!shader_ok_) std::fprintf(stderr, "[wosw] ERROR: particle shader failed to compile\n");
  return shader_ok_;
}

void ParticleField::draw(Graphics& g, float pointScale) {
  if (!shader_ok_) return;
  g.depthTesting(false);          // additive galaxy: order-independent, no depth write
  g.blending(true);
  g.blendAdd();
  g.shader(shader_);
  g.shader().uniform("uPointSize", point_size_ * pointScale);
  g.shader().uniform("uCore",      core_sigma_);
  g.shader().uniform("uHalo",      halo_sigma_);
  g.shader().uniform("uHaloStr",   halo_strength_);
  g.shader().uniform("uIntensity", intensity_);
  g.pointSize(point_size_ * pointScale);  // fallback for drivers ignoring gl_PointSize
  g.draw(mesh_);
  g.blendTrans();
  g.depthTesting(true);
}

}  // namespace wosw
