#include "viz/WebRenderer.hpp"
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <utility>

namespace wosw {

using namespace al;

// Custom line shader (allolib-native ShaderProgram, same path as ParticleField) so the
// whole web can be faded by a single uniform for the depth crossfade. Per-vertex color
// (location 1) is preserved; uIntensity scales the additive alpha.
static const char* kVert = R"GLSL(
#version 330
uniform mat4 al_ModelViewMatrix;
uniform mat4 al_ProjectionMatrix;
layout (location = 0) in vec3 vertexPosition;
layout (location = 1) in vec4 vertexColor;
out vec4 vcol;
void main() {
  vcol = vertexColor;
  gl_Position = al_ProjectionMatrix * al_ModelViewMatrix * vec4(vertexPosition, 1.0);
}
)GLSL";

static const char* kFrag = R"GLSL(
#version 330
in vec4 vcol;
out vec4 fragColor;
uniform float uIntensity;
void main() { fragColor = vec4(vcol.rgb, vcol.a * uIntensity); }   // additive blend in draw()
)GLSL";

// Low per-edge alpha: with additive blending, dense regions of the web brighten on
// their own as edges overlap (no per-edge opacity logic needed).
static constexpr float kEdgeAlpha = 0.22f;     // was 0.06 — lines were invisible

// brighten an endpoint color so the (otherwise dim, cluster-colored) lines read clearly
static inline al::Vec3f edgeColor(const al::Vec3f& c) {
  return al::Vec3f(std::min(1.f, c.x * 1.5f + 0.12f),
                   std::min(1.f, c.y * 1.5f + 0.12f),
                   std::min(1.f, c.z * 1.5f + 0.12f));
}

void WebRenderer::addEdge(const Vec3f& a, const Vec3f& b,
                          const Vec3f& ca, const Vec3f& cb, float w) {
  float al = kEdgeAlpha * (0.5f + 0.5f * w);    // stronger similarity -> brighter
  Vec3f bca = edgeColor(ca), bcb = edgeColor(cb);
  mesh_.vertex(a.x, a.y, a.z); mesh_.color(bca.x, bca.y, bca.z, al);
  mesh_.vertex(b.x, b.y, b.z); mesh_.color(bcb.x, bcb.y, bcb.z, al);
}

// edges.bin "WSWE": [magic, int32 ver=1, int32 E] then E*(uint32 i, uint32 j, float w)
bool WebRenderer::loadEdges(const std::string& path,
                            const std::vector<Vec3f>& P, const std::vector<Vec3f>& C) {
  FILE* f = std::fopen(path.c_str(), "rb");
  if (!f) return false;
  char magic[4] = {};
  int32_t ver = 0, E = 0;
  bool ok = std::fread(magic, 1, 4, f) == 4
         && std::fread(&ver, 4, 1, f) == 1
         && std::fread(&E, 4, 1, f) == 1;
  if (!ok || magic[0]!='W'||magic[1]!='S'||magic[2]!='W'||magic[3]!='E' || ver != 1 || E <= 0) {
    std::fclose(f); return false;
  }
  const int N = int(P.size());
  for (int e = 0; e < E; ++e) {
    uint32_t i = 0, j = 0; float w = 0.f;
    if (std::fread(&i, 4, 1, f) != 1 || std::fread(&j, 4, 1, f) != 1
        || std::fread(&w, 4, 1, f) != 1) break;
    if (int(i) >= N || int(j) >= N) continue;   // edge references a point we don't have
    addEdge(P[i], P[j], C[i], C[j], w);
    adj_[i].push_back({int(j), w});           // adjacency for the Conductor (both directions)
    adj_[j].push_back({int(i), w});
    ++n_edges_;
  }
  std::fclose(f);
  return n_edges_ > 0;
}

// Dev fallback only: 3D k-NN over the (procedural) points so the webs render before
// the factory's cosine graph exists. Real assets always replace this.
void WebRenderer::knnFallback(const std::vector<Vec3f>& P, const std::vector<Vec3f>& C, int k) {
  const int N = int(P.size());
  const int step = (N > 8000) ? (N / 8000) : 1;   // cap work for very large N
  std::vector<std::pair<float, int>> d;
  for (int i = 0; i < N; i += step) {
    d.clear(); d.reserve(N);
    for (int j = 0; j < N; ++j) {
      if (j == i) continue;
      d.emplace_back((P[i] - P[j]).magSqr(), j);
    }
    const int kk = std::min(k, int(d.size()));
    std::partial_sort(d.begin(), d.begin() + kk, d.end());
    for (int n = 0; n < kk; ++n) {
      const int j = d[n].second;
      adj_[i].push_back({j, 0.6f});                                      // adjacency (directed)
      if (j > i) { addEdge(P[i], P[j], C[i], C[j], 0.6f); ++n_edges_; }  // mesh (undirected once)
    }
  }
}

bool WebRenderer::init(const std::string& assetDir,
                       const std::vector<Vec3f>& P, const std::vector<Vec3f>& C) {
  mesh_.reset();
  mesh_.primitive(Mesh::LINES);
  n_edges_ = 0; loaded_real_ = false;
  adj_.assign(P.size(), {});

  const std::string candidates[] = {
    assetDir + "/edges.bin",
    "assets/edges.bin",
    "../../assets/edges.bin",            // launched from reagency/src/bin (run.sh)
    "reagency/assets/edges.bin",
    "MAT201B_Projects/reagency/assets/edges.bin",
  };
  for (const auto& c : candidates) {
    if (loadEdges(c, P, C)) {
      loaded_real_ = true;
      std::fprintf(stderr, "[wosw] loaded %d web edges from %s\n", n_edges_, c.c_str());
      break;
    }
  }
  if (!loaded_real_ && !P.empty()) {
    knnFallback(P, C, 8);
    std::fprintf(stderr, "[wosw] edges.bin not found — 3D k-NN fallback (%d edges)\n", n_edges_);
  }
  mesh_.update();
  shader_ok_ = shader_.compile(kVert, kFrag);
  if (!shader_ok_) std::fprintf(stderr, "[wosw] ERROR: web shader failed to compile\n");
  return true;
}

void WebRenderer::draw(Graphics& g, float bright) {
  if (n_edges_ == 0 || !shader_ok_ || bright <= 0.f) return;
  g.depthTesting(false);
  g.blending(true);
  g.blendAdd();
  g.lineWidth(1.f);
  g.shader(shader_);
  g.shader().uniform("uIntensity", bright);
  g.draw(mesh_);
  g.blendTrans();
  g.depthTesting(true);
}

}  // namespace wosw
