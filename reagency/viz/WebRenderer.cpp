#include "viz/WebRenderer.hpp"
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <utility>

namespace wosw {

using namespace al;

// Low per-edge alpha: with additive blending, dense regions of the web brighten on
// their own as edges overlap (no per-edge opacity logic needed).
static constexpr float kEdgeAlpha = 0.06f;

void WebRenderer::addEdge(const Vec3f& a, const Vec3f& b,
                          const Vec3f& ca, const Vec3f& cb, float w) {
  float al = kEdgeAlpha * (0.4f + 0.6f * w);   // stronger similarity -> slightly brighter
  mesh_.vertex(a.x, a.y, a.z); mesh_.color(ca.x, ca.y, ca.z, al);
  mesh_.vertex(b.x, b.y, b.z); mesh_.color(cb.x, cb.y, cb.z, al);
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
      if (j > i) { addEdge(P[i], P[j], C[i], C[j], 0.6f); ++n_edges_; }  // undirected, once
    }
  }
}

bool WebRenderer::init(const std::string& assetDir,
                       const std::vector<Vec3f>& P, const std::vector<Vec3f>& C) {
  mesh_.reset();
  mesh_.primitive(Mesh::LINES);
  n_edges_ = 0; loaded_real_ = false;

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
  return true;
}

void WebRenderer::draw(Graphics& g) {
  if (n_edges_ == 0) return;
  g.depthTesting(false);
  g.blending(true);
  g.blendAdd();
  g.lineWidth(1.f);
  g.meshColor();          // per-vertex colors via allolib's built-in shader
  g.draw(mesh_);
  g.blendTrans();
  g.depthTesting(true);
}

}  // namespace wosw
