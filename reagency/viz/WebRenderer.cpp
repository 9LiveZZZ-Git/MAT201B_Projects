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
  pos_.assign(P.begin(), P.end());           // A: node positions for the firing pass

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

void WebRenderer::drawActive(Graphics& g, const Vec3f& cur, const Vec3f& next,
                             const Vec3f& head, const std::vector<Vec3f>& neighbors,
                             const std::vector<Vec3f>& trail, float bright) {
  if (!shader_ok_ || bright <= 0.f) return;
  active_.reset();
  active_.primitive(Mesh::LINES);

  // the WALK PATH: a glowing polyline back through the recently-visited nodes (newest = bright,
  // fading into the cloud) — the trace the machine left connecting its points. Drawn cyan-white.
  const int T = int(trail.size());
  for (int i = 0; i + 1 < T; ++i) {
    const float aN = 0.85f * (1.f - float(i)     / float(T));   // near end (newer)
    const float aF = 0.85f * (1.f - float(i + 1) / float(T));   // far end (older)
    active_.vertex(trail[i].x,   trail[i].y,   trail[i].z);   active_.color(0.65f, 0.90f, 1.00f, aN);
    active_.vertex(trail[i+1].x, trail[i+1].y, trail[i+1].z); active_.color(0.55f, 0.78f, 1.00f, aF);
  }

  // the current node's k-NN star: what the machine connects THIS point to (brightened)
  for (const auto& nb : neighbors) {
    active_.vertex(cur.x, cur.y, cur.z); active_.color(0.70f, 0.82f, 1.00f, 0.55f);
    active_.vertex(nb.x, nb.y, nb.z);    active_.color(0.70f, 0.82f, 1.00f, 0.10f);
  }
  // the committed connection cur->next, with a HOT travelling front at the fixation head.
  active_.vertex(cur.x, cur.y, cur.z);   active_.color(1.00f, 1.00f, 1.00f, 0.85f);
  active_.vertex(head.x, head.y, head.z);active_.color(1.00f, 1.00f, 1.00f, 1.00f);
  active_.vertex(head.x, head.y, head.z);active_.color(0.85f, 0.93f, 1.00f, 0.85f);
  active_.vertex(next.x, next.y, next.z);active_.color(0.55f, 0.70f, 1.00f, 0.30f);

  // a bright spark at the head (a 3-axis cross) so the moving front reads as a pulse
  const float s = 0.14f;
  auto spark = [&](const Vec3f& a, const Vec3f& b) {
    active_.vertex(a.x, a.y, a.z); active_.color(1.f, 1.f, 1.f, 0.0f);
    active_.vertex(b.x, b.y, b.z); active_.color(1.f, 1.f, 1.f, 1.0f);
  };
  spark(head - Vec3f(s, 0, 0), head + Vec3f(s, 0, 0));
  spark(head - Vec3f(0, s, 0), head + Vec3f(0, s, 0));
  spark(head - Vec3f(0, 0, s), head + Vec3f(0, 0, s));
  active_.update();

  g.depthTesting(false); g.blending(true); g.blendAdd();
  g.lineWidth(3.5f);                          // thick + bright so the walk reads at a glance
  g.shader(shader_);
  g.shader().uniform("uIntensity", bright);
  g.draw(active_);
  g.lineWidth(1.f);
  g.blendTrans(); g.depthTesting(true);
}

// ----- A: NEURONS FIRING IN THE WEB -----------------------------------------------------
// Additive Gaussian point-sprite shader for spikes (the traveling fire) AND node flashes
// (the arrival pop). Same sanctioned path as ParticleField/VesselSplats: custom GLSL through
// al::ShaderProgram, gl_PointSize from a uniform, radial falloff, texCoord at location 2. NO raw GL.
static const char* kFireVert = R"GLSL(
#version 330
uniform mat4 al_ModelViewMatrix;
uniform mat4 al_ProjectionMatrix;
uniform float uPointSize;
layout (location = 0) in vec3 vertexPosition;
layout (location = 1) in vec4 vertexColor;     // rgb + brightness(alpha)
layout (location = 2) in vec2 vertexTexCoord;  // x = size scale (flash>spike), y unused
out vec4 vcol;
void main() {
  vcol = vertexColor;
  gl_Position = al_ProjectionMatrix * al_ModelViewMatrix * vec4(vertexPosition, 1.0);
  gl_PointSize = uPointSize * vertexTexCoord.x;
}
)GLSL";
static const char* kFireFrag = R"GLSL(
#version 330
in vec4 vcol;
out vec4 fragColor;
uniform float uIntensity;
void main() {
  vec2 c = gl_PointCoord * 2.0 - 1.0;
  float r2 = dot(c, c);
  if (r2 > 1.0) discard;
  float core = exp(-r2 / (2.0 * 0.16 * 0.16));   // tight hot core
  float halo = exp(-r2 / (2.0 * 0.50 * 0.50));   // soft electric glow
  float i = core + 0.55 * halo;
  fragColor = vec4(vcol.rgb * uIntensity, vcol.a * i);   // additive blend in draw()
}
)GLSL";

void WebRenderer::buildFireShader() {
  fire_ok_ = fireShader_.compile(kFireVert, kFireFrag);
  if (!fire_ok_) std::fprintf(stderr, "[wosw] ERROR: firing shader failed to compile\n");
}

// Deterministic 32-bit integer hash (no state, identical on every node).
static inline uint32_t fhash(uint32_t x) {
  x ^= x >> 16; x *= 0x7feb352dU; x ^= x >> 15; x *= 0x846ca68bU; x ^= x >> 16;
  return x;
}
static inline float fhash01(uint32_t a, uint32_t b) {
  return float(fhash(a * 2654435761U ^ (b + 0x9e3779b9U))) * (1.f / 4294967296.f);
}

void WebRenderer::drawFiring(Graphics& g, double simTime, int curNode, float bright,
                            float intensity) {
  if (!fire_ok_) { buildFireShader(); if (!fire_ok_) return; }
  if (bright <= 0.f || intensity <= 0.f || adj_.empty()) return;

  // ---- determinism + budget knobs (all const; identical on every node) ----
  const double TICK = 1.0 / 20.0;          // firing time-window length (~20 windows/sec — slightly more churn)
  const int    N    = int(adj_.size());
  const uint32_t q  = uint32_t(std::floor(simTime / TICK));   // current window id (synced)
  const int    MAX_SPIKES = 2000;          // hard cap on live spikes (perf — still bounded)
  const int    MAX_FLASH  = 1000;          // hard cap on arrival node-flashes
  const float  pBack = 0.045f * intensity; // spontaneous background fire prob per directed edge/window (denser brain)
  const float  pNear = 0.80f  * intensity; // boosted fire prob for edges incident to curNode/1-ring
  const double LIFE  = 2.0 * TICK;         // spike flight time (windows it lives)

  firing_.reset();
  firing_.primitive(Mesh::POINTS);

  // The 1-ring boost: an edge's SOURCE is "hot" if it is curNode OR a neighbor of curNode. deg is
  // small (~7) so this membership test is O(deg) per source — cheap, stateless, identical per node.
  auto isHotSource = [&](int src) -> bool {
    if (src == curNode) return true;
    if (curNode < 0 || curNode >= N) return false;
    for (const auto& e : adj_[curNode]) if (e.first == src) return true;
    return false;
  };

  int spikes = 0, flashes = 0;
  const int stride = (N > 24000) ? 2 : 1;   // visit ~half the sources on huge graphs (hot kept)
  for (int i = 0; i < N; ++i) {
    const bool hot = isHotSource(i);
    if (!hot && stride > 1 && (i & 1)) continue;   // deterministic source subsample (hot always kept)
    if (spikes >= MAX_SPIKES) break;
    const auto& nbrs = adj_[i];
    const al::Vec3f a = pos_[i];
    const int deg = int(nbrs.size());
    for (int k = 0; k < deg; ++k) {
      const int j = nbrs[k].first;
      const float w = nbrs[k].second;
      const uint32_t edgeId = uint32_t(i) * 131071U + uint32_t(k);
      const float pBase = hot ? pNear : pBack;
      const float p = pBase * (0.5f + 0.5f * w);    // stronger similarity fires more
      for (int wq = 0; wq < 2; ++wq) {
        const uint32_t win = q - uint32_t(wq);
        if (fhash01(edgeId, win) >= p) continue;    // this edge did NOT fire in window 'win'
        const double born = double(win) * TICK;
        const double age  = simTime - born;
        if (age < 0.0 || age > LIFE) continue;
        float u = float(age / LIFE);                // 0..1 progress along the edge
        const al::Vec3f b = pos_[j];
        const al::Vec3f pP = a + (b - a) * u;       // spike position
        float env = (u < 0.18f) ? (u / 0.18f) : (1.f - (u - 0.18f) / 0.82f);
        env = env < 0.f ? 0.f : env;
        env *= env;
        const float br = (hot ? 1.0f : 0.6f) * env * (0.6f + 0.4f * w);
        const al::Vec3f col = hot ? al::Vec3f(0.80f, 0.95f, 1.00f)
                                  : al::Vec3f(0.45f, 0.72f, 1.00f);
        firing_.vertex(pP.x, pP.y, pP.z);
        firing_.color(col.x, col.y, col.z, br);
        firing_.texCoord(1.0f, 0.f);                // size scale 1.0 for a spike
        if (++spikes >= MAX_SPIKES) break;
        if (u > 0.80f && flashes < MAX_FLASH) {
          float fl = (u - 0.80f) / 0.20f;           // 0..1 in the arrival window
          fl = 1.f - fl;                            // pop then fade
          const float fb = (hot ? 1.0f : 0.7f) * fl;
          firing_.vertex(b.x, b.y, b.z);
          firing_.color(0.90f, 0.97f, 1.00f, fb);
          firing_.texCoord(2.2f, 0.f);              // bigger sprite for the node flash
          ++flashes;
        }
      }
      if (spikes >= MAX_SPIKES) break;
    }
  }
  if (spikes == 0) return;
  firing_.update();

  g.depthTesting(false);
  g.blending(true);
  g.blendAdd();
  g.shader(fireShader_);
  g.shader().uniform("uPointSize", 16.f);
  g.shader().uniform("uIntensity", bright);
  g.pointSize(16.f);                               // fallback for drivers ignoring gl_PointSize
  g.draw(firing_);
  g.blendTrans();
  g.depthTesting(true);
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
