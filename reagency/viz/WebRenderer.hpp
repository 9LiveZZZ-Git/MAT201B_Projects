#pragma once
// WebRenderer — the similarity "webs": the k-NN graph of the galaxy, i.e. "what the
// model thinks belongs together." Loads edges.bin (WSWE) from the factory (cosine
// k-NN in full CLIP space). If absent, builds a 3D k-NN fallback over the loaded
// points so it renders before real assets exist.
//
// Rendered as many LOW-ALPHA additive lines (al::Mesh::LINES) — the web structure
// emerges from accumulated overlap rather than per-edge logic (cutterkom). Built once.
#include "al/graphics/al_Graphics.hpp"
#include "al/graphics/al_Shader.hpp"
#include "al/graphics/al_VAOMesh.hpp"
#include "al/math/al_Vec.hpp"
#include <string>
#include <utility>
#include <vector>

namespace wosw {

struct WebRenderer {
  // positions/colors are the galaxy points (from ParticleField), indexed the same as
  // edges.bin node ids.
  bool init(const std::string& assetDir,
            const std::vector<al::Vec3f>& positions,
            const std::vector<al::Vec3f>& colors);

  // bright scales the additive line alpha (webs fade toward the vessel end of the
  // depth crossfade, where the galaxy structure recedes).
  void draw(al::Graphics& g, float bright = 1.f);

  // v2: the LIVE connection — drawn bright OVER the static web so the audience SEES the machine
  // connecting its points. The current node's k-NN star (what it links this node to), the committed
  // cur->next edge, and a hot traveling FRONT at the fixation head moving along that edge. Built
  // per-frame (a handful of segments); allolib-native, reuses the line shader. No CV at runtime —
  // this animates the PRE-BAKED CLIP similarity (the offline CV that built the web).
  void drawActive(al::Graphics& g, const al::Vec3f& cur, const al::Vec3f& next,
                  const al::Vec3f& head, const std::vector<al::Vec3f>& neighbors,
                  const std::vector<al::Vec3f>& trail, float bright);

  // A — "neurons firing in the web": a living electrical field. Many short-lived SPIKES travel
  // along the pre-baked k-NN edges everywhere (steady spontaneous background), biased DENSER around
  // curNode + its 1-ring (the machine's current attention). PURE FUNCTION of synced double simTime +
  // synced curNode + the identically-loaded adjacency (adj_): a hash of (edgeId, time-window) decides
  // which edges fire in each window and the spike progress is the fraction of elapsed time in that
  // window — every dome node computes the SAME firing set with NO new sync state and NO un-synced
  // randomness/clock. Bounded: a deterministic edge subset per window caps live spikes. Reuses the
  // additive blend path; adds a point-sprite firing pass. intensity scales spawn density + brightness.
  void drawFiring(al::Graphics& g, double simTime, int curNode, float bright,
                  float intensity = 1.f);

  int  edgeCount()  const { return n_edges_; }
  bool loadedReal() const { return loaded_real_; }

  // The graph the Conductor walks: adjacency[i] = list of (neighbor, weight) for node i.
  const std::vector<std::vector<std::pair<int, float>>>& adjacency() const { return adj_; }

 private:
  al::VAOMesh       mesh_;
  al::VAOMesh       active_;   // v2: small per-frame mesh for the live connection trace
  al::VAOMesh       firing_;   // A: small per-frame POINTS mesh for the neural-firing spikes
  al::ShaderProgram shader_;
  al::ShaderProgram fireShader_;   // A: additive point-sprite shader for spikes + node flashes
  bool              shader_ok_ = false;
  bool              fire_ok_   = false;
  void buildFireShader();          // A: compile fireShader_ on first use
  int  n_edges_     = 0;
  bool loaded_real_ = false;
  std::vector<std::vector<std::pair<int, float>>> adj_;   // graph for the Conductor
  std::vector<al::Vec3f> pos_;   // A: node world positions, for the firing pass (set at init)

  bool loadEdges(const std::string& path,
                 const std::vector<al::Vec3f>& P, const std::vector<al::Vec3f>& C);
  void knnFallback(const std::vector<al::Vec3f>& P, const std::vector<al::Vec3f>& C, int k);
  void addEdge(const al::Vec3f& a, const al::Vec3f& b,
               const al::Vec3f& ca, const al::Vec3f& cb, float w);
};

}  // namespace wosw
