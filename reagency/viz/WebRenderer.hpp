#pragma once
// WebRenderer — the similarity "webs": the k-NN graph of the galaxy, i.e. "what the
// model thinks belongs together." Loads edges.bin (WSWE) from the factory (cosine
// k-NN in full CLIP space). If absent, builds a 3D k-NN fallback over the loaded
// points so it renders before real assets exist.
//
// Rendered as many LOW-ALPHA additive lines (al::Mesh::LINES) — the web structure
// emerges from accumulated overlap rather than per-edge logic (cutterkom). Built once.
#include "al/graphics/al_Graphics.hpp"
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

  void draw(al::Graphics& g);

  int  edgeCount()  const { return n_edges_; }
  bool loadedReal() const { return loaded_real_; }

  // The graph the Conductor walks: adjacency[i] = list of (neighbor, weight) for node i.
  const std::vector<std::vector<std::pair<int, float>>>& adjacency() const { return adj_; }

 private:
  al::VAOMesh mesh_;
  int  n_edges_     = 0;
  bool loaded_real_ = false;
  std::vector<std::vector<std::pair<int, float>>> adj_;   // graph for the Conductor

  bool loadEdges(const std::string& path,
                 const std::vector<al::Vec3f>& P, const std::vector<al::Vec3f>& C);
  void knnFallback(const std::vector<al::Vec3f>& P, const std::vector<al::Vec3f>& C, int k);
  void addEdge(const al::Vec3f& a, const al::Vec3f& b,
               const al::Vec3f& ca, const al::Vec3f& cb, float w);
};

}  // namespace wosw
