#pragma once
// Conductor — the autonomous "mind." It traverses the galaxy's k-NN graph LIVE on the
// primary: from the current node it scores the neighbors (edge weight x novelty x seeded
// noise) and samples the next. HESITATION is the (normalized) entropy of that choice
// distribution — high when several neighbors are equally plausible, so the machine
// visibly lingers. This is the genuine runtime decision the audience watches; it is NOT
// a baked path (suits the endless wander). Determinism: a single seed drives everything,
// so a capture is reproducible across runs and dome nodes.
#include "al/math/al_Vec.hpp"
#include <random>
#include <utility>
#include <vector>

namespace wosw {

struct Conductor {
  // Copies positions + adjacency by value (lifetime-safe). adjacency[i] = list of
  // (neighborIndex, weight) for node i.
  void init(const std::vector<al::Vec3f>& positions,
            const std::vector<std::vector<std::pair<int, float>>>& adjacency,
            unsigned seed = 1);

  void step(float dt);             // advance the walk

  al::Vec3f focusPos() const;      // smoothed interpolated fixation point (for the camera)
  int   curNode()    const { return cur_; }
  int   nextNode()   const { return next_; }
  float hesitation() const { return hesitation_; }
  float progress()   const { return t_; }
  long  visits()     const { return visits_; }   // nodes arrived at (drives vessel morph)

 private:
  std::vector<al::Vec3f>                            pos_;
  std::vector<std::vector<std::pair<int, float>>>  adj_;
  std::vector<float>                               visited_;   // recency, for novelty bias
  std::mt19937 rng_{1};
  int   cur_ = 0, next_ = 0;
  long  visits_ = 0;
  float t_ = 0.f;            // 0..1 progress from cur_ to next_
  float speed_ = 0.75f;      // base traversal speed (v2: faster so the walk visibly progresses)
  float hesitation_ = 0.f;

  void chooseNext();
};

}  // namespace wosw
