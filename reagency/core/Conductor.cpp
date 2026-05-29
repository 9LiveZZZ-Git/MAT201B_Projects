#include "core/Conductor.hpp"
#include <cmath>

namespace wosw {

using namespace al;

void Conductor::init(const std::vector<Vec3f>& positions,
                     const std::vector<std::vector<std::pair<int, float>>>& adjacency,
                     unsigned seed) {
  pos_ = positions;
  adj_ = adjacency;
  rng_.seed(seed);
  visited_.assign(pos_.size(), 0.f);
  if (pos_.empty()) { cur_ = next_ = 0; return; }
  cur_ = next_ = int(rng_() % pos_.size());
  chooseNext();
  t_ = 0.f;
}

void Conductor::chooseNext() {
  if (adj_.empty() || cur_ < 0 || cur_ >= int(adj_.size())) { hesitation_ = 0.f; return; }
  const auto& nbrs = adj_[cur_];
  std::uniform_real_distribution<float> uni(0.f, 1.f);
  if (nbrs.empty()) {                                  // dead end: jump elsewhere, fully unsure
    next_ = int(rng_() % pos_.size());
    hesitation_ = 1.f;
    return;
  }
  // score = weight x novelty x noise; novelty decays with how recently we visited a node.
  std::vector<float> score(nbrs.size());
  float sum = 0.f;
  for (size_t k = 0; k < nbrs.size(); ++k) {
    const int   j = nbrs[k].first;
    const float w = nbrs[k].second;
    const float novelty = 1.f / (1.f + visited_[j]);
    const float noise   = 0.7f + 0.6f * uni(rng_);
    score[k] = std::max(1e-4f, w * novelty * noise);
    sum += score[k];
  }
  // hesitation = normalized Shannon entropy of the choice distribution (1 = maximally torn).
  float H = 0.f;
  for (float s : score) { const float p = s / sum; if (p > 0.f) H -= p * std::log(p); }
  const float Hmax = std::log(float(nbrs.size()));
  hesitation_ = (Hmax > 1e-4f) ? (H / Hmax) : 0.f;
  // sample a neighbor proportionally to its score.
  float r = uni(rng_) * sum, acc = 0.f;
  next_ = nbrs[0].first;
  for (size_t k = 0; k < nbrs.size(); ++k) {
    acc += score[k];
    if (r <= acc) { next_ = nbrs[k].first; break; }
  }
}

void Conductor::step(float dt) {
  if (pos_.empty()) return;
  const float v = speed_ * (1.f - 0.6f * hesitation_);   // hesitation slows the approach
  t_ += v * dt;
  if (t_ >= 1.f) {
    t_ = 0.f;
    if (next_ >= 0 && next_ < int(visited_.size())) visited_[next_] += 1.f;
    for (auto& vv : visited_) vv *= 0.995f;              // slow decay so old nodes refresh
    cur_ = next_;
    chooseNext();
  }
}

Vec3f Conductor::focusPos() const {
  if (pos_.empty()) return Vec3f(0, 0, 0);
  const Vec3f& a = pos_[cur_];
  const Vec3f& b = pos_[next_];
  const float s = t_ * t_ * (3.f - 2.f * t_);            // smoothstep ease
  return a + (b - a) * s;
}

}  // namespace wosw
