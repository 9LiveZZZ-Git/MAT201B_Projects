#pragma once
#include "core/Agent.hpp"
#include "al/math/al_Vec.hpp"
#include <vector>
#include <cmath>

namespace corvid {

struct Neighbor {
  int        idx;
  double     dist;
  al::Vec3d  delta; // toroidal displacement from self toward neighbor
};

// Shortest-path displacement a→b on a torus of side W.
inline al::Vec3d toroidalDelta(const al::Vec3d& a, const al::Vec3d& b, double W) {
  al::Vec3d d = b - a;
  const double half = W * 0.5;
  for (int k = 0; k < 3; ++k) {
    if      (d[k] >  half) d[k] -= W;
    else if (d[k] < -half) d[k] += W;
  }
  return d;
}

// Wrap position into (-W/2, W/2].
inline void wrapPos(al::Vec3d& p, double W) {
  const double half = W * 0.5;
  for (int k = 0; k < 3; ++k) {
    if      (p[k] >  half) p[k] -= W;
    else if (p[k] <= -half) p[k] += W;
  }
}

// Average forward vector of neighbors → alignment direction (unit).
inline al::Vec3d computeAlignment(const std::vector<Neighbor>& nbs,
                                  const Agent* agents) {
  al::Vec3d avg(0, 0, 0);
  for (const auto& nb : nbs) avg += agents[nb.idx].nav.uf();
  double m = avg.mag();
  return (m > 1e-8) ? avg / m : avg;
}

// Average toroidal displacement toward neighbor center → cohesion direction (unit).
inline al::Vec3d computeCohesion(const std::vector<Neighbor>& nbs) {
  al::Vec3d avg(0, 0, 0);
  for (const auto& nb : nbs) avg += nb.delta;
  avg /= double(nbs.size());
  double m = avg.mag();
  return (m > 1e-8) ? avg / m : avg;
}

// 1/d² repulsion from neighbors inside tSep → separation direction (unit).
inline al::Vec3d computeSeparation(const std::vector<Neighbor>& nbs, double tSep) {
  al::Vec3d force(0, 0, 0);
  for (const auto& nb : nbs) {
    if (nb.dist < tSep && nb.dist > 1e-8)
      force += (-nb.delta / nb.dist) / nb.dist;
  }
  double m = force.mag();
  return (m > 1e-8) ? force / m : force;
}

} // namespace corvid
