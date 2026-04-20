#pragma once
#include "al/math/al_Vec.hpp"
#include <cmath>
#include <vector>

// Uniform 3-D spatial hash over a toroidal world of side W.
// Cell size = neighborhood radius T → a 3^3 probe covers all candidates.
// Rebuild once per frame; query per agent → O(N) average total.
struct SpatialHash {
  int                           dim      = 1;
  double                        cellSize = 1.0;
  double                        worldHalf = 5.0;
  std::vector<std::vector<int>> grid;

  void rebuild(double T, double W) {
    cellSize  = T;
    worldHalf = W * 0.5;
    dim       = std::max(1, int(W / cellSize));
    grid.assign(dim * dim * dim, {});
  }

  void clear() {
    for (auto& c : grid) c.clear();
  }

  int key(int cx, int cy, int cz) const {
    cx = ((cx % dim) + dim) % dim;
    cy = ((cy % dim) + dim) % dim;
    cz = ((cz % dim) + dim) % dim;
    return cx + cy * dim + cz * dim * dim;
  }

  void insert(int idx, const al::Vec3d& pos) {
    int cx = int(std::floor((pos[0] + worldHalf) / cellSize));
    int cy = int(std::floor((pos[1] + worldHalf) / cellSize));
    int cz = int(std::floor((pos[2] + worldHalf) / cellSize));
    grid[key(cx, cy, cz)].push_back(idx);
  }

  void query(const al::Vec3d& pos, std::vector<int>& out) const {
    out.clear();
    int cx = int(std::floor((pos[0] + worldHalf) / cellSize));
    int cy = int(std::floor((pos[1] + worldHalf) / cellSize));
    int cz = int(std::floor((pos[2] + worldHalf) / cellSize));
    for (int dz = -1; dz <= 1; ++dz)
    for (int dy = -1; dy <= 1; ++dy)
    for (int dx = -1; dx <= 1; ++dx) {
      const auto& c = grid[key(cx+dx, cy+dy, cz+dz)];
      out.insert(out.end(), c.begin(), c.end());
    }
  }
};
