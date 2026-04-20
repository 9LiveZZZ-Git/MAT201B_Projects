// p3.cpp — Reynolds Flocking with Wraparound
// Alignment, separation, cohesion. Toroidal world. Uniform-grid spatial hash.

#include "al/app/al_App.hpp"
#include "al/graphics/al_Shapes.hpp"
#include "al/io/al_ControlNav.hpp"
#include "al/math/al_Random.hpp"
#include "al/ui/al_ControlGUI.hpp"
#include "al/ui/al_Parameter.hpp"

#include <algorithm>
#include <vector>

using namespace al;

const int    N = 200;
const double W = 10.0;  // world cube side; wrap uses this

// ---------------------------------------------------------------
// Spatial hash: uniform 3-D grid over the toroidal world.
// Cell size = neighborhood radius T so a 3^3 probe covers all
// candidates within T. Rebuild once per frame, query per agent.
// ---------------------------------------------------------------
struct SpatialHash {
  int                          dim = 1;
  double                       cellSize = 1.0;
  std::vector<std::vector<int>> grid;

  // Call when T (and therefore cellSize) may have changed.
  void rebuild(double t) {
    cellSize   = t;
    dim        = std::max(1, int(W / cellSize));
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

  void insert(int idx, const Vec3d& pos) {
    const double half = W * 0.5;
    int cx = int(floor((pos[0] + half) / cellSize));
    int cy = int(floor((pos[1] + half) / cellSize));
    int cz = int(floor((pos[2] + half) / cellSize));
    grid[key(cx, cy, cz)].push_back(idx);
  }

  // Returns all agent indices in the 3^3 surrounding cells.
  void query(const Vec3d& pos, std::vector<int>& out) const {
    out.clear();
    const double half = W * 0.5;
    int cx = int(floor((pos[0] + half) / cellSize));
    int cy = int(floor((pos[1] + half) / cellSize));
    int cz = int(floor((pos[2] + half) / cellSize));
    for (int dz = -1; dz <= 1; ++dz)
    for (int dy = -1; dy <= 1; ++dy)
    for (int dx = -1; dx <= 1; ++dx) {
      const auto& c = grid[key(cx + dx, cy + dy, cz + dz)];
      out.insert(out.end(), c.begin(), c.end());
    }
  }
};

// ---------------------------------------------------------------
// Toroidal helpers
// ---------------------------------------------------------------
static Vec3d toroidalDelta(const Vec3d& a, const Vec3d& b) {
  Vec3d d    = b - a;
  const double half = W * 0.5;
  for (int k = 0; k < 3; ++k) {
    if      (d[k] >  half) d[k] -= W;
    else if (d[k] < -half) d[k] += W;
  }
  return d;
}

static void wrapPos(Vec3d& p) {
  const double half = W * 0.5;
  for (int k = 0; k < 3; ++k) {
    if      (p[k] >  half) p[k] -= W;
    else if (p[k] <= -half) p[k] += W;
  }
}

// ---------------------------------------------------------------
struct Agent {
  Nav nav;
  int id = -1;
};

// ---------------------------------------------------------------
struct MyApp : App {
  std::vector<Agent> agents;
  Mesh               tetra;
  ControlGUI         gui;
  SpatialHash        hash;
  double             prevT = -1.0;  // detect T changes for hash rebuild

  // --- tunables ---
  Parameter     turnRate   {"Turn Rate",      "Agents", 0.05f,  0.001f,  0.3f};
  Parameter     moveSpeed  {"Move Speed",     "Agents", 2.0f,   0.1f,   10.0f};
  Parameter     tNeighbor  {"Neighbor Radius","Agents", 1.0f,   0.1f,    4.0f};
  Parameter     tSep       {"Sep Threshold",  "Agents", 0.35f,  0.02f,   2.0f};
  ParameterInt  kNeighbors {"K Neighbors",    "Agents", 8,      1,       30};
  Parameter     alignW     {"Align Weight",   "Agents", 1.0f,   0.0f,    5.0f};
  Parameter     cohesionW  {"Cohesion Weight","Agents", 0.8f,   0.0f,    5.0f};
  Parameter     sepW       {"Sep Weight",     "Agents", 2.5f,   0.0f,   10.0f};

  // ----------------------------------------------------------------
  void initAgents() {
    agents.resize(N);
    const double half = W * 0.5;
    for (int i = 0; i < N; ++i) {
      agents[i].id = i;
      agents[i].nav.pos(rnd::uniformS() * half,
                        rnd::uniformS() * half,
                        rnd::uniformS() * half);
      agents[i].nav.quat(Quatd().fromEuler(
        rnd::uniform(0.0, M_2PI),
        rnd::uniform(0.0, M_2PI),
        rnd::uniform(0.0, M_2PI)));
    }
  }

  void onCreate() override {
    nav().pos(0, 0, W * 1.5);
    nav().faceToward(Vec3d(0, 0, 0));

    addTetrahedron(tetra, 0.12f);
    tetra.generateNormals();

    gui << turnRate << moveSpeed
        << tNeighbor << tSep << kNeighbors
        << alignW << cohesionW << sepW;
    gui.init();

    initAgents();
    hash.rebuild((double)tNeighbor);
    prevT = (double)tNeighbor;
  }

  void onAnimate(double dt) override {
    const double T = (double)tNeighbor;
    const double S = (double)tSep;
    const int    K = (int)kNeighbors;

    // Rebuild hash only when T has changed
    if (T != prevT) { hash.rebuild(T); prevT = T; }
    hash.clear();
    for (int i = 0; i < N; ++i) hash.insert(i, agents[i].nav.pos());

    std::vector<int> candidates;
    candidates.reserve(64);

    struct Neighbor { int idx; double dist; Vec3d delta; };
    std::vector<Neighbor> nbuf;
    nbuf.reserve(64);

    for (int i = 0; i < N; ++i) {
      Nav&         self    = agents[i].nav;
      const Vec3d& selfPos = self.pos();

      // --- spatial-hash query, then toroidal distance filter ---
      hash.query(selfPos, candidates);
      nbuf.clear();
      for (int j : candidates) {
        if (j == i) continue;
        Vec3d  delta = toroidalDelta(selfPos, agents[j].nav.pos());
        double dist  = delta.mag();
        if (dist < T) nbuf.push_back({j, dist, delta});
      }

      // Keep K closest
      if ((int)nbuf.size() > K) {
        std::partial_sort(nbuf.begin(), nbuf.begin() + K, nbuf.end(),
                          [](const Neighbor& a, const Neighbor& b) {
                            return a.dist < b.dist;
                          });
        nbuf.resize(K);
      }

      if (nbuf.empty()) {
        self.moveF((double)moveSpeed);
        self.step(dt);
        wrapPos(self.pos());
        continue;
      }

      // --- accumulate flocking forces ---
      Vec3d cohDelta(0, 0, 0);
      Vec3d avgFwd  (0, 0, 0);
      Vec3d sepForce(0, 0, 0);

      for (const Neighbor& nb : nbuf) {
        cohDelta += nb.delta;
        avgFwd   += agents[nb.idx].nav.uf();
        if (nb.dist < S)
          sepForce += (-nb.delta / nb.dist) / nb.dist;
      }

      const double n = double(nbuf.size());
      cohDelta /= n;
      avgFwd   /= n;

      // --- blend steering direction ---
      Vec3d steer(0, 0, 0);

      double cohMag = cohDelta.mag();
      if (cohMag > 1e-8)
        steer += (cohDelta / cohMag) * (double)cohesionW;

      double algMag = avgFwd.mag();
      if (algMag > 1e-8)
        steer += (avgFwd / algMag) * (double)alignW;

      double sepMag = sepForce.mag();
      if (sepMag > 1e-8)
        steer += (sepForce / sepMag) * (double)sepW;

      double steerMag = steer.mag();
      if (steerMag > 1e-8)
        self.faceToward(selfPos + steer / steerMag, (double)turnRate);

      self.moveF((double)moveSpeed);
      self.step(dt);
      wrapPos(self.pos());
    }
  }

  void onDraw(Graphics& g) override {
    g.clear(0.05f);
    g.lighting(false);
    g.color(0.4f, 0.7f, 1.0f);

    for (int i = 0; i < N; ++i) {
      g.pushMatrix();
      g.translate(agents[i].nav.pos());
      g.rotate(agents[i].nav.quat());
      g.draw(tetra);
      g.popMatrix();
    }

    gui.draw(g);
  }

  void onSound(AudioIOData& io) override {
    while (io()) {
      io.out(0) = 0.f;
      io.out(1) = 0.f;
    }
  }
};

int main() {
  MyApp app;
  app.configureAudio(44100, 512, 2, 0);
  app.start();
}
