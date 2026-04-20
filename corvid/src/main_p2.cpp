// corvid_p2 — Reynolds flocking, spatial hash, toroidal world, ControlGUI.
// Adapted from MAT201B_projects/p3.cpp.

#include "al/app/al_App.hpp"
#include "al/graphics/al_Shapes.hpp"
#include "al/math/al_Random.hpp"
#include "al/ui/al_ControlGUI.hpp"
#include "al/ui/al_Parameter.hpp"
#include "core/Agent.hpp"
#include "core/SpatialHash.hpp"
#include "reactive/Boids.hpp"

#include <algorithm>
#include <vector>

using namespace al;
using namespace corvid;

const int    N = 200;
const double W = 10.0;

struct CorvidP2 : App {
  std::array<Agent, N> agents;
  Mesh                 tetra;
  SpatialHash          hash;
  ControlGUI           gui;
  double               prevT = -1.0;

  Parameter turnRate   {"Turn Rate",       "Agents", 0.05f, 0.001f, 0.3f};
  Parameter moveSpeed  {"Move Speed",      "Agents", 2.0f,  0.1f,  10.0f};
  Parameter tNeighbor  {"Neighbor Radius", "Agents", 1.0f,  0.1f,   4.0f};
  Parameter tSep       {"Sep Threshold",   "Agents", 0.35f, 0.02f,  2.0f};
  Parameter alignW     {"Align Weight",    "Agents", 1.0f,  0.0f,   5.0f};
  Parameter cohesionW  {"Cohesion Weight", "Agents", 0.8f,  0.0f,   5.0f};
  Parameter sepW       {"Sep Weight",      "Agents", 2.5f,  0.0f,  10.0f};
  ParameterInt kNeighbors {"K Neighbors",  "Agents", 8,     1,     30};

  void initAgents() {
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

    gui << turnRate << moveSpeed << tNeighbor << tSep
        << kNeighbors << alignW << cohesionW << sepW;
    gui.init();

    initAgents();
    hash.rebuild((double)tNeighbor, W);
    prevT = (double)tNeighbor;
  }

  void onAnimate(double dt) override {
    const double T  = (double)tNeighbor;
    const double S  = (double)tSep;
    const int    K  = (int)kNeighbors;

    if (T != prevT) { hash.rebuild(T, W); prevT = T; }
    hash.clear();
    for (int i = 0; i < N; ++i) hash.insert(i, agents[i].nav.pos());

    std::vector<int>      candidates;
    std::vector<Neighbor> nbuf;
    candidates.reserve(64);
    nbuf.reserve(64);

    for (int i = 0; i < N; ++i) {
      Nav&         self    = agents[i].nav;
      const Vec3d& selfPos = self.pos();

      hash.query(selfPos, candidates);
      nbuf.clear();
      for (int j : candidates) {
        if (j == i) continue;
        Vec3d  delta = toroidalDelta(selfPos, agents[j].nav.pos(), W);
        double dist  = delta.mag();
        if (dist < T) nbuf.push_back({j, dist, delta});
      }

      if ((int)nbuf.size() > K) {
        std::partial_sort(nbuf.begin(), nbuf.begin() + K, nbuf.end(),
          [](const Neighbor& a, const Neighbor& b){ return a.dist < b.dist; });
        nbuf.resize(K);
      }

      if (nbuf.empty()) {
        self.moveF((double)moveSpeed);
        self.step(dt);
        Vec3d p = self.pos();
        wrapPos(p, W);
        self.pos(p);
        continue;
      }

      Vec3d steer(0, 0, 0);
      steer += computeAlignment (nbuf, agents.data()) * (double)alignW;
      steer += computeCohesion  (nbuf)                * (double)cohesionW;
      steer += computeSeparation(nbuf, S)             * (double)sepW;

      double sm = steer.mag();
      if (sm > 1e-8) self.faceToward(selfPos + steer / sm, (double)turnRate);

      self.moveF((double)moveSpeed);
      self.step(dt);
      Vec3d p = self.pos();
      wrapPos(p, W);
      self.pos(p);
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
    while (io()) { io.out(0) = 0.f; io.out(1) = 0.f; }
  }

  bool onKeyDown(const Keyboard& k) override {
    if (k.key() == Keyboard::ESCAPE) quit();
    return true;
  }
};

int main() {
  CorvidP2 app;
  app.configureAudio(44100, 512, 2, 0);
  app.start();
}
