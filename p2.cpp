// p2.cpp — Random Chasing with Separation

#include "al/app/al_App.hpp"
#include "al/graphics/al_Shapes.hpp"
#include "al/io/al_ControlNav.hpp"
#include "al/math/al_Random.hpp"
#include "al/ui/al_ControlGUI.hpp"
#include "al/ui/al_Parameter.hpp"

#include <vector>

using namespace al;

const int N = 200;

struct Agent {
  Nav nav;
  int id         = -1;
  int loveTarget = -1;
};

struct MyApp : App {
  std::vector<Agent> agents;
  Mesh               tetra;
  ControlGUI         gui;

  // --- tunables ---
  Parameter turnRate   {"Turn Rate",       "Agents", 0.04f, 0.001f, 0.5f};
  Parameter moveSpeed  {"Move Speed",      "Agents", 2.0f,  0.1f,   10.0f};
  Parameter tSep       {"Sep Threshold",   "Agents", 0.5f,  0.05f,  3.0f};
  Parameter loveWeight {"Love Weight",     "Agents", 1.0f,  0.0f,   5.0f};
  Parameter sepWeight  {"Sep Weight",      "Agents", 3.0f,  0.0f,   10.0f};

  // ----------------------------------------------------------------
  void initAgents() {
    agents.resize(N);
    for (int i = 0; i < N; ++i) {
      agents[i].id = i;
      agents[i].nav.pos(rnd::uniformS() * 4.0,
                        rnd::uniformS() * 4.0,
                        rnd::uniformS() * 4.0);
      agents[i].nav.quat(Quatd().fromEuler(
        rnd::uniform(0.0, M_2PI),
        rnd::uniform(0.0, M_2PI),
        rnd::uniform(0.0, M_2PI)));
      int t;
      do { t = int(rnd::uniform(0.0, double(N))); } while (t == i);
      agents[i].loveTarget = t;
    }
  }

  void onCreate() override {
    nav().pos(0, 0, 12);
    nav().faceToward(Vec3d(0, 0, 0));

    addTetrahedron(tetra, 0.12f);
    tetra.generateNormals();

    gui << turnRate << moveSpeed << tSep << loveWeight << sepWeight;
    gui.init();

    initAgents();
  }

  void onAnimate(double dt) override {
    for (int i = 0; i < N; ++i) {
      Nav&         self    = agents[i].nav;
      const Vec3d& selfPos = self.pos();

      // --- love: unit vector toward target ---
      Vec3d towardLove = agents[agents[i].loveTarget].nav.pos() - selfPos;
      double loveDist  = towardLove.mag();
      if (loveDist > 1e-8) towardLove /= loveDist;

      // --- separation: 1/d²-weighted repulsion from near neighbors ---
      Vec3d sepForce(0, 0, 0);
      for (int j = 0; j < N; ++j) {
        if (j == i) continue;
        Vec3d  away = selfPos - agents[j].nav.pos();
        double dist = away.mag();
        if (dist < (double)tSep && dist > 1e-8)
          sepForce += (away / dist) / dist;
      }

      // --- blend and steer ---
      Vec3d steer = towardLove * (double)loveWeight;
      double sepMag = sepForce.mag();
      if (sepMag > 1e-8)
        steer += (sepForce / sepMag) * (double)sepWeight;

      double steerMag = steer.mag();
      if (steerMag > 1e-8)
        self.faceToward(selfPos + steer / steerMag, (double)turnRate);

      self.moveF((double)moveSpeed);
      self.step(dt);
    }
  }

  void onDraw(Graphics& g) override {
    g.clear(0.05f);
    g.lighting(false);
    g.color(0.8f, 0.5f, 0.2f);

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
