// corvid_p1 — 200 love-chase agents, oriented tetrahedra, ControlGUI.
// Adapted from MAT201B_projects/p1.cpp.

#include "al/app/al_App.hpp"
#include "al/graphics/al_Shapes.hpp"
#include "al/math/al_Random.hpp"
#include "al/ui/al_ControlGUI.hpp"
#include "al/ui/al_Parameter.hpp"
#include "core/Agent.hpp"

#include <vector>

using namespace al;
using namespace corvid;

const int N = 200;

struct CorvidP1 : App {
  std::vector<Agent> agents;
  Mesh               tetra;
  ControlGUI         gui;

  Parameter turnRate  {"Turn Rate",  "Agents", 0.04f, 0.001f, 0.5f};
  Parameter moveSpeed {"Move Speed", "Agents", 2.0f,  0.1f,  10.0f};

  std::vector<int> loveTarget;

  void initAgents() {
    agents.resize(N);
    loveTarget.resize(N);
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
      loveTarget[i] = t;
    }
  }

  void onCreate() override {
    nav().pos(0, 0, 12);
    nav().faceToward(Vec3d(0, 0, 0));

    addTetrahedron(tetra, 0.12f);
    tetra.generateNormals();

    gui << turnRate << moveSpeed;
    gui.init();

    initAgents();
  }

  void onAnimate(double dt) override {
    for (int i = 0; i < N; ++i) {
      Nav& self = agents[i].nav;
      self.faceToward(agents[loveTarget[i]].nav.pos(), (double)turnRate);
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
    while (io()) { io.out(0) = 0.f; io.out(1) = 0.f; }
  }

  bool onKeyDown(const Keyboard& k) override {
    if (k.key() == Keyboard::ESCAPE) quit();
    return true;
  }
};

int main() {
  CorvidP1 app;
  app.configureAudio(44100, 512, 2, 0);
  app.start();
}
