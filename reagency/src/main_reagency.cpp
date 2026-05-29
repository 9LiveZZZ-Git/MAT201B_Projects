// World of Shadow Work — M0 scaffold.
// An autonomous, distributed AlloSphere piece: a navigable multimodal CLIP galaxy
// (every grain = one corpus image/word). M0 = load + draw the galaxy with a slow
// autonomous orbit. Later milestones add the live wander, human trace, smoke/vessel
// shells, and audio. Runtime is allolib-only; custom GLSL through al::ShaderProgram.
#include "al/app/al_App.hpp"
#include "al/app/al_DistributedApp.hpp"
#include "al/math/al_Quat.hpp"
#include "al/math/al_Vec.hpp"

#include "core/WoSWState.hpp"
#include "viz/ParticleField.hpp"

#include <cmath>
#include <memory>
#include <string>

using namespace al;
using namespace wosw;

// DistributedAppWithState: the primary simulates + broadcasts WoSWState; renderer
// nodes apply the synced camera and draw identically. On a single desktop the one
// node is primary, so this is also the dev/preview app.
struct WoSW : public DistributedAppWithState<WoSWState> {
  ParticleField field;
  std::string   assetDir = "assets";

  void onCreate() override {
    field.init(assetDir);
    nav().pos().set(0.0, 0.0, 16.0);
    nav().faceToward(Vec3d(0, 0, 0), Vec3d(0, 1, 0));
    // Renderers must not take local nav input — the primary's pose is authoritative.
    if (!isPrimary()) navControl().active(false);
  }

  void onAnimate(double dt) override {
    auto& s = state();
    if (isPrimary()) {
      s.frame++;
      s.simTime += float(dt);
      // M0: slow autonomous orbit so the galaxy reads as 3D / "alive".
      const float t = s.simTime * 0.05f, R = 16.f;
      nav().pos().set(R * std::sin(t), 2.0 * std::sin(t * 0.5f), R * std::cos(t));
      nav().faceToward(Vec3d(0, 0, 0), Vec3d(0, 1, 0));
      // pack pose for renderers
      const auto p = nav().pos();
      const auto q = nav().quat();
      s.navPos[0] = float(p.x); s.navPos[1] = float(p.y); s.navPos[2] = float(p.z);
      s.navQuat[0] = float(q.x); s.navQuat[1] = float(q.y);
      s.navQuat[2] = float(q.z); s.navQuat[3] = float(q.w);
    } else {
      // apply the primary's exact pose (Quatd is w,x,y,z)
      nav().pos().set(s.navPos[0], s.navPos[1], s.navPos[2]);
      nav().quat() = Quatd(s.navQuat[3], s.navQuat[0], s.navQuat[1], s.navQuat[2]);
    }
  }

  void onDraw(Graphics& g) override {
    g.clear(0.03f, 0.03f, 0.05f);     // subtly-lifted void (reads better in the dome)
    field.draw(g, 1.f);
  }

  void onSound(AudioIOData& io) override {
    if (!isPrimary()) return;          // primary-only audio (M0: silent)
    (void)io;
  }
};

int main(int argc, char** argv) {
  auto app = std::make_unique<WoSW>();
  if (argc > 1) app->assetDir = argv[1];     // run_demo.sh passes an absolute assets path
  app->configureAudio(44100, 512, 2, 0);
  app->title("World of Shadow Work");
  app->start(/*packetSize=*/4096);           // WoSWState is tiny
  return 0;
}
