// World of Shadow Work — runtime (M2).
// An autonomous, distributed AlloSphere piece: a navigable multimodal CLIP galaxy
// (every grain = one corpus image/word). The Conductor walks the k-NN graph LIVE; the
// camera follows its fixation and wobbles when it hesitates (= entropy of its choice),
// with a glowing orb marking where the machine is "looking." Later milestones add the
// human trace, smoke/vessel shells, and audio. Runtime is allolib-only; custom GLSL
// through al::ShaderProgram.
#include "al/app/al_App.hpp"
#include "al/app/al_DistributedApp.hpp"
#include "al/math/al_Quat.hpp"
#include "al/math/al_Vec.hpp"
#include "al/graphics/al_Shapes.hpp"

#include "core/WoSWState.hpp"
#include "core/Conductor.hpp"
#include "viz/ParticleField.hpp"
#include "viz/WebRenderer.hpp"

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
  WebRenderer   webs;
  Conductor     conductor;
  al::Mesh      marker_;              // glowing orb at the machine's current fixation
  al::Vec3f     camFocus_{0, 0, 0};   // smoothed camera target (hides graph jumps)
  std::string   assetDir = "assets";

  void onCreate() override {
    field.init(assetDir);
    auto pos = field.positions();
    auto col = field.colors();
    webs.init(assetDir, pos, col);
    conductor.init(pos, webs.adjacency(), 7);
    addSphere(marker_, 1.0, 16, 16);
    camFocus_ = conductor.focusPos();
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

      // The autonomous mind walks its own k-NN graph live; we render its real choices.
      conductor.step(float(dt));
      s.curNode    = conductor.curNode();
      s.nextNode   = conductor.nextNode();
      s.hesitation = conductor.hesitation();
      const Vec3f focus = conductor.focusPos();
      s.focusPos[0] = focus.x; s.focusPos[1] = focus.y; s.focusPos[2] = focus.z;

      // Camera follows the fixation: smoothed focus + a slow orbit, wobbling when the
      // machine hesitates (entropy high). Smoothing the focus hides graph jumps.
      camFocus_ += (focus - camFocus_) * 0.04f;
      const float t = s.simTime, R = 7.5f;
      Vec3d eye(camFocus_.x + R * std::sin(t * 0.13f),
                camFocus_.y + 2.2,
                camFocus_.z + R * std::cos(t * 0.13f));
      eye += Vec3d(std::sin(t * 7.0f), std::cos(t * 6.3f), std::sin(t * 5.1f)) * (0.5 * s.hesitation);
      nav().pos().set(eye.x, eye.y, eye.z);
      nav().faceToward(Vec3d(camFocus_.x, camFocus_.y, camFocus_.z), Vec3d(0, 1, 0));

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
    webs.draw(g);                     // similarity webs under the points
    field.draw(g, 1.f);

    // The machine's current "attention": a warm glowing orb at the fixation, pulsing
    // bigger when it hesitates. Position is synced state, so it matches on every node.
    auto& s = state();
    g.depthTesting(false);
    g.blending(true);
    g.blendAdd();
    g.pushMatrix();
    g.translate(s.focusPos[0], s.focusPos[1], s.focusPos[2]);
    g.scale(0.35f + 0.12f * std::sin(s.simTime * 4.f) + 0.5f * s.hesitation);
    g.color(1.0f, 0.85f, 0.6f, 1.f);
    g.draw(marker_);
    g.popMatrix();
    g.blendTrans();
    g.depthTesting(true);
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
