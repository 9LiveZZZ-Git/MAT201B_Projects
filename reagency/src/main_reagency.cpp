// World of Shadow Work — runtime.
// An autonomous, distributed AlloSphere piece: a multimodal CLIP galaxy that SURROUNDS the
// viewer (camera inside; webs central). The Conductor walks the k-NN graph LIVE; when it
// fixates an image, that photo surfaces at its node and dissolves into grains that stream
// into the galaxy (the human->machine cede). A morphing splat "vessel" sits at the core.
// Runtime is allolib-only; custom GLSL through al::ShaderProgram.
#include "al/app/al_App.hpp"
#include "al/app/al_DistributedApp.hpp"
#include "al/math/al_Quat.hpp"
#include "al/math/al_Vec.hpp"

#include "core/WoSWState.hpp"
#include "core/Conductor.hpp"
#include "viz/ParticleField.hpp"
#include "viz/WebRenderer.hpp"
#include "viz/VesselSplats.hpp"
#include "viz/HumanTrace.hpp"

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
  VesselSplats  vessel;
  HumanTrace    trace;
  int           lastCurNode_ = -1;
  static constexpr float TRACE_LIFE = 4.5f;   // seconds a surfaced photo lives
  std::string   assetDir = "assets";

  void onCreate() override {
    field.init(assetDir);
    auto pos = field.positions();
    auto col = field.colors();
    webs.init(assetDir, pos, col);
    conductor.init(pos, webs.adjacency(), 7);
    vessel.init(assetDir);
    trace.init(assetDir);
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

      // The mind still walks its own k-NN graph (it drives the vessel morph + hesitation
      // + future audio), but it no longer moves the camera — that read as distracting
      // zoom. The wander shows in the morphing vessel, not in camera motion.
      conductor.step(float(dt));
      s.curNode    = conductor.curNode();
      s.nextNode   = conductor.nextNode();
      s.hesitation = conductor.hesitation();
      const Vec3f focus = conductor.focusPos();
      s.focusPos[0] = focus.x; s.focusPos[1] = focus.y; s.focusPos[2] = focus.z;
      s.vesselKf   = float(conductor.visits()) + conductor.progress();

      // Human trace ring: age + expire surfaced photos; surface a new one when the machine
      // settles on a hero image (multiple can be alive at once, at their node positions).
      for (int i = 0; i < WoSWState::N_TRACE; ++i)
        if (s.traceNode[i] >= 0) {
          s.traceAge[i] += float(dt);
          if (s.traceAge[i] > TRACE_LIFE) s.traceNode[i] = -1;
        }
      const int cn = conductor.curNode();
      if (cn != lastCurNode_) {                  // arrived at a new node
        lastCurNode_ = cn;
        if (cn >= 0 && field.typeOf(cn) == 0 && field.atlasOf(cn) >= 0) {
          bool present = false; int freeSlot = -1, oldest = 0;
          for (int i = 0; i < WoSWState::N_TRACE; ++i) {
            if (s.traceNode[i] == cn) present = true;
            if (s.traceNode[i] < 0 && freeSlot < 0) freeSlot = i;
            if (s.traceAge[i] > s.traceAge[oldest]) oldest = i;
          }
          if (!present) {
            const int slot = (freeSlot >= 0) ? freeSlot : oldest;
            s.traceNode[slot] = cn; s.traceAge[slot] = 0.f;
          }
        }
      }

      // Camera. A DENSE corpus (the real ~10k) lets us sit INSIDE the cloud and be enveloped;
      // a small preview corpus is too sparse for inside-out (you'd see mostly void), so we
      // orbit just outside and look in. Auto-switches once the corpus is dense.
      const double t = s.simTime;
      if (field.count() >= 3000) {
        // immersive: inside the cloud, gentle drift + slow yaw (omni fills the 360 in the dome)
        const Vec3d eye(1.8 * std::sin(t * 0.045), 0.9 * std::sin(t * 0.031), 1.8 * std::cos(t * 0.045));
        nav().pos().set(eye.x, eye.y, eye.z);
        const double yaw = t * 0.05;
        const Vec3d look = eye + Vec3d(std::sin(yaw), 0.12 * std::sin(t * 0.02), std::cos(yaw)) * 6.0;
        nav().faceToward(look, Vec3d(0, 1, 0));
      } else {
        // sparse preview: close orbit, looking in, so the whole cloud reads
        const double a = t * 0.05, R = 9.0;
        nav().pos().set(R * std::sin(a), 1.5 * std::sin(a * 0.5), R * std::cos(a));
        nav().faceToward(Vec3d(0, 0, 0), Vec3d(0, 1, 0));
      }

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

    // The vessel morphs on every node from the synced keyframe position.
    vessel.update(s.vesselKf, s.simTime);
  }

  void onDraw(Graphics& g) override {
    g.clear(0.03f, 0.03f, 0.05f);     // subtly-lifted void (reads better in the dome)
    webs.draw(g);                     // similarity webs under the points
    field.draw(g, 1.f);

    // The "vessel" at the core — faint central haze (the missing splat middle; comes from Colab).
    vessel.draw(g, Vec3f(0, 0, 0), 1.3f, 0.3f);

    // Human traces: photos surface at their node positions all around you (multiple at once),
    // each dissolving into grains that stream out into the galaxy.
    auto& s = state();
    if (trace.ready()) {
      Quatd q(s.navQuat[3], s.navQuat[0], s.navQuat[1], s.navQuat[2]);
      Vec3d rgt = q.rotate(Vec3d(1, 0, 0));
      Vec3d upv = q.rotate(Vec3d(0, 1, 0));
      const Vec3f camR(float(rgt.x), float(rgt.y), float(rgt.z));
      const Vec3f camU(float(upv.x), float(upv.y), float(upv.z));
      for (int i = 0; i < WoSWState::N_TRACE; ++i) {
        if (s.traceNode[i] < 0) continue;
        trace.draw(g, field.posOf(s.traceNode[i]), camR, camU,
                   field.atlasOf(s.traceNode[i]), s.traceAge[i], TRACE_LIFE);
      }
    }
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
