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

      // Human trace: surface the fixated image's photo, dissolving as the machine moves on.
      const int cn = conductor.curNode();
      if (cn >= 0 && field.typeOf(cn) == 0 && field.atlasOf(cn) >= 0) {
        s.traceNode = float(cn);
        const float pr = conductor.progress();
        const float ta = (pr < 0.15f) ? (pr / 0.15f) : (1.f - (pr - 0.15f) / 0.85f);
        s.traceAlpha = ta < 0.f ? 0.f : ta;
      } else {
        s.traceNode = -1.f; s.traceAlpha = 0.f;
      }

      // Camera: calm slow orbit, closer in so the galaxy fills the frame.
      const float t = s.simTime * 0.05f, R = 11.f;
      nav().pos().set(R * std::sin(t), 1.0 * std::sin(t * 0.5f), R * std::cos(t));
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

    // The vessel morphs on every node from the synced keyframe position.
    vessel.update(s.vesselKf, s.simTime);
  }

  void onDraw(Graphics& g) override {
    g.clear(0.03f, 0.03f, 0.05f);     // subtly-lifted void (reads better in the dome)
    webs.draw(g);                     // similarity webs under the points
    field.draw(g, 1.f);

    // The outer "vessel": the machine's morphing image-to-3D dream-body, at the galaxy
    // center. The camera calmly orbits it; depth-crossfade with the galaxy comes later.
    vessel.draw(g, Vec3f(0, 0, 0), 1.6f, 0.5f);

    // Human trace: the fixated image's photo surfaces in front of the camera, then dissolves.
    auto& s = state();
    if (int(s.traceNode) >= 0 && s.traceAlpha > 0.001f && trace.ready()) {
      Quatd q(s.navQuat[3], s.navQuat[0], s.navQuat[1], s.navQuat[2]);
      Vec3d fwd = q.rotate(Vec3d(0, 0, -1));
      Vec3d rgt = q.rotate(Vec3d(1, 0, 0));
      Vec3d upv = q.rotate(Vec3d(0, 1, 0));
      Vec3f center(s.navPos[0] + float(fwd.x) * 5.f,
                   s.navPos[1] + float(fwd.y) * 5.f,
                   s.navPos[2] + float(fwd.z) * 5.f);
      trace.draw(g, center,
                 Vec3f(float(rgt.x), float(rgt.y), float(rgt.z)),
                 Vec3f(float(upv.x), float(upv.y), float(upv.z)),
                 field.atlasOf(int(s.traceNode)), s.traceAlpha, 2.6f);
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
