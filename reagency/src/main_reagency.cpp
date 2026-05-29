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
#include "viz/LabelLayer.hpp"
#include "audio/AudioEngine.hpp"

#include <cmath>
#include <memory>
#include <string>
#include <vector>

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
  LabelLayer    labels;
  AudioEngine   audio_;                       // primary-only generative voice (M5)
  int           prevCurNode_ = -1;            // arrival edge-detect for audio events
  std::vector<int> heroNodes_;                // corpus image nodes that have an atlas thumbnail
  int           heroCursor_ = 0;
  float         traceTimer_ = 0.f;
  static constexpr float TRACE_LIFE  = 9.0f;  // seconds a surfaced photo lives (lingers)
  static constexpr float TRACE_EVERY = 6.0f;  // cadence between new photos (calm, ~rotation-paced)
  // Depth crossfade: a slow autonomous "tide" between the galaxy (home) and a descent to the
  // core vessel. depthPhase_ advances in time but SLOWS when the machine hesitates, so the
  // descent breathes with the mind. Dwells longer in the galaxy; dips fully to the vessel once
  // per period. depth: 1 = galaxy out among the points, 0 = sunk into the core vessel shell.
  float         depthPhase_ = 0.f;
  static constexpr float DEPTH_PERIOD = 80.0f;  // seconds for a full galaxy->vessel->galaxy tide
  std::string   assetDir = "assets";

  void onCreate() override {
    field.init(assetDir);
    auto pos = field.positions();
    auto col = field.colors();
    webs.init(assetDir, pos, col);
    conductor.init(pos, webs.adjacency(), 7);
    vessel.init(assetDir);
    trace.init(assetDir);
    labels.init(assetDir);
    audio_.init(assetDir, audioIO().framesPerSecond());   // primary-only; scale from manifest.json
    for (int i = 0; i < field.count(); ++i)            // traceable corpus photos
      if (field.typeOf(i) == 0 && field.atlasOf(i) >= 0) heroNodes_.push_back(i);
    nav().pos().set(0.0, 0.0, 16.0);
    nav().faceToward(Vec3d(0, 0, 0), Vec3d(0, 1, 0));
    // Renderers must not take local nav input — the primary's pose is authoritative.
    if (!isPrimary()) navControl().active(false);
  }

  // Stereo pan in [-1,1] for a world point, from the camera's right axis (audio localization).
  float panOf(const Vec3f& w) {
    Vec3d d = Vec3d(w.x, w.y, w.z) - nav().pos();
    double L = d.mag();
    if (L < 1e-5) return 0.f;
    d /= L;
    Vec3d r = nav().quat().rotate(Vec3d(1, 0, 0));
    double p = d.dot(r);
    return float(p < -1 ? -1 : (p > 1 ? 1 : p));
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

      // AUDIO: on each NEW node arrival, sound the node's note + ignite an arpeggio over its
      // k-NN edges (the web "belongs-together" made audible). Pitch/timbre from type/density.
      if (s.curNode != prevCurNode_ && s.curNode >= 0) {
        audio_.onArrival(s.curNode, field.typeOf(s.curNode),
                         field.densityOf(s.curNode), field.clusterOf(s.curNode));
        const auto& adj = webs.adjacency();
        if (s.curNode < int(adj.size())) audio_.igniteArp(adj[s.curNode], field.clusterOf(s.curNode));
        prevCurNode_ = s.curNode;
      }

      // Human trace ring: age + expire; surface a NEW photo on a steady cadence (round-robin
      // through the corpus's hero images) so the dissolve is ALWAYS on screen — not only when
      // the wander happens to land on one (which was far too rare to see).
      for (int i = 0; i < WoSWState::N_TRACE; ++i)
        if (s.traceNode[i] >= 0) {
          s.traceAge[i] += float(dt);
          if (s.traceAge[i] > TRACE_LIFE) { audio_.traceOff(i); s.traceNode[i] = -1; }
        }
      traceTimer_ += float(dt);
      if (!heroNodes_.empty() && traceTimer_ > TRACE_EVERY) {
        traceTimer_ = 0.f;
        const int node = heroNodes_[(heroCursor_++) % int(heroNodes_.size())];
        int freeSlot = -1, oldest = 0;
        for (int i = 0; i < WoSWState::N_TRACE; ++i) {
          if (s.traceNode[i] < 0 && freeSlot < 0) freeSlot = i;
          if (s.traceAge[i] > s.traceAge[oldest]) oldest = i;
        }
        const int slot = (freeSlot >= 0) ? freeSlot : oldest;
        s.traceNode[slot] = node; s.traceAge[slot] = 0.f;
        // AUDIO: a warm presence at the photo + a synthesized vocal WHISPER of the word the
        // machine classifies it as, which dissolves into a granular fade (twin of the photo's
        // dissolve into grains). The word comes from LabelLayer (empty until the regen → a
        // stable per-node pseudo-vocalisation is whispered instead).
        const float tpan = panOf(field.posOf(node));
        audio_.traceOn(slot, node, tpan);
        audio_.whisper(labels.wordForNode(node), node, tpan);
      }

      // Depth tide: advance the phase, slowed by hesitation; map to depth that dwells near the
      // galaxy (1) and dips to the vessel (0) once per period (pow<1 biases time toward 1).
      depthPhase_ += float(dt) * (1.f - 0.4f * s.hesitation) / DEPTH_PERIOD;
      const float u   = depthPhase_ * 6.2831853f;
      const float raw = 0.5f - 0.5f * std::cos(u);          // 0..1, starts at vessel
      s.depth = std::pow(raw, 0.6f);                          // dwell longer in the galaxy

      // Camera. A DENSE corpus (the real ~10k) lets us sit INSIDE the cloud and be enveloped;
      // a small preview corpus is too sparse for inside-out (you'd see mostly void), so we
      // orbit just outside and look in. Auto-switches once the corpus is dense. The depth tide
      // pulls the eye toward the core vessel as it descends (depth->0) and back out (depth->1).
      const double t = s.simTime;
      const double d = s.depth;
      if (field.count() >= 3000) {
        // immersive: inside the cloud; radius shrinks toward the core as we descend
        const double k = 0.30 + 0.95 * d;
        const Vec3d eye(1.8 * k * std::sin(t * 0.045), 0.9 * k * std::sin(t * 0.031),
                        1.8 * k * std::cos(t * 0.045));
        nav().pos().set(eye.x, eye.y, eye.z);
        const double yaw = t * 0.05;
        const Vec3d look = eye + Vec3d(std::sin(yaw), 0.12 * std::sin(t * 0.02), std::cos(yaw)) * 6.0;
        nav().faceToward(look, Vec3d(0, 1, 0));
      } else {
        // sparse preview: orbit looking in; radius descends toward the core vessel with depth
        const double a = t * 0.05, R = 6.0 + 9.0 * d;
        nav().pos().set(R * std::sin(a), 2.0 * std::sin(a * 0.5), R * std::cos(a));
        nav().faceToward(Vec3d(0, 0, 0), Vec3d(0, 1, 0));
      }

      // pack pose for renderers
      const auto p = nav().pos();
      const auto q = nav().quat();
      s.navPos[0] = float(p.x); s.navPos[1] = float(p.y); s.navPos[2] = float(p.z);
      s.navQuat[0] = float(q.x); s.navQuat[1] = float(q.y);
      s.navQuat[2] = float(q.z); s.navQuat[3] = float(q.w);

      // AUDIO continuous controls: hesitation -> timbre/reverb, depth -> bright<->dark morph,
      // focus point -> stereo localization, progress -> (reserved). Also pumps the arpeggio.
      audio_.update(float(dt), s.hesitation, s.depth, conductor.progress(),
                    panOf(Vec3f(s.focusPos[0], s.focusPos[1], s.focusPos[2])));
    } else {
      // apply the primary's exact pose (Quatd is w,x,y,z)
      nav().pos().set(s.navPos[0], s.navPos[1], s.navPos[2]);
      nav().quat() = Quatd(s.navQuat[3], s.navQuat[0], s.navQuat[1], s.navQuat[2]);
    }

    // The vessel morphs on every node from the synced keyframe position.
    vessel.update(s.vesselKf, s.simTime);
  }

  void onDraw(Graphics& g) override {
    auto& s = state();
    // Depth crossfade weights (synced depth, so identical on every dome node). dd = smoothstep,
    // vd = vessel-ness. As we descend (depth->0): galaxy dims, webs recede, the vessel cocoon
    // brightens and swells around the core; rising back out reverses it.
    const float depth = s.depth;
    const float dd = depth * depth * (3.f - 2.f * depth);
    const float vd = 1.f - dd;
    const float galBright = 0.45f + 0.55f * dd;
    const float webBright = 0.20f + 0.80f * dd;
    const float vesAlpha  = 0.10f + 0.70f * vd;
    const float vesScale  = 1.3f + 1.7f * vd;

    g.clear(0.03f, 0.03f, 0.05f);     // subtly-lifted void (reads better in the dome)
    webs.draw(g, webBright);          // similarity webs under the points
    field.draw(g, 1.f, galBright);

    // The "vessel" at the core — swells into an enveloping shell as we descend, faint central
    // haze out in the galaxy (the missing splat middle comes from Colab).
    vessel.draw(g, Vec3f(0, 0, 0), vesScale, vesAlpha);

    // Camera-facing axes (for billboards), from the synced pose.
    Quatd q(s.navQuat[3], s.navQuat[0], s.navQuat[1], s.navQuat[2]);
    Vec3d rgt = q.rotate(Vec3d(1, 0, 0));
    Vec3d upv = q.rotate(Vec3d(0, 1, 0));
    const Vec3f camR(float(rgt.x), float(rgt.y), float(rgt.z));
    const Vec3f camU(float(upv.x), float(upv.y), float(upv.z));

    // The machine's CLASSIFICATION: floating word labels at the cluster centroids (a standing
    // map of what the machine thinks the imagery IS).
    labels.drawClusters(g, camR, camU, 0.32f);     // words floating IN the galaxy, subtly

    // Human traces: photos surface at their node positions (multiple at once), dissolving into
    // grains that stream into the galaxy — and the word the machine classifies each as.
    if (trace.ready()) {
      for (int i = 0; i < WoSWState::N_TRACE; ++i) {
        if (s.traceNode[i] < 0) continue;
        const Vec3f p = field.posOf(s.traceNode[i]);
        const float a = s.traceAge[i];
        trace.draw(g, p, camR, camU, field.atlasOf(s.traceNode[i]), a, TRACE_LIFE);
        // its classifying word, just below the photo, fading on the same envelope
        const float r01 = a / TRACE_LIFE;
        float lab = (r01 < 0.12f) ? (r01 / 0.12f)
                  : (r01 < 0.5f) ? 1.f : (1.f - (r01 - 0.5f) / 0.5f);
        if (lab < 0.f) lab = 0.f;
        labels.drawSlot(g, p - camU * 1.4f, camR, camU, labels.slotForNode(s.traceNode[i]), lab, 0.45f);
      }
    }
  }

  void onSound(AudioIOData& io) override {
    if (!isPrimary()) return;          // primary-only audio
    audio_.render(io);
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
