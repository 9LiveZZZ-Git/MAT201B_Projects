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
#include "viz/CaptionLayer.hpp"
#include "viz/StoryLayer.hpp"
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
  CaptionLayer  captions_;                    // v2: the machine's VOICE (me/you/them/us) [dormant]
  StoryLayer    stories_;                      // v2: the THEM register IN THE GALAXY (proximity-read)
  AudioEngine   audio_;                       // primary-only generative voice (M5)
  int           prevCurNode_ = -1;            // arrival edge-detect for audio events
  std::vector<int> heroNodes_;                // corpus image nodes that have an atlas thumbnail
  int           heroCursor_ = 0;
  float         traceTimer_ = 0.f;
  static constexpr float TRACE_LIFE  = 9.0f;  // seconds a surfaced photo lives (lingers)
  // ----- v2 5-ACT NARRATIVE RE-SCORE -----
  // A ~14-min ritual cycle replaces the old 80s cos() tide. cycleClock_ advances with time but
  // SLOWS when the machine hesitates, so the ritual breathes with the mind. cycleT in [0,CYCLE)
  // maps to a piecewise depth envelope across the acts: I seduction -> II reading -> III extraction
  // -> IV turn (floor 0.10, the plunge) -> V integration. The COMFORT FAILS: Act V re-brightens only
  // partway (~0.50), never back to the Act-I seduction; the loop seam is continuous (V end == I
  // start) so it never hard-resets. depth: 1 = galaxy out among the points, 0 = sunk into the core.
  static constexpr float CYCLE = 840.f;
  float         cycleClock_ = 0.f;
  static float actEnvelope(float t, int& act) {            // depth(t) + act index (1..5)
    struct KF { float t, d; };
    static const KF kf[] = {
      {  0.f, 0.50f}, {120.f, 0.95f}, {210.f, 0.90f}, {360.f, 0.60f},
      {510.f, 0.15f}, {600.f, 0.10f}, {720.f, 0.30f}, {840.f, 0.50f},
    };
    act = (t < 210.f) ? 1 : (t < 360.f) ? 2 : (t < 510.f) ? 3 : (t < 600.f) ? 4 : 5;
    for (int i = 0; i < 7; ++i)
      if (t >= kf[i].t && t <= kf[i + 1].t) {
        float u = (t - kf[i].t) / (kf[i + 1].t - kf[i].t);
        u = u * u * (3.f - 2.f * u);                        // smoothstep
        return kf[i].d + (kf[i + 1].d - kf[i].d) * u;
      }
    return 0.50f;
  }
  static float actTraceEvery(int act) {                    // the cede accelerates in extraction/turn
    switch (act) { case 3: return 3.0f; case 4: return 2.0f; case 5: return 5.0f; default: return 6.0f; }
  }
  // Manual dive: SPACE toggles a descent straight to the core vessel (depth->0), overriding the
  // tide so the splat "vessel" can be summoned on demand (demo/verify; also the seed of the
  // Act-IV plunge). Eases in/out for a smooth, reversible plunge.
  float         diveOverride_ = 0.f;            // eased 0..1
  bool          diveActive_   = false;          // toggled by SPACE
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
    captions_.init(assetDir);                             // v2 voice (dormant; superseded by stories_)
    stories_.init(assetDir);                              // v2 THEM stories in the galaxy
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

      // v2 5-act re-score: advance the cycle clock (slowed by hesitation), read the act + depth.
      // Everything below (trace cadence, the depth crossfade, the dive) subordinates to this.
      cycleClock_ += float(dt) * (1.f - 0.4f * s.hesitation);
      const float cycleT = std::fmod(cycleClock_, CYCLE);
      int act; const float actDepth = actEnvelope(cycleT, act);   // act 1..5; depth 1=galaxy..0=core

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
      if (!heroNodes_.empty() && traceTimer_ > actTraceEvery(act)) {
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

      // Manual dive override (SPACE): ease toward the core, layered on the 5-act depth envelope
      // (actDepth, computed above). Single writer of s.depth.
      const float diveTarget = diveActive_ ? 1.f : 0.f;
      diveOverride_ += (diveTarget - diveOverride_) * std::min(1.f, float(dt) * 2.5f);  // ~0.4s ease
      s.depth = actDepth * (1.f - diveOverride_);              // diveOverride_->1 sinks to the core (depth 0)

      (void)act;   // (act still drives trace cadence above; the voice now lives in the galaxy)

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

      // AUDIO continuous controls + the 5-ACT orchestration: each act has its own palette
      // (I seduction -> II reading -> III extraction grind -> IV bare-pulse turn -> V haunted residue).
      audio_.update(float(dt), s.hesitation, s.depth, conductor.progress(),
                    panOf(Vec3f(s.focusPos[0], s.focusPos[1], s.focusPos[2])), act);
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
    // haze out in the galaxy. SPACE dives straight to it (see diveOverride_).
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

    // Act-IV TURN: the cede REVERSES at the deep-core descent (Act IV, or a SPACE dive). Derived
    // from the synced depth so it's identical on every dome node: grains stream INWARD, not out.
    float turnConverge = (0.20f - s.depth) / (0.20f - 0.08f);
    turnConverge = turnConverge < 0.f ? 0.f : (turnConverge > 1.f ? 1.f : turnConverge);
    turnConverge = turnConverge * turnConverge * (3.f - 2.f * turnConverge);   // smoothstep

    // Human traces: photos surface at their node positions (multiple at once), dissolving into
    // grains that stream into the galaxy — and the word the machine classifies each as.
    if (trace.ready()) {
      for (int i = 0; i < WoSWState::N_TRACE; ++i) {
        if (s.traceNode[i] < 0) continue;
        const Vec3f p = field.posOf(s.traceNode[i]);
        const float a = s.traceAge[i];
        trace.draw(g, p, camR, camU, field.atlasOf(s.traceNode[i]), a, TRACE_LIFE, turnConverge);
        // its classifying word, just below the photo, fading on the same envelope
        const float r01 = a / TRACE_LIFE;
        float lab = (r01 < 0.12f) ? (r01 / 0.12f)
                  : (r01 < 0.5f) ? 1.f : (1.f - (r01 - 0.5f) / 0.5f);
        if (lab < 0.f) lab = 0.f;
        labels.drawSlot(g, p - camU * 1.4f, camR, camU, labels.slotForNode(s.traceNode[i]), lab, 0.45f);
      }
    }

    // v2 THEM register IN THE GALAXY: verified labor stories placed in the cloud, read by proximity
    // as you fly past — legible, and on every dome node (not a primary-only head-locked caption).
    const Vec3f camPos(s.navPos[0], s.navPos[1], s.navPos[2]);
    stories_.draw(g, camPos, camR, camU);
  }

  void onSound(AudioIOData& io) override {
    if (!isPrimary()) return;          // primary-only audio
    audio_.render(io);
  }

  // SPACE toggles the manual dive to the core vessel — summon the splats on demand.
  bool onKeyDown(const Keyboard& k) override {
    if (k.key() == ' ') { diveActive_ = !diveActive_; return true; }
    // audition the 5 acts: keys 1-5 jump the cycle clock to each act's middle (audio + visuals).
    if (k.key() >= '1' && k.key() <= '5') {
      static const float ACT_T[5] = {120.f, 285.f, 435.f, 555.f, 720.f};
      cycleClock_ = ACT_T[k.key() - '1'];
      return true;
    }
    return false;
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
