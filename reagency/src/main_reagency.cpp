// World of Shadow Work — runtime.
// An autonomous, distributed AlloSphere piece: a multimodal CLIP galaxy that SURROUNDS the
// viewer (camera inside; webs central). The Conductor walks the k-NN graph LIVE; when it
// fixates an image, that photo surfaces at its node and dissolves into grains that stream
// into the galaxy (the human->machine cede). A morphing splat "vessel" sits at the core.
// Runtime is allolib-only; custom GLSL through al::ShaderProgram.
#include "al/app/al_App.hpp"
// Suppress allolib's own -Woverloaded-virtual (DistributedAppWithState<T>::start(uint16_t) intentionally
// overloads DistributedApp::start() — a vendored-framework design, not ours to change). The pragma must
// wrap THIS include: the warning's diagnostic location is inside the header, so a later pragma misses it.
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Woverloaded-virtual"
#elif defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Woverloaded-virtual"
#endif
#include "al/app/al_DistributedApp.hpp"
#if defined(__clang__)
#pragma clang diagnostic pop
#elif defined(__GNUC__)
#pragma GCC diagnostic pop
#endif
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
#include "viz/CreditLayer.hpp"
#include "viz/DreamLayer.hpp"
#include "viz/EmergencePlayer.hpp"
#include "viz/DetectionHUD.hpp"      // feature B: perpetual CV blob-detection overlay + running sort
#include "audio/AudioEngine.hpp"

#include <cmath>
#include <cstdint>
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
  CaptionLayer  captions_;                    // v2 T7: the machine's VOICE (ME/YOU/US, head-locked)
  bool          showVoice_ = false;            // artist's call: the overt head-locked confession is OFF
                                               // (the thesis carries via the visuals + in-galaxy THEM)
  StoryLayer    stories_;                      // v2: the THEM register IN THE GALAXY (proximity-read)
  CreditLayer   credits_;                      // v2 T8: persistent THEM credit-envelope (head-locked, never fades)
  DreamLayer    dreams_;                        // v2 Phase-2: diffusion dreams billboarded at their concatenated nodes
  EmergencePlayer emergence_;                   // v2 Phase-2: the "watch it think" forming-from-noise layer
  DetectionHUD  detect_;                        // feature B: perpetual CV blob-detection overlay + running sort
  bool          detectOn_ = true;               // feature B: always on (toggle removed; kept for the guard)

  // feature B/4: pick a deterministically-RANDOM class slot for a reticle, re-rolled on a synced time
  // grid so the detector looks like it is rapidly cycling candidate classifications. Pure function of
  // (node, floor(simTime*wordRate)) -> identical on every dome node. Falls back to the node's own
  // class slot if the label table is empty.
  int detectWordSlot_(int node, double simTime) const {
    const int nW = labels.slotCount();
    if (nW <= 0) return labels.slotForNode(node);
    const double wordRate = 3.0;                                  // ~3 candidate guesses / sec
    uint32_t a = uint32_t(node < 0 ? 0 : node) * 0x9E3779B1u;
    uint32_t b = uint32_t(int64_t(std::floor(simTime * wordRate)));
    uint32_t h = a ^ (b + 0x85EBCA6Bu + (a << 6) + (a >> 2));
    h ^= h >> 15; h *= 0x2C1B3C6Du; h ^= h >> 12; h *= 0x297A2D39u; h ^= h >> 15;
    return labels.slotByOrdinal(int(h % uint32_t(nW)));
  }
  // A: neural-firing field intensity (spawn density + brightness). 0 disables. 'n' cycles presets.
  // Identical on every node by construction (same compile-time default); only the primary reads keys,
  // so on the dome leave it at the default (it is NOT synced — see WoSWState note in the plan).
  float         fireIntensity_ = 1.0f;
  static constexpr float EMERGE_T    = 2.5f;    // seconds to play a dream's formation while attended
  static constexpr float EMERGE_NEAR = 2.2f;    // camera-distance under which a dream starts forming
  AudioEngine   audio_;                       // primary-only generative voice (M5)
  int           prevCurNode_ = -1;            // arrival edge-detect for audio events
  std::vector<int> heroNodes_;                // corpus image nodes that have an atlas thumbnail
  std::vector<int> dreamNodes_;               // type==2 dream nodes (for proximity emergence)
  int           heroCursor_ = 0;
  float         traceTimer_ = 0.f;
  static constexpr float TRACE_LIFE  = 9.0f;  // seconds a surfaced photo lives (lingers)
  static constexpr float CREDIT_LIFE = 300.f; // v2 T8: a THEM credit holds this long, then recycles to the next worker
  int           creditCursor_ = 0;            // cycles them.txt so every worker surfaces over a session
  bool          creditsOn_    = false;        // T8 disabled on the degenerate 877 preview (set in onCreate)
  // ----- v2 5-ACT NARRATIVE RE-SCORE -----
  // A ~14-min ritual cycle replaces the old 80s cos() tide. cycleClock_ advances with time but
  // SLOWS when the machine hesitates, so the ritual breathes with the mind. cycleT in [0,CYCLE)
  // maps to a piecewise depth envelope across the acts: I seduction -> II reading -> III extraction
  // -> IV turn (floor 0.10, the plunge) -> V integration. The COMFORT FAILS: Act V re-brightens only
  // partway (~0.40, the comfort-fail cap), never back to the Act-I seduction peak (0.95); the loop
  // seam is continuous (V end == I start, both 0.40) so it never hard-resets — a fresh viewer enters
  // at the same partial depth a departing one left on. depth: 1 = galaxy among the points, 0 = core.
  static constexpr float CYCLE = 840.f;
  double        cycleClock_ = 0.0;   // double: the accumulator bands the envelope/posterize over hours
  static float actEnvelope(float t, int& act) {            // depth(t) + act index (1..5)
    struct KF { float t, d; };
    static const KF kf[] = {
      {  0.f, 0.40f}, {120.f, 0.95f}, {210.f, 0.90f}, {360.f, 0.60f},
      {510.f, 0.15f}, {600.f, 0.10f}, {720.f, 0.30f}, {840.f, 0.40f},
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
  // v2 T9: the participatory Act-IV turn. Near-stillness for DWELL_T seconds in Act IV arms it; the
  // plunge deepens (turnDive_) and the YOU line fires ON that frame. Desktop = auto-orbit proxy;
  // dome = tracked head pose. (Single writer of s.depth preserved — turnDive_ folds into that line.)
  float         turnDive_     = 0.f;            // eased 0..1 once the turn arms
  Vec3d         prevNavPos_   = Vec3d(0, 0, 16);
  Quatd         prevNavQuat_  = Quatd(1, 0, 0, 0);
  static constexpr float DWELL_T     = 3.0f;    // seconds of near-stillness in Act IV to arm the turn
  static constexpr float DWELL_EPS_P = 0.02f;   // positional delta/frame under which the pose "dwells"
  static constexpr float DWELL_EPS_A = 0.01f;   // angular (1-|dot|) delta/frame threshold
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
    captions_.init(assetDir);                             // v2 T7 voice (ME/YOU/US confession)
    stories_.init(assetDir);                              // v2 THEM stories in the galaxy
    credits_.init(assetDir);                              // v2 T8 persistent THEM credit-envelope
    dreams_.init(assetDir);                               // v2 Phase-2 diffusion dreams at their nodes
    emergence_.init(assetDir);                            // v2 Phase-2 emergence "watch it think"
    dreams_.setEmergence(&emergence_);                    // every dream forms-from-noise by proximity (all 192)
    detect_.init();                                       // feature B: CV detection HUD (no assets; shader only)
    audio_.init(assetDir, audioIO().framesPerSecond());   // primary-only; scale from manifest.json
    for (int i = 0; i < field.count(); ++i) {          // traceable corpus photos + dream nodes
      if (field.typeOf(i) == 0 && field.atlasOf(i) >= 0) heroNodes_.push_back(i);
      if (field.typeOf(i) == 2) dreamNodes_.push_back(i);
    }
    creditsOn_ = field.count() >= 3000;                   // T8 off on the degenerate 877 preview

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

  // v2 T8: promote a THEM worker credit into the persistent ring (called on image-dissolve in
  // extraction). The expired trace is only the TOKEN; the slot stores a worker index cycling
  // through them.txt so every cited worker surfaces over a session.
  void promoteCredit(WoSWState& s) {
    const int nW = credits_.count();
    if (nW <= 0) return;
    int slot = -1, oldest = 0;
    for (int i = 0; i < WoSWState::N_CREDIT; ++i) {
      if (s.creditNode[i] < 0 && slot < 0) slot = i;
      if (s.creditAge[i] > s.creditAge[oldest]) oldest = i;
    }
    if (slot < 0) slot = oldest;
    s.creditNode[slot] = creditCursor_ % nW;
    s.creditAge[slot]  = 0.f;
    creditCursor_++;
  }

  void onAnimate(double dt) override {
    auto& s = state();
    if (isPrimary()) {
      s.frame++;
      s.simTime += dt;

      // The mind still walks its own k-NN graph (it drives the vessel morph + hesitation
      // + future audio), but it no longer moves the camera — that read as distracting
      // zoom. The wander shows in the morphing vessel, not in camera motion.
      conductor.step(float(dt));
      s.curNode    = conductor.curNode();
      s.nextNode   = conductor.nextNode();
      s.hesitation = conductor.hesitation();

      // v2 Phase-2: emergence "watch it think" — play a dream forming from noise when the machine
      // ATTENDS its node, OR when the camera lingers NEAR one (fly close to a dream and watch it
      // form). Phase advances over EMERGE_T while active; clears otherwise.
      int wantDream = -1;
      if (s.curNode >= 0 && field.typeOf(s.curNode) == 2) {
        wantDream = s.curNode;
      } else {
        float best = EMERGE_NEAR; const auto cp = nav().pos();
        const Vec3f cpf(float(cp.x), float(cp.y), float(cp.z));
        for (int dn : dreamNodes_) {
          const float dd = (field.posOf(dn) - cpf).mag();
          if (dd < best) { best = dd; wantDream = dn; }
        }
      }
      if (wantDream >= 0) {
        if (s.emergeDream != wantDream) { s.emergeDream = wantDream; s.emergePhase = 0.f; }
        s.emergePhase = std::min(1.f, s.emergePhase + float(dt) / EMERGE_T);
      } else {
        s.emergeDream = -1;
      }
      const Vec3f focus = conductor.focusPos();
      s.focusPos[0] = focus.x; s.focusPos[1] = focus.y; s.focusPos[2] = focus.z;
      s.vesselKf   = float(conductor.visits()) + conductor.progress();

      // v2 5-act re-score: advance the cycle clock (slowed by hesitation), read the act + depth.
      // Everything below (trace cadence, the depth crossfade, the dive) subordinates to this.
      cycleClock_ += dt * (1.0 - 0.4 * double(s.hesitation));
      const float cycleT = float(std::fmod(cycleClock_, double(CYCLE)));
      int act; const float actDepth = actEnvelope(cycleT, act);   // act 1..5; depth 1=galaxy..0=core

      // AUDIO: on each NEW node arrival, sound the node's note + ignite an arpeggio over its
      // k-NN edges (the web "belongs-together" made audible). Pitch/timbre from type/density.
      if (s.curNode != prevCurNode_ && s.curNode >= 0) {
        // v2: record the walk — shift the trail (newest at [0]) so every node can draw the glowing
        // path the machine traced through the web.
        for (int i = WoSWState::N_TRAIL - 1; i > 0; --i) s.walkTrail[i] = s.walkTrail[i - 1];
        s.walkTrail[0] = s.curNode;
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
          if (s.traceAge[i] > TRACE_LIFE) {
            audio_.traceOff(i);
            // v2 T8: as the image dissolves in extraction (Act III+), the labor it rests on is
            // promoted into the persistent THEM credit-envelope — held outside "we", never fades.
            if (creditsOn_ && act >= 3) promoteCredit(s);
            s.traceNode[i] = -1;
          }
        }
      // v2 T8: age the THEM credits; at CREDIT_LIFE they don't vanish — they RECYCLE to the next
      // worker row, so THEM never disappears mid-session once the extraction has begun.
      if (creditsOn_)
        for (int i = 0; i < WoSWState::N_CREDIT; ++i)
          if (s.creditNode[i] >= 0) {
            s.creditAge[i] += float(dt);
            if (s.creditAge[i] > CREDIT_LIFE && credits_.count() > 0) {
              s.creditNode[i] = creditCursor_ % credits_.count();
              s.creditAge[i] = 0.f; creditCursor_++;
            }
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
        audio_.whisperChorus(labels.wordForNode(node), node, tpan, 0, 0.25f);   // 2b: n=0 -> conductor's cChorusN_; spread fans the ghosts
      }

      // Manual dive override (SPACE): ease toward the core, layered on the 5-act depth envelope
      // (actDepth, computed above). Single writer of s.depth.
      const float diveTarget = diveActive_ ? 1.f : 0.f;
      diveOverride_ += (diveTarget - diveOverride_) * std::min(1.f, float(dt) * 2.5f);  // ~0.4s ease
      // v2 T9: the gaze-dwell turn deepens the plunge (participatory), eased and layered onto the
      // SAME single depth write — no second writer. s.turnArmed is set by the dwell detector below.
      const float turnTarget = s.turnArmed ? 1.f : 0.f;
      turnDive_ += (turnTarget - turnDive_) * std::min(1.f, float(dt) * 1.5f);          // ~0.7s ease
      s.depth = actDepth * (1.f - diveOverride_) * (1.f - 0.5f * turnDive_);   // dwell sinks it further

      // v2 T10: glitchEnergy = affect SCOPED TO THE VOICE (the galaxy never reads it) — faint in the
      // reading/extraction acts, a brief spike on the armed turn. extractionDebt = the HAUNT LATCH:
      // it accrues at the turn and NEVER resets in a session (a faint posterize residue thereafter).
      {
        float ge = (act == 2 || act == 3) ? 0.10f : 0.f;
        if (act == 4 && s.turnArmed) ge = 0.60f;
        s.glitchEnergy += (ge - s.glitchEnergy) * std::min(1.f, float(dt) * 2.f);
        const float debtTarget = (act >= 4) ? (0.30f + 0.70f * turnDive_) : 0.f;
        if (debtTarget > s.extractionDebt) s.extractionDebt = debtTarget;   // monotonic — the haunt
      }

      // v2 T7: the machine's VOICE — OFF by artist's choice (showVoice_). The overt head-locked
      // confession captions are removed; the thesis carries via the visuals + the in-galaxy THEM.
      if (showVoice_) {
        captions_.update(act, float(dt));
        // YOU fires EXACTLY on the convergence frame (the dwell armed it); forceYou refuses before a ME.
        if (s.turnArmed && !s.youFired && captions_.forceYou()) s.youFired = 1;
        s.capCur = captions_.curId(); s.capTyped = captions_.typed(); s.capAlpha = captions_.alpha();
      }

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

      // v2 T9: participatory gaze/position DWELL. In Act IV, near-stillness for ~DWELL_T seconds ARMS
      // the turn — so "you did not look for them" is literally true of THIS viewer in THIS moment.
      // Desktop: the camera is auto-driven and barely moves in the Act-IV hold, so it arms after the
      // hold (the documented proxy); dome: the same delta keys off the tracked head pose. The arm/
      // fire latches reset at the loop seam (act -> 1). turnArmed feeds the depth write next frame.
      {
        const double dp = (Vec3d(p.x, p.y, p.z) - prevNavPos_).mag();
        const double qd = 1.0 - std::fabs(q.x * prevNavQuat_.x + q.y * prevNavQuat_.y
                                        + q.z * prevNavQuat_.z + q.w * prevNavQuat_.w);
        prevNavPos_ = Vec3d(p.x, p.y, p.z); prevNavQuat_ = q;
        if (act == 4 && dp < DWELL_EPS_P && qd < DWELL_EPS_A) s.dwell += float(dt) / DWELL_T;
        else                                                 s.dwell -= float(dt) / DWELL_T;
        s.dwell = s.dwell < 0.f ? 0.f : (s.dwell > 1.f ? 1.f : s.dwell);
        if (act == 4 && s.dwell >= 1.f) s.turnArmed = 1;
        if (act == 1) { s.turnArmed = 0; s.youFired = 0; s.dwell = 0.f; }   // new cycle at the seam
      }

      // AUDIO continuous controls + the 5-ACT orchestration: each act has its own palette
      // (I seduction -> II reading -> III extraction grind -> IV bare-pulse turn -> V haunted residue).
      audio_.update(float(dt), s.hesitation, s.depth, conductor.progress(),
                    panOf(Vec3f(s.focusPos[0], s.focusPos[1], s.focusPos[2])), act,
                    s.emergePhase, s.emergeDream >= 0);   // 2c: feed the SYNCED dream-emergence (sim thread, primary)
    } else {
      // apply the primary's exact pose (Quatd is w,x,y,z)
      nav().pos().set(s.navPos[0], s.navPos[1], s.navPos[2]);
      nav().quat() = Quatd(s.navQuat[3], s.navQuat[0], s.navQuat[1], s.navQuat[2]);
    }

    // The vessel morphs on every node from the synced keyframe position. *0.5 = HALF SPEED
    // (the splats were generating/morphing too fast); the synced s.vesselKf is unchanged so any
    // other reader stays consistent.
    vessel.update(s.vesselKf * 0.5f, float(s.simTime));
  }

  void onDraw(Graphics& g) override {
    auto& s = state();
    // Depth crossfade weights (synced depth, so identical on every dome node). dd = smoothstep,
    // vd = vessel-ness. As we descend (depth->0): galaxy dims, webs recede, the vessel cocoon
    // brightens and swells around the core; rising back out reverses it.
    const float depth = s.depth;
    const float dd = depth * depth * (3.f - 2.f * depth);
    const float vd = 1.f - dd;
    const float galBright = 0.45f + 0.55f * dd;   // v1 values (pixel size + brightness restored)
    const float webBright = 0.20f + 0.80f * dd;
    const float vesAlpha  = 0.70f + 0.30f * vd;   // solid crisp points (no-bloom alpha cloud, not additive glow)
    const float vesScale  = 4.0f + 1.0f * vd;     // BIG — fills the main screen (camera sits ~7 ahead, outside it)

    g.clear(0.03f, 0.03f, 0.05f);     // subtly-lifted void (reads better in the dome)
    webs.draw(g, webBright);          // similarity webs under the points
    // A: NEURONS FIRING IN THE WEB — the living electrical field over the WHOLE similarity graph,
    // biased denser around s.curNode (the machine's attention). Pure function of synced simTime +
    // curNode + the identically-loaded adjacency; identical on every dome node, no new sync state.
    webs.drawFiring(g, s.simTime, s.curNode, webBright, fireIntensity_);
    field.draw(g, 1.f, galBright, s.extractionDebt);   // T10: posterize haunt residue after the turn

    // v2: the LIVE connection — the machine visibly wiring its points along the CLIP web. The
    // current node's k-NN star (what it links this point to) + the committed cur->next edge with a
    // hot travelling pulse at the fixation head, drawn bright over the static web. Animates the
    // PRE-BAKED similarity (the offline CV that built the web); no runtime CV. Lights dream-node
    // edges too once the concatenated dreams are installed.
    if (s.curNode >= 0 && s.curNode < field.count()) {
      const Vec3f cur = field.posOf(s.curNode);
      const Vec3f nxt = (s.nextNode >= 0) ? field.posOf(s.nextNode) : cur;
      const Vec3f head(s.focusPos[0], s.focusPos[1], s.focusPos[2]);
      std::vector<Vec3f> nbr;
      const auto& adj = webs.adjacency();
      if (s.curNode < int(adj.size()))
        for (const auto& e : adj[s.curNode]) nbr.push_back(field.posOf(e.first));
      std::vector<Vec3f> trail;                       // the recently-walked path (newest first)
      for (int i = 0; i < WoSWState::N_TRAIL; ++i)
        if (s.walkTrail[i] >= 0 && s.walkTrail[i] < field.count())
          trail.push_back(field.posOf(s.walkTrail[i]));
      const float traceBright = galBright > 0.7f ? galBright : 0.7f;   // stays prominent into the descent
      webs.drawActive(g, cur, nxt, head, nbr, trail, traceBright);
    }

    // Camera-facing axes + eye (synced pose) — used by the vessel placement and all billboards below.
    Quatd q(s.navQuat[3], s.navQuat[0], s.navQuat[1], s.navQuat[2]);
    Vec3d rgt = q.rotate(Vec3d(1, 0, 0));
    Vec3d upv = q.rotate(Vec3d(0, 1, 0));
    Vec3d fwd = q.rotate(Vec3d(0, 0, -1));
    const Vec3f camR(float(rgt.x), float(rgt.y), float(rgt.z));
    const Vec3f camU(float(upv.x), float(upv.y), float(upv.z));
    const Vec3f camF(float(fwd.x), float(fwd.y), float(fwd.z));
    const Vec3f camPos(s.navPos[0], s.navPos[1], s.navPos[2]);

    // The "vessel" — the machine's splat hallucination. FORMED IN FRONT of the eye (centred in view,
    // tighter so the shape reads) rather than at the core, where the orbit camera sits inside the cloud.
    // The depth tide still controls its brightness/scale (faint out in the galaxy, swelling in descent).
    vessel.draw(g, camPos + camF * 6.0f, vesScale, vesAlpha);

    // CLASSIFICATION TEXT CULLED (Design C): the floating cluster-centroid word map was removed to
    // de-clutter the galaxy; the machine's per-object classification now lives only in the detection
    // HUD (stamped ABOVE each surfaced photo/dream). The type==1 word-DOTS in ParticleField are NOT
    // text and remain. (drawClusters intentionally not called.)

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
        // UNDER-PHOTO CLASS WORD CULLED (Design C, merged): the classifying label here was removed to
        // de-clutter. The class word is now stamped ABOVE the same photo by the detection HUD via
        // detectWordSlot_ (rapidly-cycling candidate class), so the classification is not lost — and is
        // drawn exactly once. The now-unused r01/lab fade-envelope locals are removed with it.
      }
    }

    // v2 T10: the Act-IV void gesture — a single dim-warm POSITIVE quad at the core drawn with
    // REVERSE-SUBTRACT blending, so it DARKENS a warm-shadowed hole as the turn converges (the cede
    // leaves a wound, not a tasteful sublime). Built-in mesh shader (dome-warp-safe); blend state is
    // RESTORED immediately after, or every later layer would be darkened too.
    if (turnConverge > 0.02f) {
      Mesh q; q.primitive(Mesh::TRIANGLE_STRIP);
      const Vec3f cc(0, 0, 0);
      const float rr = 1.2f + 1.6f * turnConverge;
      const Vec3f rx = camR * rr, uy = camU * rr;
      const float k = 0.28f * turnConverge;        // dim warm positive -> subtracted = a warm-shadow void
      q.vertex(cc - rx - uy); q.color(k, k * 0.7f, k * 0.45f);
      q.vertex(cc + rx - uy); q.color(k, k * 0.7f, k * 0.45f);
      q.vertex(cc - rx + uy); q.color(k, k * 0.7f, k * 0.45f);
      q.vertex(cc + rx + uy); q.color(k, k * 0.7f, k * 0.45f);
      g.meshColor();                               // revert to allolib's built-in per-vertex shader
      g.depthTesting(false); g.blending(true); g.blendSub();
      g.draw(q);
      g.blendTrans(); g.depthTesting(true);        // RESTORE (mandatory)
    }

    // v2 THEM register IN THE GALAXY: verified labor stories placed in the cloud, read by proximity
    // as you fly past — legible, and on every dome node (not a primary-only head-locked caption).
    stories_.draw(g, camPos, camR, camU);

    // v2 Phase-2: the diffusion DREAMS surface at their concatenated galaxy nodes (proximity-read),
    // brighter as the Act-IV turn converges (the confabulation rises as the archive melts). The
    // active-web trace lights their edges as the Conductor passes — the ML connecting its dream points.
    // ALL 192 dreams now form-from-noise BY PROXIMITY inside DreamLayer (it owns the EmergencePlayer):
    // each near dream crossfades emergence latents -> resolved image and re-dreams on a slow loop while
    // you linger. Deterministic from synced camPos + simTime, so no separate attended draw is needed.
    dreams_.draw(g, camPos, camR, camU, 0.60f + 0.40f * turnConverge, float(s.simTime));

    // -------- feature B: BLOB DETECTION + SORTING, perpetually "running" --------
    // Animate the PRE-BAKED CLIP classification as a live CV read-out. PURE function of synced
    // simTime + synced node selections + node key (cluster/density) -> identical on every dome node.
    if (detectOn_ && detect_.ready()) {
      const float hudBright = 0.55f + 0.45f * dd;   // fades with the same depth crossfade as the galaxy
      // (1) lock-on reticles on every SURFACED HERO PHOTO (the trace ring) — sits ON the object.
      for (int i = 0; i < WoSWState::N_TRACE; ++i) {
        if (s.traceNode[i] < 0) continue;
        const int node = s.traceNode[i];
        const Vec3f p = field.posOf(node);
        // feature 3 (matched to Design B size-swap): fit the box to the now-smaller corpus photo
        // (HumanTrace h = 0.78). With the +1.9x acquire overshoot the box starts a touch larger, then
        // snaps to the photo edge.
        detect_.boxHalf_ = 0.78f;                                  // corpus photo world half-size (smaller)
        detect_.drawReticle(g, p, camR, camU, node, s.simTime,
                            s.traceAge[i], field.densityOf(node), hudBright);
        // feature 4: stamp a deterministically-RANDOM class word that changes over time, so the
        // detector looks like it is rapidly classifying (hash(node, floor(t*wordRate)) -> ordinal).
        labels.drawSlot(g, p + camU * 1.20f, camR, camU, detectWordSlot_(node, s.simTime), 0.85f * hudBright, 0.30f);
      }
      // (2) lock-on reticles on the NEAR-CAMERA DREAM billboards (detecting the confabulations).
      {
        int shown = 0;
        for (int dn : dreamNodes_) {
          if (shown >= 3) break;                                  // cap: at most 3 dream reticles
          const Vec3f dp = field.posOf(dn);
          const float dist = (dp - camPos).mag();
          if (dist > 4.0f) continue;                              // only the close ones
          // feature 3: dreams are ~3/4 size (DreamLayer WHALF 0.60 per artist), so the reticle matches.
          detect_.boxHalf_ = 0.60f;
          const float age = (4.0f - dist) * 0.6f;                 // acquire from proximity; no clock
          detect_.drawReticle(g, dp, camR, camU, dn, s.simTime, age, field.densityOf(dn), hudBright);
          // feature 4: deterministically-random rapid class word for the dream too.
          labels.drawSlot(g, dp + camU * 1.05f, camR, camU, detectWordSlot_(dn, s.simTime), 0.75f * hudBright, 0.28f);
          ++shown;
        }
      }
      // (3) the running SORT STRIP: a fixed row of markers re-ordering by cluster. Sampled from the
      // active trace nodes + a deterministic spread of galaxy indices so the keys are real + synced.
      {
        static constexpr int NS = 16;
        float keys[NS], hue[NS];
        const int cnt = field.count();
        for (int k = 0; k < NS; ++k) {
          int node;
          if (k < WoSWState::N_TRACE && s.traceNode[k] >= 0) node = s.traceNode[k];
          else node = cnt > 0 ? int((uint64_t(k) * 2654435761u) % uint32_t(cnt)) : 0;  // deterministic spread
          const int cl = field.clusterOf(node);
          keys[k] = float(cl < 0 ? 0 : cl) + field.densityOf(node);   // sort key
          hue[k]  = float((cl < 0 ? k : cl) % 12) / 12.f;             // stable colour by cluster
        }
        detect_.drawSortStrip(g, camPos, camF, camR, camU, keys, hue, NS, s.simTime, hudBright);
      }
    }

    // v2 T8: persistent THEM credit-envelope — head-locked, accrues as images dissolve in
    // extraction, never fades below the floor. Differential persistence vs the warm dissolving
    // voice/US is the false-equivalence guard (only honest PAIRED with the Act-V comfort-fail).
    if (creditsOn_)
      credits_.draw(g, camPos, camF, camR, camU, s.creditNode, s.creditAge,
                    WoSWState::N_CREDIT, CREDIT_LIFE);

    // v2 T7: the machine's VOICE is OFF (showVoice_) — the overt narration captions are removed by
    // artist's choice; the in-galaxy THEM stories + the credit-envelope carry the legible referent.
    if (showVoice_) {
      captions_.applySynced(s.capCur, s.capTyped, s.capAlpha);
      captions_.draw(g, camPos, camF, camR, camU, s.glitchEnergy);   // T10: glitch scoped to the voice
    }
  }

  void onSound(AudioIOData& io) override {
    if (!isPrimary()) return;          // primary-only audio
    audio_.render(io);
  }

  // SPACE toggles the manual dive to the core vessel — summon the splats on demand.
  bool onKeyDown(const Keyboard& k) override {
    if (k.key() == ' ') { diveActive_ = !diveActive_; return true; }
    // TOGGLES REMOVED (merged A+B+C): the CV detection HUD and the neural firing are ALWAYS ON at
    // their defaults now (artist's choice). detectOn_ stays true and fireIntensity_ stays 1.0f so
    // every dome node matches; the 'd' and 'n' key handlers were intentionally deleted. SPACE dive
    // and the 1-5 act-audition keys are preserved.
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
