#pragma once
// World of Shadow Work — broadcast state (POD) for al::DistributedAppWithState.
//
// The ENTIRE per-frame sync payload. Heavy geometry (the particle buffer, the web
// edges) is loaded identically on every node at onCreate(); only these scalars are
// broadcast, and all motion is computed in GLSL from synced uniforms. Tiny (~244 B),
// so it fits a small UDP packet with no fragmentation. Renderers DROP any packet
// whose frame <= the last applied frame (in-order guarantee).
#include <cstdint>

namespace wosw {

struct WoSWState {
  uint32_t frame   = 0;        // monotonic; renderers drop stale/duplicate packets
  float    seed    = 1.f;      // the dream seed (reproducible captures)
  double   simTime = 0.0;      // double: float bands the act-envelope/posterize over a multi-hour install

  // Camera — primary computes; ALL nodes apply to nav() before drawing.
  float    navPos[3]  = {0.f, 0.f, 16.f};
  float    navQuat[4] = {0.f, 0.f, 0.f, 1.f};  // x, y, z, w

  float    depth = 1.f;        // shell crossfade: 0 = vessel ... 1 = galaxy (M0 = galaxy)

  // Conductor / legibility (M2+).
  int32_t  curNode  = -1;      // node the machine is attending
  int32_t  nextNode = -1;      // candidate it is about to commit to
  float    hesitation = 0.f;   // = entropy of the next-step distribution
  float    focusPos[3] = {0.f, 0.f, 0.f};  // current fixation point (legibility)
  float    vesselKf    = 0.f;              // continuous vessel keyframe position (morph)

  // Human trace ring — multiple photos surfacing at their node positions (sphere edition).
  static constexpr int N_TRACE = 6;
  int32_t  traceNode[N_TRACE] = {-1, -1, -1, -1, -1, -1};
  float    traceAge[N_TRACE]  = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f};

  // v2 T8 — persistent THEM credit ring (mirrors N_TRACE). A node promoted here on trace
  // expiry is the *token* that lights a credit slot; the worker row shown cycles through
  // them.txt. Credits never free to -1 (recycle at CREDIT_LIFE) so THEM never vanishes.
  static constexpr int N_CREDIT = 6;
  int32_t  creditNode[N_CREDIT] = {-1, -1, -1, -1, -1, -1};
  float    creditAge[N_CREDIT]  = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f};

  // v2 T9 — participatory gaze/position-dwell turn (Act IV).
  float    dwell     = 0.f;    // [0,1] dwell accumulation (primary writes; renderers read)
  uint8_t  turnArmed = 0;      // latched true once dwell crossed threshold this cycle
  uint8_t  youFired  = 0;      // latched true the frame the YOU line was triggered
  uint8_t  _pad9[2]  = {0, 0}; // keep the following floats aligned

  // v2 T10 — affect floats. glitchEnergy is scoped to the VOICE layer only (galaxy stays
  // clean); extractionDebt is a monotonic haunt latch that NEVER resets in a session.
  float    glitchEnergy   = 0.f;
  float    extractionDebt = 0.f;

  // v2 T7 — caption sync, so the dome shows ONE synchronized voice (not per-node clock drift).
  int32_t  capCur   = -1;      // active caption id (or -1)
  float    capTyped = 0.f;     // glyphs revealed so far (drives uRevealX + cursor)
  float    capAlpha = 0.f;     // act-scheduled fade

  // v2 Phase-2 — emergence "watch it think": the dream node the machine is attending + its
  // formation phase (0 = first captured latent .. 1 = formed). -1 when not attending a dream.
  int32_t  emergeDream = -1;
  float    emergePhase = 0.f;

  // v2 — the WALK trail: the last N nodes the Conductor arrived at (newest first), so every node
  // can draw the glowing path the machine traced through the web (the ML visibly connecting points).
  static constexpr int N_TRAIL = 10;
  int32_t  walkTrail[N_TRAIL] = {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1};
};

}  // namespace wosw
