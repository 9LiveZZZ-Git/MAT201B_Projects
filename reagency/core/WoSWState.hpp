#pragma once
// World of Shadow Work — broadcast state (POD) for al::DistributedAppWithState.
//
// The ENTIRE per-frame sync payload. Heavy geometry (the particle buffer, the web
// edges) is loaded identically on every node at onCreate(); only these scalars are
// broadcast, and all motion is computed in GLSL from synced uniforms. Tiny (~64 B),
// so it fits a small UDP packet with no fragmentation. Renderers DROP any packet
// whose frame <= the last applied frame (in-order guarantee).
#include <cstdint>

namespace wosw {

struct WoSWState {
  uint32_t frame   = 0;        // monotonic; renderers drop stale/duplicate packets
  float    simTime = 0.f;
  float    seed    = 1.f;      // the dream seed (reproducible captures)

  // Camera — primary computes; ALL nodes apply to nav() before drawing.
  float    navPos[3]  = {0.f, 0.f, 16.f};
  float    navQuat[4] = {0.f, 0.f, 0.f, 1.f};  // x, y, z, w

  float    depth = 1.f;        // shell crossfade: 0 = vessel ... 1 = galaxy (M0 = galaxy)

  // Conductor / legibility (M2+).
  int32_t  curNode  = -1;      // node the machine is attending
  int32_t  nextNode = -1;      // candidate it is about to commit to
  float    hesitation = 0.f;   // = entropy of the next-step distribution
  float    focusPos[3] = {0.f, 0.f, 0.f};  // current fixation point (camera + marker)

  // Human trace (M2+): which image is surfacing, and how strongly.
  float    traceNode  = -1.f;
  float    traceAlpha = 0.f;
};

}  // namespace wosw
