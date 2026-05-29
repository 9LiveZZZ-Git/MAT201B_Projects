#pragma once
// Broadcast state for distributing the full corvid_m1 demo to the AlloSphere
// (Part B, Phase 10). POD blob sent by the primary (simulator) each frame and
// received by every renderer node, so all dome projectors show identical agents,
// thinking skybox, and crow splats.
//
// Design (per docs/REWRITE_PLAN.md Phase 10):
//   - Primary computes the sim + RavenNet + tuned lens; replicas only DRAW.
//   - Everything the renderer needs is here and POD: agent transforms, a
//     deterministic color seed (lineage id) + energy so each node recomputes the
//     same color, the compact ThoughtVector, the focused agent, and sim_time.
//   - Kept small (~7 KB, in the proven range of SimpleSharedState's 5.6 KB) so it
//     fits the UDP state datagram.
#include "cognition/Lens.hpp"   // ThoughtVector (POD)
#include <cstdint>

namespace corvid {

// Max agents broadcast to renderers (visual subset; matches SimpleSharedState scale).
static constexpr int VIZ_MAX_AGENTS = 200;

struct AgentXform {
    float    pos[3]   = {0, 0, 0};
    float    quat[4]  = {0, 0, 0, 1};   // x,y,z,w
    uint32_t lineage  = 0;              // deterministic hue seed (matches desktop)
    float    energy   = 0.f;            // brightness
    float    flash    = 0.f;            // flash timer (>0 => birth/death flash)
    int32_t  flashKind= 0;             // 0 birth(green), 1 death(red)
    int32_t  live     = 0;             // 1 if alive
};

struct CorvidVizState {
    uint32_t      tick      = 0;
    float         sim_time  = 0.f;
    int32_t       n_agents  = 0;        // valid entries in xform[]
    int32_t       focus_idx = -1;       // index into xform[] of focused agent
    ThoughtVector thought;              // drives skybox + splats on every node
    AgentXform    xform[VIZ_MAX_AGENTS];
};

} // namespace corvid
