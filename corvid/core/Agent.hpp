#pragma once
#include "al/io/al_ControlNav.hpp"
#include "al/math/al_Vec.hpp"

namespace corvid {

// Per-agent state for M1+ (spec §2.2.5).
// Stored in a fixed pool (std::array<Agent, N_POOL>) with a free-list; never
// in a vector that could reallocate.  All cross-thread references use an
// AgentHandle {slot, generation} rather than raw indices.
struct Agent {
    // Identity
    uint32_t  id          = 0;
    uint32_t  lineage_id  = 0;      // shared with descendants; visual grouping
    uint32_t  parent_a    = 0;
    uint32_t  parent_b    = 0;
    uint32_t  generation  = 0;      // incremented on slot reuse (handle validation)

    // Lifecycle
    float     birth_t     = 0.f;
    float     death_t     = -1.f;   // -1 while alive
    bool      live        = false;

    // Physiology
    float     energy      = 1.0f;   // [0,1]; 0 → death
    float     last_reproduce_t = -999.f;

    // Cognition (M2+): affect vector [hunger, fear, fatigue, curiosity]
    al::Vec4f affect      = {0.f, 0.f, 0.f, 0.f};

    // Motion primitive (spec §1.4)
    al::Nav   nav;

    // Visual helpers (not serialised)
    float flash_timer     = 0.f;    // >0 while birth/death flash active
    int   flash_kind      = 0;      // 0=birth(green), 1=death(red)
};

// Lightweight cross-thread reference (spec §2.2.0.a).
struct AgentHandle {
    uint32_t slot       = 0;
    uint32_t generation = 0;
};

} // namespace corvid
