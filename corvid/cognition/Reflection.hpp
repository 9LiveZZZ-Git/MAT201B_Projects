#pragma once
// Shared types for per-agent reflection (spec §2.5.2).
// Always compiled in; heuristicReflect() is the fallback when LLM unavailable.
#include "core/Goal.hpp"
#include <cstdint>
#include <cstring>

namespace corvid {

// Snapshot of one agent sent TO the reflection system
struct ReflectJob {
    int      slot      = -1;
    uint32_t agent_id  = 0;
    float    sim_time  = 0.f;
    float    energy    = 0.f;
    float    birth_t   = 0.f;
    char     digest[512] = {};  // text digest of recent memories + events
};

// Per-agent result returned FROM the reflection system
struct ReflectResult {
    int       slot      = -1;
    uint32_t  agent_id  = 0;
    GoalEntry goals[MAX_GOALS];
    int       n_goals   = 0;
    char      text[256] = {};   // raw output for HUD display
    bool      from_llm  = false;
};

// Heuristic fallback (energy-based; always available even with LLM off)
ReflectResult heuristicReflect(const ReflectJob& job);

} // namespace corvid
