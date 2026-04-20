#pragma once
#include <cstdint>

namespace corvid {

static constexpr int MEM_ENC_DIM = 64;  // encoded vector dimension

// One episodic memory token (spec §2.2, §2.4.1).
// Written when a significant event is perceived; evicted by salience.
struct Memory {
    uint64_t id        = 0;       // monotonic global counter
    float    timestamp = 0.f;     // sim_time when written
    int      kind      = 0;       // MemoryKind index
    uint32_t place_id  = 0;       // Place cell where it occurred
    float    salience  = 1.f;     // higher = keep longer; decays over time
    float    vec[MEM_ENC_DIM] = {}; // encoded observation (fixed-encoder output)
};

} // namespace corvid
