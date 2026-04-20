#pragma once
#include "al/math/al_Vec.hpp"
#include <array>
#include <cmath>
#include <cstring>

namespace corvid {

// Place grid — the spatial memory substrate (spec §2.2.2).
// 8³ cells for the alpha demo (spec target is 32³; bump PLACE_GRID_N later).
// Cell size = W / PLACE_GRID_N; world is [-W/2, W/2]³ centered at origin.

static constexpr int   PLACE_GRID_N     = 8;
static constexpr int   PLACE_GRID_CELLS = PLACE_GRID_N * PLACE_GRID_N * PLACE_GRID_N;

struct Place {
    uint32_t  id          = 0;
    al::Vec3f center;                   // cell centre in world coords

    float     event_counts[16]  = {};   // per MemoryKind (spec §2.2.1), EMA-decayed
    float     avg_valence       = 0.f;  // running EMA of interaction valences
    float     novelty_score     = 0.f;  // normalised activity; drives render brightness

    void reset() {
        std::memset(event_counts, 0, sizeof(event_counts));
        avg_valence  = 0.f;
        novelty_score = 0.f;
    }
};

// Compute the place-grid index for a world position.
// world_half = W/2 (half the world side length).
inline int placeIndex(const al::Vec3f& pos, float world_half) {
    const float cell = (2.f * world_half) / PLACE_GRID_N;
    int px = static_cast<int>(std::floor((pos.x + world_half) / cell));
    int py = static_cast<int>(std::floor((pos.y + world_half) / cell));
    int pz = static_cast<int>(std::floor((pos.z + world_half) / cell));
    px = std::max(0, std::min(px, PLACE_GRID_N - 1));
    py = std::max(0, std::min(py, PLACE_GRID_N - 1));
    pz = std::max(0, std::min(pz, PLACE_GRID_N - 1));
    return px + py * PLACE_GRID_N + pz * PLACE_GRID_N * PLACE_GRID_N;
}

// Initialise a Place array from the world parameters.
inline void initPlaces(std::array<Place, PLACE_GRID_CELLS>& places, float world_half) {
    const float cell = (2.f * world_half) / PLACE_GRID_N;
    for (int pz = 0; pz < PLACE_GRID_N; ++pz)
    for (int py = 0; py < PLACE_GRID_N; ++py)
    for (int px = 0; px < PLACE_GRID_N; ++px) {
        int idx = px + py * PLACE_GRID_N + pz * PLACE_GRID_N * PLACE_GRID_N;
        auto& pl  = places[idx];
        pl.reset();
        pl.id     = static_cast<uint32_t>(idx);
        pl.center = {
            -world_half + (px + 0.5f) * cell,
            -world_half + (py + 0.5f) * cell,
            -world_half + (pz + 0.5f) * cell
        };
    }
}

// Write an event at a world position into the place grid.
// kind: MemoryKind index [0,15].  valence: [-1,1].
inline void writePlace(std::array<Place, PLACE_GRID_CELLS>& places,
                       const al::Vec3f& pos, int kind, float valence,
                       float world_half)
{
    int idx = placeIndex(pos, world_half);
    auto& pl = places[idx];
    int k = kind & 0xF;
    pl.event_counts[k] = 0.999f * pl.event_counts[k] + 1.0f;
    pl.avg_valence      = 0.99f * pl.avg_valence + 0.01f * valence;
    // novelty: sum of all event counts, clamped to [0,1]
    float sum = 0.f;
    for (int i = 0; i < 16; ++i) sum += pl.event_counts[i];
    pl.novelty_score = std::min(1.f, sum * 0.05f);
}

// Decay all place event counts (call once per sim tick).
inline void decayPlaces(std::array<Place, PLACE_GRID_CELLS>& places, float decay) {
    for (auto& pl : places)
        for (int k = 0; k < 16; ++k)
            pl.event_counts[k] *= decay;
}

} // namespace corvid
