#pragma once
#include "core/Agent.hpp"
#include "core/Memory.hpp"
#include "core/Place.hpp"
#include "environment/Entity.hpp"
#include "al/math/al_Vec.hpp"
#include <array>
#include <cstdint>
#include <memory>
#include <vector>

namespace corvid {

// Feature vector layout (FEAT_DIM = 32 floats, spec §2.4.1).
// Packed in a fixed order so the seeded encoder weights are deterministic.
static constexpr int FEAT_DIM    = 32;
static constexpr int ENC_DIM     = 64;   // = MEM_ENC_DIM
static constexpr uint32_t SEED_ENCODER = 3233857602u;  // "corvid42" seeded

// Fixed-seed linear encoder: ReLU(W·feat + b), W∈R^{64×32}, b∈R^{64}.
// Weights derived from std::mt19937(SEED_ENCODER), scaled N(0, 1/√32).
// Non-trainable; kept fixed so encoded vectors are comparable across runs.
struct FixedEncoder {
    float W[ENC_DIM][FEAT_DIM];
    float b[ENC_DIM];

    void init();  // defined in Perception.cpp
    void encode(const float feat[FEAT_DIM], float out[ENC_DIM]) const;
};

// Per-agent observation feature vector (32 floats).
// Caller fills this; encode() converts it to a 64-d memory vector.
struct ObsVec {
    float v[FEAT_DIM] = {};
};

// Build the raw feature vector for one agent.
// Requires live agent, its current place, neighbor positions and velocities,
// nearest food/predator found in the frame.
struct PerceptInput {
    const Agent&   agent;
    const al::Vec3f vel;       // agent velocity (from parallel vel array)
    const Place&   place;
    float          world_half;
    float          max_speed;
    // Neighbor summary (precomputed by caller)
    int            n_neighbors = 0;
    al::Vec3f      avg_nb_rel  = {0,0,0};
    al::Vec3f      avg_nb_vel  = {0,0,0};
    // Nearest food
    bool           has_food    = false;
    al::Vec3f      food_dir    = {0,0,0};
    float          food_dist   = 1.f;   // normalized [0,1]
    // Nearest predator
    bool           has_predator = false;
    al::Vec3f      pred_dir    = {0,0,0};
    float          pred_dist   = 1.f;   // normalized [0,1]
};

ObsVec buildObsVec(const PerceptInput& in);

// Salience score for a memory event (used on push to ring).
// Higher for dangerous/nutritive events; lower for routine boids movement.
float memSalience(int kind, float energy_before, float energy_after);

} // namespace corvid
