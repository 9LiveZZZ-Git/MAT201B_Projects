#include "Perception.hpp"
#include <cmath>
#include <random>

namespace corvid {

void FixedEncoder::init() {
    std::mt19937 rng(SEED_ENCODER);
    // He init scale for ReLU: sqrt(2 / fan_in)
    const float scale = std::sqrt(2.f / float(FEAT_DIM));
    std::normal_distribution<float> dist(0.f, scale);
    for (int o = 0; o < ENC_DIM; ++o) {
        for (int i = 0; i < FEAT_DIM; ++i)
            W[o][i] = dist(rng);
        b[o] = 0.f;  // zero bias — keeps encoded vectors zero-centred at init
    }
}

void FixedEncoder::encode(const float feat[FEAT_DIM], float out[ENC_DIM]) const {
    for (int o = 0; o < ENC_DIM; ++o) {
        float acc = b[o];
        for (int i = 0; i < FEAT_DIM; ++i)
            acc += W[o][i] * feat[i];
        out[o] = acc > 0.f ? acc : 0.f;  // ReLU
    }
}

ObsVec buildObsVec(const PerceptInput& in) {
    ObsVec obs;
    float* v = obs.v;
    const float wh = in.world_half;
    const float ms = in.max_speed > 0.f ? in.max_speed : 1.f;

    // [0-2] position normalized to [-1, 1]
    auto pos = in.agent.nav.pos();
    v[0] = float(pos.x) / wh;
    v[1] = float(pos.y) / wh;
    v[2] = float(pos.z) / wh;

    // [3-5] velocity normalized
    v[3] = in.vel.x / ms;
    v[4] = in.vel.y / ms;
    v[5] = in.vel.z / ms;

    // [6] energy
    v[6] = in.agent.energy;

    // [7-10] affect (hunger, fear, fatigue, curiosity)
    v[7]  = in.agent.affect[0];
    v[8]  = in.agent.affect[1];
    v[9]  = in.agent.affect[2];
    v[10] = in.agent.affect[3];

    // [11] neighbor count normalized
    v[11] = float(in.n_neighbors) / 10.f;

    // [12-14] average neighbor relative position normalized
    v[12] = in.avg_nb_rel.x / wh;
    v[13] = in.avg_nb_rel.y / wh;
    v[14] = in.avg_nb_rel.z / wh;

    // [15-17] nearest food direction (unit or zero)
    v[15] = in.food_dir.x;
    v[16] = in.food_dir.y;
    v[17] = in.food_dir.z;

    // [18] nearest food distance normalized [0,1]
    v[18] = in.food_dist;

    // [19] has food nearby
    v[19] = in.has_food ? 1.f : 0.f;

    // [20-22] nearest predator direction
    v[20] = in.pred_dir.x;
    v[21] = in.pred_dir.y;
    v[22] = in.pred_dir.z;

    // [23] nearest predator distance normalized
    v[23] = in.pred_dist;

    // [24] has predator nearby
    v[24] = in.has_predator ? 1.f : 0.f;

    // [25-28] key place event counts (normalized by /10)
    v[25] = in.place.event_counts[1] / 10.f;   // MK_FOOD
    v[26] = in.place.event_counts[0] / 10.f;   // MK_PREDATOR
    v[27] = in.place.event_counts[8] / 10.f;   // MK_BIRTH
    v[28] = in.place.event_counts[5] / 10.f;   // MK_DEATH_WITNESSED

    // [29] place avg_valence
    v[29] = in.place.avg_valence;

    // [30] place novelty_score
    v[30] = in.place.novelty_score;

    // [31] agent age normalized (60s ≈ mature)
    // age computed by caller putting it in — approximate via energy proxy
    v[31] = std::min(1.f, (in.agent.energy));

    return obs;
}

float memSalience(int kind, float energy_before, float energy_after) {
    float base = 0.3f;
    // energy change magnitude → salience boost
    float de = std::abs(energy_after - energy_before);
    base += de * 2.f;
    // kind-specific bonuses
    if (kind == 0 /* MK_PREDATOR */ || kind == 5 /* MK_DEATH_WITNESSED */)
        base += 0.4f;
    if (kind == 8 /* MK_BIRTH */)
        base += 0.2f;
    return std::min(1.f, base);
}

} // namespace corvid
