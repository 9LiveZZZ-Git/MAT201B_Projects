#pragma once
// PIMPL wrapper — hides all torch headers from the sim layer.
// Spec §2.4.2; M3: forward-only. M4: adds train_step (adapter LoRA B updated via PPO).
#include "RavenNetConfig.hpp"
#include <cstdint>
#include <vector>

namespace corvid {

// Must match Perception.hpp ENC_DIM — duplicated here so training/ can include
// this header without pulling in all of cognition/Perception.hpp.
static constexpr int ENC_DIM_CONST = 64;

// Flat batch produced by PPOBuffer::compute_batch, consumed by RavenBrain::train_step.
struct TrainBatch {
    std::vector<float>   obs;           // [N * ENC_DIM_CONST]
    std::vector<int32_t> actions;       // [N]
    std::vector<float>   advantages;    // [N] — GAE-normalised
    std::vector<float>   returns;       // [N] — advantage + value
    std::vector<float>   old_logprobs;  // [N] — log π_old(a|s)
    std::vector<int64_t> adapter_idx;   // [N] — adapter slot per sample
    int N = 0;
};

struct RavenBrain {
    // Initialize trunk + adapter table.
    bool init(const RavenNetConfig& cfg);

    // Batched inference (spec §2.4.2.a). torch::InferenceMode inside.
    // obs_flat   : N × d_obs floats row-major
    // adapter_idx: N int64s
    // out_biases : N × d_action floats (written)
    // out_values : N floats (written)
    void forward(const float*   obs_flat,
                 const int64_t* adapter_idx,
                 int            N,
                 float*         out_biases,
                 float*         out_values);

    // PPO adapter update (M4). Unfreezes B_all_, runs one PPO gradient step.
    // Returns mean KL(old || new). batch.N must be >= 4.
    float train_step(const TrainBatch& batch);

    float last_ms          = 0.f;
    float last_kl          = 0.f;
    float last_policy_loss = 0.f;
    float last_value_loss  = 0.f;

    RavenNetConfig cfg;

    ~RavenBrain();
    struct Impl;
    Impl* impl_ = nullptr;
};

} // namespace corvid
