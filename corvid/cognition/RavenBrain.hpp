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
    std::vector<float>   obs;             // [N * ENC_DIM_CONST]
    std::vector<int32_t> actions;         // [N]
    std::vector<float>   advantages;      // [N] — GAE-normalised
    std::vector<float>   returns;         // [N] — advantage + value
    std::vector<float>   old_logprobs;    // [N] — log π_old(a|s)
    std::vector<int64_t> adapter_idx;     // [N] — adapter slot per sample
    std::vector<float>   teaching_scale;  // [N] — M5: 3.0 for juveniles, 1.0 otherwise
    int N = 0;
};

struct RavenBrain {
    // Initialize trunk + adapter table.
    bool init(const RavenNetConfig& cfg);

    // Batched inference (spec §2.4.2.a). Hand-rolled forward, no autograd.
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

    // Part B: forward that also exposes the LoRA-augmented hidden activations
    // (post-ReLU, d_hidden per sample), for the tuned-lens readout (Lens.hpp).
    // out_hidden is N * d_hidden floats. Any of the out_* pointers may be null.
    void forward_tap(const float*   obs_flat,
                     const int64_t* adapter_idx,
                     int            N,
                     float*         out_biases,   // N * d_action  (may be null)
                     float*         out_values,   // N             (may be null)
                     float*         out_hidden);  // N * d_hidden  (may be null)

    // M5: copy parent adapter rows into child slot then add scale-aware Gaussian noise.
    // If parent_b_slot >= 0, applies block-level uniform crossover before mutation.
    // sigma_base = 0.01 per spec §3.10.11 (ES midpoint).
    void inherit_adapter(int child_slot, int parent_a_slot,
                         int parent_b_slot = -1, float sigma_base = 0.01f);

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
