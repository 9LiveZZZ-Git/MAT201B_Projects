#pragma once
// PPO rollout buffer — no torch dependency.
// Collects per-agent experience over ROLLOUT steps, computes GAE,
// outputs a flat TrainBatch consumed by RavenBrain::train_step.
// Spec §3.2: γ=0.99, λ=0.95, clip ε=0.2, rollout=24 steps @ 5 Hz.
#include "cognition/RavenBrain.hpp"
#include <array>
#include <algorithm>
#include <cmath>

namespace corvid {

static constexpr int PPO_ROLLOUT  = 24;  // 4.8 s @ 5 Hz per agent

// One timestep of experience for one agent
struct StepRecord {
    float   obs[ENC_DIM_CONST] = {};
    int     action    = 0;
    float   reward    = 0.f;
    float   value     = 0.f;    // V(s) from critic at this step
    float   logprob   = 0.f;    // log π(a|s)
    float   done      = 0.f;    // 1.0 if episode ended this step
    int64_t adapter_idx = 0;
};

template<int N_POOL, int ROLLOUT = PPO_ROLLOUT>
class PPOBuffer {
public:
    PPOBuffer(float gamma = 0.99f, float lam = 0.95f)
        : gamma_(gamma), lam_(lam) {}

    void push(int slot, const StepRecord& rec) {
        if (counts_[slot] >= ROLLOUT) return;
        bufs_[slot][counts_[slot]++] = rec;
    }

    // Mark last stored step for slot as terminal
    void mark_done(int slot) {
        if (counts_[slot] > 0)
            bufs_[slot][counts_[slot]-1].done = 1.f;
    }

    void reset_agent(int slot) { counts_[slot] = 0; }

    bool any_ready() const {
        for (int i = 0; i < N_POOL; ++i)
            if (counts_[i] >= ROLLOUT) return true;
        return false;
    }

    // Compute GAE and build TrainBatch from all full rollouts.
    // bootstrap_values[slot] = V(s_{T+1}); 0 if agent is dead.
    TrainBatch compute_batch(const float* bootstrap_values) const {
        TrainBatch b;
        for (int s = 0; s < N_POOL; ++s) {
            int T = counts_[s];
            if (T < ROLLOUT) continue;

            // GAE backward pass
            float gae = 0.f;
            float next_val = bootstrap_values[s];
            float advs[ROLLOUT];
            for (int t = T-1; t >= 0; --t) {
                const auto& step = bufs_[s][t];
                float not_done = 1.f - step.done;
                float delta = step.reward + gamma_ * next_val * not_done - step.value;
                gae = delta + gamma_ * lam_ * not_done * gae;
                advs[t] = gae;
                next_val = step.value;
            }

            // Per-rollout advantage normalisation (§3.2.1)
            float mean = 0.f;
            for (int t = 0; t < T; ++t) mean += advs[t];
            mean /= float(T);
            float var = 0.f;
            for (int t = 0; t < T; ++t) var += (advs[t]-mean)*(advs[t]-mean);
            float std = std::sqrt(var / float(T) + 1e-8f);

            for (int t = 0; t < T; ++t) {
                const auto& step = bufs_[s][t];
                b.obs.insert(b.obs.end(), step.obs, step.obs + ENC_DIM_CONST);
                b.actions.push_back(int32_t(step.action));
                b.advantages.push_back((advs[t] - mean) / std);
                b.returns.push_back(advs[t] + step.value);
                b.old_logprobs.push_back(step.logprob);
                b.adapter_idx.push_back(step.adapter_idx);
                ++b.N;
            }
        }
        return b;
    }

    void drain_ready() {
        for (int i = 0; i < N_POOL; ++i)
            if (counts_[i] >= ROLLOUT) counts_[i] = 0;
    }

private:
    float gamma_, lam_;
    std::array<std::array<StepRecord, ROLLOUT>, N_POOL> bufs_{};
    std::array<int, N_POOL> counts_{};
};

// log π(action | logits) under categorical softmax
inline float log_softmax_action(const float* logits, int action, int n) {
    float mx = logits[0];
    for (int i = 1; i < n; ++i) if (logits[i] > mx) mx = logits[i];
    float sum = 0.f;
    for (int i = 0; i < n; ++i) sum += std::exp(logits[i] - mx);
    return logits[action] - mx - std::log(sum + 1e-10f);
}

// Sample from softmax(logits) given uniform u ∈ [0,1)
inline int sample_softmax(const float* logits, int n, float u) {
    float mx = logits[0];
    for (int i = 1; i < n; ++i) if (logits[i] > mx) mx = logits[i];
    float probs[8]; float sum = 0.f;
    for (int i = 0; i < n; ++i) { probs[i] = std::exp(logits[i]-mx); sum += probs[i]; }
    float thr = u * sum, acc = 0.f;
    for (int i = 0; i < n; ++i) { acc += probs[i]; if (acc >= thr) return i; }
    return n-1;
}

} // namespace corvid
