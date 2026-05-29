#pragma once
// Tuned-lens "thinking" readout (Part B, Phase 6) — allolib-native, no deps.
//
// Interpretability probe over the hand-rolled RavenNet (RavenBrain). A learned
// affine probe P: R^{d_hidden} -> R^{d_action} decodes the LoRA-augmented hidden
// activations into an action distribution. Trained online (cross-entropy) to
// match the policy head's own softmax — this is the "tuned lens" (a learned
// decoder) rather than a raw "logit lens" (reusing the output head directly).
//
// The compact ThoughtVector it produces is what drives the generative skybox
// (Phase 7) and the crow splat cloud (Phase 8), and what gets broadcast to the
// dome render nodes (Phase 10). Keep it POD + small.
#include "core/Goal.hpp"
#include <cstdint>
#include <vector>

namespace corvid {

static constexpr int LENS_ACT = 6;  // action dim (= RavenNetConfig::d_action)

// Compact, broadcast-friendly summary of one agent's "thought".
struct ThoughtVector {
    float    action[LENS_ACT] = {};  // decoded action distribution (sums ~1)
    float    entropy   = 0.f;        // normalized [0,1]; high = uncertain/"hallucinating"
    float    confidence= 0.f;        // 1 - entropy
    float    value     = 0.f;        // critic estimate (raw)
    int32_t  dominant  = 0;          // argmax action index
    int32_t  goal      = int(GoalKind::EXPLORE);  // dominant goal id
};

struct Lens {
    // probe weights: P[a * d_hidden + h], bias[a]
    int d_hidden = 0;
    std::vector<float> P;
    std::vector<float> bp;

    // Adam state
    std::vector<float> mP, vP, mb, vb;
    long  adam_t = 0;

    void init(int d_hidden_);

    // Decode hidden activations -> action distribution (softmax of probe).
    // hidden: d_hidden floats. out: LENS_ACT floats.
    void decode(const float* hidden, float* out_dist) const;

    // One supervised step: train the probe to match target_dist (the policy
    // head's softmax) given hidden activations. Returns mean cross-entropy.
    // hidden_flat: N * d_hidden ; target_flat: N * LENS_ACT.
    float train_step(const float* hidden_flat, const float* target_flat, int N);

    // Build a ThoughtVector from one agent's hidden activations + critic value.
    ThoughtVector think(const float* hidden, float value, int dominant_goal) const;
};

} // namespace corvid
