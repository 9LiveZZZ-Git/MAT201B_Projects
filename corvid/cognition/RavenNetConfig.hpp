#pragma once
#include <cstdint>

namespace corvid {

// RavenNet architecture config (spec §2.4.2).
// Tiny tier used for M3: d_obs=64, d_hidden=128, d_action=6, lora_rank=4.
// Medium tier (~10M params) is M5+.
struct RavenNetConfig {
    int d_obs      = 64;   // FixedEncoder output dim (= MEM_ENC_DIM)
    int d_hidden   = 128;  // trunk hidden width
    int d_action   = 6;    // action biases: [align, sep, cohere, pred, food, obs]
    int lora_rank  = 4;    // per-agent LoRA rank (r)
    int n_agents   = 256;  // adapter table rows (= N_POOL)
};

} // namespace corvid
