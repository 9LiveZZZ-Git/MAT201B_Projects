#pragma once
// PIMPL wrapper — hides all torch headers from the sim layer.
// main_corvid.cpp includes only this header; torch/torch.h is confined to
// RavenBrain.cpp.  Spec §2.4.2; M3 acceptance: ≤ 3 ms/tick at N=200 CPU.
#include "RavenNetConfig.hpp"
#include <cstdint>

namespace corvid {

struct RavenBrain {
    // Initialize trunk + adapter table.  Returns false if torch unavailable.
    bool init(const RavenNetConfig& cfg);

    // Batched forward pass (spec §2.4.2.a).
    // obs_flat    : N × d_obs floats (row-major)
    // adapter_idx : N int64s — slot index into adapter table
    // out_biases  : N × d_action floats (written)
    // out_values  : N floats          (written; critic estimate, unused until M4)
    void forward(const float*   obs_flat,
                 const int64_t* adapter_idx,
                 int            N,
                 float*         out_biases,
                 float*         out_values);

    float last_ms = 0.f;   // wall-time of last forward(), logged if > 3 ms

    RavenNetConfig cfg;

    ~RavenBrain();  // defined in RavenBrain.cpp (Impl must be complete there)

    struct Impl;
    Impl* impl_ = nullptr;  // raw ptr: unique_ptr triggers MSVC incomplete-type check at declaration
};

} // namespace corvid
