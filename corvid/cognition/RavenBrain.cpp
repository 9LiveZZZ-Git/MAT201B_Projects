#include "RavenBrain.hpp"
#include <torch/torch.h>
#include <chrono>
#include <cstdio>

namespace corvid {

// ---------------------------------------------------------------------------
// RavenNet — torch::nn::Module (spec §2.4.2.b)
// Tiny tier: fc1(d_obs→d_hidden) + LoRA, relu, fc2(d_hidden→d_action),
// value_head(d_hidden→1).  Adapters: A_all[N, d_obs, r], B_all[N, r, d_hidden]
// initialized with A=N(0, σ), B=0 so lora delta starts at zero.
// ---------------------------------------------------------------------------
struct RavenNetImpl : torch::nn::Module {
    RavenNetImpl(const RavenNetConfig& cfg) : cfg_(cfg) {
        fc1_        = register_module("fc1",   torch::nn::Linear(cfg.d_obs,    cfg.d_hidden));
        fc2_        = register_module("fc2",   torch::nn::Linear(cfg.d_hidden, cfg.d_action));
        value_head_ = register_module("vhead", torch::nn::Linear(cfg.d_hidden, 1));

        // Adapter tables: A [N, d_obs, r], B [N, r, d_hidden]
        // B zero-init → LoRA output starts at zero (standard LoRA init).
        float a_std = 1.f / std::sqrt(float(cfg.d_obs));
        A_all_ = register_parameter("A_all",
            torch::randn({cfg.n_agents, cfg.d_obs, cfg.lora_rank}) * a_std,
            /*requires_grad=*/false);  // fixed for M3; unfrozen at M4
        B_all_ = register_parameter("B_all",
            torch::zeros({cfg.n_agents, cfg.lora_rank, cfg.d_hidden}),
            /*requires_grad=*/false);
    }

    struct Out { torch::Tensor action_logits, value; };

    // obs: [N, d_obs]  adapter_idx: [N] int64
    Out forward(const torch::Tensor& obs, const torch::Tensor& adapter_idx) {
        // Gather per-agent adapters (spec §2.4.2.a)
        auto A = torch::index_select(A_all_, 0, adapter_idx);  // [N, d_obs, r]
        auto B = torch::index_select(B_all_, 0, adapter_idx);  // [N, r, d_hidden]

        // Base fc1 output + LoRA delta
        auto base = fc1_->forward(obs);  // [N, d_hidden]

        // Batched LoRA: lora = bmm(bmm(x.unsqueeze(1), A), B).squeeze(1)
        // x.unsqueeze(1): [N, 1, d_obs]
        // @A:              [N, 1, r]
        // @B:              [N, 1, d_hidden]
        auto lora = torch::bmm(
                        torch::bmm(obs.unsqueeze(1), A),
                        B).squeeze(1);   // [N, d_hidden]

        auto h = torch::relu(base + lora);               // [N, d_hidden]
        return {fc2_->forward(h), value_head_->forward(h)};
    }

    RavenNetConfig        cfg_;
    torch::nn::Linear     fc1_{nullptr}, fc2_{nullptr}, value_head_{nullptr};
    torch::Tensor         A_all_, B_all_;
};
TORCH_MODULE(RavenNet);

// ---------------------------------------------------------------------------
// RavenBrain::Impl
// ---------------------------------------------------------------------------
struct RavenBrain::Impl {
    RavenNet net{nullptr};
    torch::Device device{torch::kCPU};
};

RavenBrain::~RavenBrain() { delete impl_; }

bool RavenBrain::init(const RavenNetConfig& c) {
    cfg   = c;
    delete impl_;
    impl_ = new Impl();
    impl_->net = RavenNet(c);
    impl_->net->eval();
    impl_->net->to(impl_->device);
    // Log to file (Windows GUI apps have no console)
    if (FILE* f = std::fopen("ravennet_init.log", "w")) {
        std::fprintf(f, "[RavenBrain] init  d_obs=%d  d_hidden=%d  d_action=%d"
                        "  lora_rank=%d  n_agents=%d  device=cpu\n",
                        c.d_obs, c.d_hidden, c.d_action, c.lora_rank, c.n_agents);
        std::fclose(f);
    }
    return true;
}

void RavenBrain::forward(const float*   obs_flat,
                         const int64_t* adapter_idx,
                         int            N,
                         float*         out_biases,
                         float*         out_values)
{
    if (!impl_ || N <= 0) return;

    auto t0 = std::chrono::high_resolution_clock::now();

    // Use InferenceMode per spec §3.2.2 (faster than NoGradGuard)
    torch::InferenceMode guard;

    auto obs_t = torch::from_blob(
        const_cast<float*>(obs_flat),
        {N, cfg.d_obs},
        torch::kFloat32).clone();   // clone: obs_flat may be stack memory

    auto idx_t = torch::from_blob(
        const_cast<int64_t*>(adapter_idx),
        {N},
        torch::kInt64).clone();

    auto out = impl_->net->forward(obs_t, idx_t);

    // Copy results back to caller's flat arrays
    auto logits_cpu = out.action_logits.contiguous().cpu();
    auto value_cpu  = out.value.contiguous().cpu();
    std::memcpy(out_biases, logits_cpu.data_ptr<float>(),
                N * cfg.d_action * sizeof(float));
    std::memcpy(out_values, value_cpu.data_ptr<float>(),
                N * sizeof(float));

    auto t1 = std::chrono::high_resolution_clock::now();
    last_ms  = std::chrono::duration<float, std::milli>(t1 - t0).count();

    if (last_ms > 3.f) {
        if (FILE* f = std::fopen("ravennet_timing.log", "a")) {
            std::fprintf(f, "WARNING forward %.2f ms > 3 ms budget (N=%d)\n", last_ms, N);
            std::fclose(f);
        }
    }
}

} // namespace corvid
