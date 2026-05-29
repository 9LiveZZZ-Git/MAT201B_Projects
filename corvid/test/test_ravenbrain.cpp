// Standalone numeric smoke test for the native (no-torch) RavenBrain.
// Validates: finite forward, no-NaN training, and that PPO moves the policy
// toward high-advantage actions (end-to-end gradient-direction check).
//
// Build (no allolib needed):
//   g++ -std=c++17 -I.. test_ravenbrain.cpp ../cognition/RavenBrain.cpp -o brain_smoke
#include "cognition/RavenBrain.hpp"
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

using namespace corvid;

static void softmax(const float* z, int n, std::vector<float>& p) {
    float mx = z[0];
    for (int i = 1; i < n; ++i) mx = std::max(mx, z[i]);
    float s = 0.f; p.resize(n);
    for (int i = 0; i < n; ++i) { p[i] = std::exp(z[i] - mx); s += p[i]; }
    for (int i = 0; i < n; ++i) p[i] /= s;
}

int main() {
    int failures = 0;

    RavenNetConfig cfg;
    cfg.d_obs = 16; cfg.d_hidden = 24; cfg.d_action = 4;
    cfg.lora_rank = 2; cfg.n_agents = 4;

    RavenBrain brain;
    brain.init(cfg);

    std::mt19937 rng(12345);
    std::normal_distribution<float> nd(0.f, 1.f);

    // Fixed observation batch, all on adapter 0, "good" action = 2.
    const int N = 16, GOOD = 2;
    std::vector<float>   obs(size_t(N) * cfg.d_obs);
    std::vector<int64_t> aidx(N, 0);
    for (auto& o : obs) o = nd(rng);

    // --- 1. forward finiteness ---
    std::vector<float> biases(size_t(N) * cfg.d_action), values(N);
    brain.forward(obs.data(), aidx.data(), N, biases.data(), values.data());
    for (float b : biases) if (!std::isfinite(b)) ++failures;
    for (float v : values) if (!std::isfinite(v)) ++failures;
    std::printf("forward finite: %s\n", failures == 0 ? "OK" : "FAIL");

    // prob of GOOD action before training (adapter 0, sample 0)
    std::vector<float> p0; softmax(biases.data(), cfg.d_action, p0);
    float before = p0[GOOD];

    // --- 2. PPO learning: reward GOOD action, ratio~1 each step (vanilla PG) ---
    for (int step = 0; step < 400; ++step) {
        brain.forward(obs.data(), aidx.data(), N, biases.data(), values.data());

        TrainBatch tb; tb.N = N;
        tb.obs = obs;
        for (int n = 0; n < N; ++n) {
            std::vector<float> p; softmax(biases.data() + n * cfg.d_action, cfg.d_action, p);
            tb.actions.push_back(GOOD);
            tb.advantages.push_back(1.0f);                 // GOOD was advantageous
            tb.returns.push_back(values[n]);               // neutral value target
            tb.old_logprobs.push_back(std::log(p[GOOD] + 1e-20f)); // ratio == 1
            tb.adapter_idx.push_back(0);
        }
        float kl = brain.train_step(tb);
        if (!std::isfinite(kl) || !std::isfinite(brain.last_policy_loss)) { ++failures; break; }
    }

    brain.forward(obs.data(), aidx.data(), N, biases.data(), values.data());
    for (float b : biases) if (!std::isfinite(b)) ++failures;
    std::vector<float> p1; softmax(biases.data(), cfg.d_action, p1);
    float after = p1[GOOD];

    bool learned = after > before + 1e-3f;
    std::printf("policy learning: p(good) %.4f -> %.4f  %s\n",
                before, after, learned ? "OK" : "FAIL");
    if (!learned) ++failures;

    // --- 3. inherit_adapter is finite ---
    brain.inherit_adapter(1, 0, -1, 0.01f);
    brain.inherit_adapter(2, 0,  1, 0.01f);
    aidx.assign(N, 2);
    brain.forward(obs.data(), aidx.data(), N, biases.data(), values.data());
    bool inh_ok = true;
    for (float b : biases) if (!std::isfinite(b)) inh_ok = false;
    std::printf("inherit_adapter finite: %s\n", inh_ok ? "OK" : "FAIL");
    if (!inh_ok) ++failures;

    std::printf("\n%s (%d failure%s)\n",
                failures == 0 ? "ALL PASS" : "FAILURES", failures, failures == 1 ? "" : "s");
    return failures == 0 ? 0 : 1;
}
