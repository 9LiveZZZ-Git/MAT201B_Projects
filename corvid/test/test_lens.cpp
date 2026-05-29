// Standalone test for the tuned-lens readout (no allolib needed).
//   g++ -std=c++17 -I.. test_lens.cpp ../cognition/Lens.cpp ../cognition/RavenBrain.cpp -o lens_smoke
//
// Validates that the lens, trained on (hidden -> policy softmax) pairs from the
// real RavenBrain, learns to reproduce the policy head's action distribution
// from hidden activations alone (CE decreases; argmax agreement rises).
#include "cognition/Lens.hpp"
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
    cfg.d_obs = 16; cfg.d_hidden = 24; cfg.d_action = LENS_ACT;
    cfg.lora_rank = 2; cfg.n_agents = 4;
    RavenBrain brain; brain.init(cfg);

    Lens lens; lens.init(cfg.d_hidden);

    std::mt19937 rng(777);
    std::normal_distribution<float> nd(0.f, 1.f);

    const int N = 64;
    std::vector<float>   obs(size_t(N) * cfg.d_obs);
    std::vector<int64_t> aidx(N);
    for (auto& o : obs) o = nd(rng);
    for (int n = 0; n < N; ++n) aidx[n] = n % cfg.n_agents;

    std::vector<float> logits(size_t(N) * cfg.d_action);
    std::vector<float> values(N);
    std::vector<float> hidden(size_t(N) * cfg.d_hidden);
    brain.forward_tap(obs.data(), aidx.data(), N, logits.data(), values.data(), hidden.data());

    // target = policy head softmax per sample
    std::vector<float> target(size_t(N) * cfg.d_action);
    for (int n = 0; n < N; ++n) {
        std::vector<float> p; softmax(&logits[n * cfg.d_action], cfg.d_action, p);
        for (int a = 0; a < cfg.d_action; ++a) target[n * cfg.d_action + a] = p[a];
    }

    float ce_first = lens.train_step(hidden.data(), target.data(), N);
    float ce_last = ce_first;
    for (int it = 0; it < 2000; ++it)
        ce_last = lens.train_step(hidden.data(), target.data(), N);

    bool finite = std::isfinite(ce_first) && std::isfinite(ce_last);
    bool learned = ce_last < ce_first - 1e-3f;
    std::printf("lens CE: %.4f -> %.4f  %s\n", ce_first, ce_last,
                (finite && learned) ? "OK" : "FAIL");
    if (!(finite && learned)) ++failures;

    // argmax agreement between lens decode and policy head
    int agree = 0;
    for (int n = 0; n < N; ++n) {
        float dist[LENS_ACT];
        lens.decode(&hidden[n * cfg.d_hidden], dist);
        int la = 0; for (int a = 1; a < cfg.d_action; ++a) if (dist[a] > dist[la]) la = a;
        int pa = 0; for (int a = 1; a < cfg.d_action; ++a)
            if (target[n*cfg.d_action+a] > target[n*cfg.d_action+pa]) pa = a;
        if (la == pa) ++agree;
    }
    float acc = float(agree) / float(N);
    std::printf("lens argmax agreement: %.0f%%  %s\n", acc * 100.f,
                acc >= 0.75f ? "OK" : "FAIL");
    if (acc < 0.75f) ++failures;

    // ThoughtVector well-formed
    ThoughtVector tv = lens.think(&hidden[0], values[0], int(GoalKind::SEEK_FOOD));
    float s = 0.f; for (float a : tv.action) s += a;
    bool tv_ok = std::fabs(s - 1.f) < 1e-3f &&
                 tv.entropy >= 0.f && tv.entropy <= 1.001f &&
                 tv.dominant >= 0 && tv.dominant < LENS_ACT;
    std::printf("thought vector: sum=%.3f entropy=%.3f dom=%d  %s\n",
                s, tv.entropy, tv.dominant, tv_ok ? "OK" : "FAIL");
    if (!tv_ok) ++failures;

    std::printf("\n%s (%d failure%s)\n",
                failures == 0 ? "ALL PASS" : "FAILURES", failures, failures == 1 ? "" : "s");
    return failures == 0 ? 0 : 1;
}
