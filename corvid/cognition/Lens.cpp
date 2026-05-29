#include "Lens.hpp"
#include <algorithm>
#include <cmath>
#include <random>

namespace corvid {

static constexpr uint32_t SEED_LENS = 0x4C454E53u;  // 'LENS'
static constexpr float LENS_LR   = 1e-3f;
static constexpr float ADAM_B1   = 0.9f;
static constexpr float ADAM_B2   = 0.999f;
static constexpr float ADAM_EPS  = 1e-8f;

void TunedLens::init(int d_hidden_) {
    d_hidden = d_hidden_;
    std::mt19937 rng(SEED_LENS);
    std::normal_distribution<float> nd(0.f, 1.f / std::sqrt(float(d_hidden)));
    P.resize(size_t(LENS_ACT) * d_hidden);
    for (auto& w : P) w = nd(rng);
    bp.assign(LENS_ACT, 0.f);
    mP.assign(P.size(), 0.f); vP.assign(P.size(), 0.f);
    mb.assign(bp.size(), 0.f); vb.assign(bp.size(), 0.f);
    adam_t = 0;
}

static void softmax(const float* z, int n, float* out) {
    float mx = z[0];
    for (int i = 1; i < n; ++i) mx = std::max(mx, z[i]);
    float s = 0.f;
    for (int i = 0; i < n; ++i) { out[i] = std::exp(z[i] - mx); s += out[i]; }
    float inv = 1.f / (s + 1e-20f);
    for (int i = 0; i < n; ++i) out[i] *= inv;
}

void TunedLens::decode(const float* hidden, float* out_dist) const {
    float logit[LENS_ACT];
    for (int a = 0; a < LENS_ACT; ++a) {
        float acc = bp[a];
        const float* row = &P[size_t(a) * d_hidden];
        for (int h = 0; h < d_hidden; ++h) acc += row[h] * hidden[h];
        logit[a] = acc;
    }
    softmax(logit, LENS_ACT, out_dist);
}

float TunedLens::train_step(const float* hidden_flat, const float* target_flat, int N) {
    if (N <= 0) return 0.f;
    const float invN = 1.f / float(N);

    std::vector<float> gP(P.size(), 0.f);
    std::vector<float> gb(bp.size(), 0.f);
    double ce = 0.0;
    float pred[LENS_ACT];

    for (int n = 0; n < N; ++n) {
        const float* h = hidden_flat + size_t(n) * d_hidden;
        const float* t = target_flat + size_t(n) * LENS_ACT;
        decode(h, pred);

        // cross-entropy H(t, pred); grad wrt logit = (pred - t)
        for (int a = 0; a < LENS_ACT; ++a) {
            ce -= double(t[a]) * std::log(pred[a] + 1e-20f);
            float dlogit = (pred[a] - t[a]) * invN;
            gb[a] += dlogit;
            float* gr = &gP[size_t(a) * d_hidden];
            for (int j = 0; j < d_hidden; ++j) gr[j] += dlogit * h[j];
        }
    }

    // Adam update
    adam_t += 1;
    float bc1 = 1.f - std::pow(ADAM_B1, float(adam_t));
    float bc2 = 1.f - std::pow(ADAM_B2, float(adam_t));
    auto step = [&](std::vector<float>& w, std::vector<float>& m,
                    std::vector<float>& v, const std::vector<float>& g) {
        for (size_t i = 0; i < w.size(); ++i) {
            m[i] = ADAM_B1 * m[i] + (1.f - ADAM_B1) * g[i];
            v[i] = ADAM_B2 * v[i] + (1.f - ADAM_B2) * g[i] * g[i];
            w[i] -= LENS_LR * (m[i] / bc1) / (std::sqrt(v[i] / bc2) + ADAM_EPS);
        }
    };
    step(P, mP, vP, gP);
    step(bp, mb, vb, gb);

    return float(ce * invN);
}

ThoughtVector TunedLens::think(const float* hidden, float value, int dominant_goal) const {
    ThoughtVector tv;
    decode(hidden, tv.action);

    float H = 0.f; int arg = 0; float best = tv.action[0];
    for (int a = 0; a < LENS_ACT; ++a) {
        H -= tv.action[a] * std::log(tv.action[a] + 1e-20f);
        if (tv.action[a] > best) { best = tv.action[a]; arg = a; }
    }
    float Hmax = std::log(float(LENS_ACT));
    tv.entropy    = Hmax > 0.f ? H / Hmax : 0.f;
    tv.confidence = 1.f - tv.entropy;
    tv.value      = value;
    tv.dominant   = arg;
    tv.goal       = dominant_goal;
    return tv;
}

} // namespace corvid
