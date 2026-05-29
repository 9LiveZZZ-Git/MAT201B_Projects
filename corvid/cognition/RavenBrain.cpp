#include "RavenBrain.hpp"
// ---------------------------------------------------------------------------
// Native, dependency-free RavenNet (allolib-only rewrite — no libtorch).
//
// Architecture matches RavenNetConfig (spec §2.4.2.b):
//   trunk : fc1(d_obs -> d_hidden) + ReLU
//   heads : action(d_hidden -> d_action), value(d_hidden -> 1)
//   per-agent LoRA: A[N, d_obs, r], B[N, r, d_hidden]
//                   A ~ N(0, 1/sqrt(d_obs)) frozen, B = 0 (LoRA delta starts at 0)
//
// PPO train_step updates the shared trunk + heads + the per-agent B adapters
// (only A stays frozen — matching the original torch impl, which handed
// net->parameters() to Adam). Adam optimizer, clipped surrogate, value + entropy.
//
// All math is plain float / std::vector. Deterministic given the fixed seed.
// ---------------------------------------------------------------------------
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

namespace corvid {

static constexpr uint32_t SEED_RAVENNET = 0x52415645u;  // 'RAVE'

static constexpr float PPO_CLIP_EPS   = 0.2f;
static constexpr float VALUE_COEF     = 0.5f;
static constexpr float ENTROPY_COEF   = 0.01f;
static constexpr float GRAD_CLIP_NORM = 0.5f;
static constexpr float ADAM_LR        = 3e-4f;
static constexpr float ADAM_B1        = 0.9f;
static constexpr float ADAM_B2        = 0.999f;
static constexpr float ADAM_EPS       = 1e-5f;

// ---------------------------------------------------------------------------
struct RavenBrain::Impl {
    RavenNetConfig c;

    // Trainable params.
    std::vector<float> W1, b1, Wa, ba, Wv;   float bv = 0.f;
    std::vector<float> A;                      // frozen LoRA A
    std::vector<float> B;                      // trainable LoRA B

    // Adam moments (parallel to each trainable param).
    std::vector<float> mW1, vW1, mb1, vb1, mWa, vWa, mba, vba, mWv, vWv, mB, vB;
    float mbv = 0.f, vbv = 0.f;
    long  adam_t = 0;

    std::mt19937 rng{SEED_RAVENNET};

    inline int W1i(int hid, int j)  const { return hid * c.d_obs + j; }
    inline int Wai(int m,   int hid) const { return m * c.d_hidden + hid; }
    inline int Ai (int ag, int j, int k)   const { return (ag * c.d_obs + j) * c.lora_rank + k; }
    inline int Bi (int ag, int k, int hid) const { return (ag * c.lora_rank + k) * c.d_hidden + hid; }
    inline int Bblock(int ag) const { return ag * c.lora_rank * c.d_hidden; }

    void init(const RavenNetConfig& cfg) {
        c = cfg;
        const int dO = c.d_obs, dH = c.d_hidden, dA = c.d_action, r = c.lora_rank, N = c.n_agents;

        std::normal_distribution<float> n1(0.f, 1.f / std::sqrt(float(dO)));
        std::normal_distribution<float> nH(0.f, 1.f / std::sqrt(float(dH)));

        W1.resize(dH * dO); for (auto& w : W1) w = n1(rng);
        b1.assign(dH, 0.f);
        Wa.resize(dA * dH); for (auto& w : Wa) w = nH(rng);
        ba.assign(dA, 0.f);
        Wv.resize(dH);      for (auto& w : Wv) w = nH(rng);
        bv = 0.f;

        std::normal_distribution<float> nA(0.f, 1.f / std::sqrt(float(dO)));
        A.resize(size_t(N) * dO * r); for (auto& w : A) w = nA(rng);
        B.assign(size_t(N) * r * dH, 0.f);

        mW1.assign(W1.size(), 0.f); vW1.assign(W1.size(), 0.f);
        mb1.assign(b1.size(), 0.f); vb1.assign(b1.size(), 0.f);
        mWa.assign(Wa.size(), 0.f); vWa.assign(Wa.size(), 0.f);
        mba.assign(ba.size(), 0.f); vba.assign(ba.size(), 0.f);
        mWv.assign(Wv.size(), 0.f); vWv.assign(Wv.size(), 0.f);
        mB.assign(B.size(), 0.f);   vB.assign(B.size(), 0.f);
        mbv = vbv = 0.f;
        adam_t = 0;
    }

    void forwardOne(const float* x, int ag,
                    float* out_logits, float* out_value,
                    std::vector<float>* pre = nullptr,
                    std::vector<float>* hvec = nullptr,
                    std::vector<float>* uvec = nullptr) const {
        const int dO = c.d_obs, dH = c.d_hidden, dA = c.d_action, r = c.lora_rank;

        float u[8] = {0};
        for (int k = 0; k < r; ++k) {
            float acc = 0.f;
            for (int j = 0; j < dO; ++j) acc += x[j] * A[Ai(ag, j, k)];
            u[k] = acc;
        }

        std::vector<float> h(dH);
        if (pre) pre->assign(dH, 0.f);
        for (int hid = 0; hid < dH; ++hid) {
            float acc = b1[hid];
            for (int j = 0; j < dO; ++j) acc += W1[W1i(hid, j)] * x[j];
            for (int k = 0; k < r; ++k)  acc += u[k] * B[Bi(ag, k, hid)];
            if (pre) (*pre)[hid] = acc;
            h[hid] = acc > 0.f ? acc : 0.f;
        }

        for (int m = 0; m < dA; ++m) {
            float acc = ba[m];
            for (int hid = 0; hid < dH; ++hid) acc += Wa[Wai(m, hid)] * h[hid];
            out_logits[m] = acc;
        }
        float v = bv;
        for (int hid = 0; hid < dH; ++hid) v += Wv[hid] * h[hid];
        *out_value = v;

        if (hvec) *hvec = std::move(h);
        if (uvec) { uvec->assign(r, 0.f); for (int k = 0; k < r; ++k) (*uvec)[k] = u[k]; }
    }

    // Adam update of one param vector given its grad (already globally scaled).
    void adam(std::vector<float>& p, std::vector<float>& m, std::vector<float>& v,
              const std::vector<float>& g, float bc1, float bc2) {
        for (size_t i = 0; i < p.size(); ++i) {
            m[i] = ADAM_B1 * m[i] + (1.f - ADAM_B1) * g[i];
            v[i] = ADAM_B2 * v[i] + (1.f - ADAM_B2) * g[i] * g[i];
            p[i] -= ADAM_LR * (m[i] / bc1) / (std::sqrt(v[i] / bc2) + ADAM_EPS);
        }
    }
};

// ---------------------------------------------------------------------------
RavenBrain::~RavenBrain() { delete impl_; }

bool RavenBrain::init(const RavenNetConfig& c) {
    cfg = c;
    delete impl_;
    impl_ = new Impl();
    impl_->init(c);
    if (FILE* f = std::fopen("ravennet_init.log", "w")) {
        std::fprintf(f, "[RavenBrain] native init  d_obs=%d d_hidden=%d d_action=%d"
                        " lora_rank=%d n_agents=%d  (no torch)\n",
                        c.d_obs, c.d_hidden, c.d_action, c.lora_rank, c.n_agents);
        std::fclose(f);
    }
    return true;
}

void RavenBrain::forward(const float* obs_flat, const int64_t* adapter_idx,
                         int N, float* out_biases, float* out_values) {
    if (!impl_ || N <= 0) return;
    auto t0 = std::chrono::high_resolution_clock::now();
    const int dO = cfg.d_obs, dA = cfg.d_action;
    for (int n = 0; n < N; ++n) {
        int ag = int(adapter_idx[n]);
        if (ag < 0 || ag >= cfg.n_agents) ag = 0;
        impl_->forwardOne(obs_flat + size_t(n) * dO, ag,
                          out_biases + size_t(n) * dA, out_values + n);
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    last_ms = std::chrono::duration<float, std::milli>(t1 - t0).count();
}

void RavenBrain::forward_tap(const float* obs_flat, const int64_t* adapter_idx,
                             int N, float* out_biases, float* out_values,
                             float* out_hidden) {
    if (!impl_ || N <= 0) return;
    const int dO = cfg.d_obs, dA = cfg.d_action, dH = cfg.d_hidden;
    std::vector<float> logits(dA), hvec;
    float val = 0.f;
    for (int n = 0; n < N; ++n) {
        int ag = int(adapter_idx[n]);
        if (ag < 0 || ag >= cfg.n_agents) ag = 0;
        impl_->forwardOne(obs_flat + size_t(n) * dO, ag,
                          logits.data(), &val, nullptr, &hvec, nullptr);
        if (out_biases) std::copy(logits.begin(), logits.end(),
                                  out_biases + size_t(n) * dA);
        if (out_values) out_values[n] = val;
        if (out_hidden) std::copy(hvec.begin(), hvec.end(),
                                  out_hidden + size_t(n) * dH);
    }
}

// ---------------------------------------------------------------------------
// PPO gradient step over trunk + heads + per-agent B (A frozen).
// ---------------------------------------------------------------------------
float RavenBrain::train_step(const TrainBatch& batch) {
    if (!impl_ || batch.N < 4) return 0.f;
    auto t0 = std::chrono::high_resolution_clock::now();

    Impl& I = *impl_;
    const int dO = cfg.d_obs, dH = cfg.d_hidden, dA = cfg.d_action, r = cfg.lora_rank;
    const int N  = batch.N;
    const float invN = 1.f / float(N);

    std::vector<float> gW1(I.W1.size(), 0.f), gb1(I.b1.size(), 0.f);
    std::vector<float> gWa(I.Wa.size(), 0.f), gba(I.ba.size(), 0.f);
    std::vector<float> gWv(I.Wv.size(), 0.f); float gbv = 0.f;
    std::vector<float> gB(I.B.size(), 0.f);

    double sum_kl = 0.0, sum_ploss = 0.0, sum_vloss = 0.0;
    std::vector<float> pre, h, u, logits(dA), prob(dA), dlogit(dA);

    for (int n = 0; n < N; ++n) {
        int ag = int(batch.adapter_idx[n]);
        if (ag < 0 || ag >= cfg.n_agents) ag = 0;
        const float* x = batch.obs.data() + size_t(n) * dO;

        float value = 0.f;
        I.forwardOne(x, ag, logits.data(), &value, &pre, &h, &u);

        float mx = logits[0];
        for (int m = 1; m < dA; ++m) mx = std::max(mx, logits[m]);
        float sum = 0.f;
        for (int m = 0; m < dA; ++m) { prob[m] = std::exp(logits[m] - mx); sum += prob[m]; }
        float inv = 1.f / (sum + 1e-20f);
        for (int m = 0; m < dA; ++m) prob[m] *= inv;

        int a = batch.actions[n]; if (a < 0 || a >= dA) a = 0;
        float new_lp = std::log(prob[a] + 1e-20f);
        float old_lp = batch.old_logprobs[n];

        float adv = batch.advantages[n];
        if (!batch.teaching_scale.empty()) adv *= batch.teaching_scale[n];
        float ret = batch.returns[n];

        float ratio = std::exp(new_lp - old_lp);
        float surr1 = ratio * adv;
        float clipped = std::min(std::max(ratio, 1.f - PPO_CLIP_EPS), 1.f + PPO_CLIP_EPS);
        float surr2 = clipped * adv;
        bool  use1  = surr1 <= surr2;
        float coef_p = use1 ? (adv * ratio) : 0.f;       // d(min surr)/d(new_lp)

        float H = 0.f;
        for (int m = 0; m < dA; ++m) H -= prob[m] * std::log(prob[m] + 1e-20f);

        float dv = 2.f * VALUE_COEF * (value - ret) * invN;   // d(VALUE_COEF*MSE)/dvalue

        for (int m = 0; m < dA; ++m) {
            float dpi = ((m == a) ? 1.f : 0.f) - prob[m];      // d new_lp / d logit
            float g_policy  = -(coef_p * invN) * dpi;
            float g_entropy = ENTROPY_COEF * prob[m] * (std::log(prob[m] + 1e-20f) + H);
            dlogit[m] = g_policy + g_entropy;
        }

        // head grads + backprop to h
        std::vector<float> dh(dH, 0.f);
        for (int m = 0; m < dA; ++m) {
            gba[m] += dlogit[m];
            for (int hid = 0; hid < dH; ++hid) {
                gWa[I.Wai(m, hid)] += dlogit[m] * h[hid];
                dh[hid] += dlogit[m] * I.Wa[I.Wai(m, hid)];
            }
        }
        gbv += dv;
        for (int hid = 0; hid < dH; ++hid) {
            gWv[hid] += dv * h[hid];
            dh[hid]  += dv * I.Wv[hid];
        }

        // through ReLU into pre, then trunk + LoRA B
        int bbase = I.Bblock(ag);
        for (int hid = 0; hid < dH; ++hid) {
            float dpre = (pre[hid] > 0.f) ? dh[hid] : 0.f;
            if (dpre == 0.f) continue;
            gb1[hid] += dpre;
            for (int j = 0; j < dO; ++j) gW1[I.W1i(hid, j)] += dpre * x[j];
            for (int k = 0; k < r; ++k)  gB[bbase + k * dH + hid] += dpre * u[k];
        }

        sum_kl    += double(old_lp - new_lp);
        sum_ploss += double(-std::min(surr1, surr2));
        sum_vloss += double((value - ret) * (value - ret));
    }

    // Global grad-norm clip across all trained params.
    double nrm2 = 0.0;
    auto acc = [&](const std::vector<float>& g) { for (float v : g) nrm2 += double(v) * v; };
    acc(gW1); acc(gb1); acc(gWa); acc(gba); acc(gWv); acc(gB);
    nrm2 += double(gbv) * gbv;
    float nrm = float(std::sqrt(nrm2));
    if (nrm > GRAD_CLIP_NORM && nrm > 0.f) {
        float s = GRAD_CLIP_NORM / nrm;
        auto sc = [&](std::vector<float>& g) { for (float& v : g) v *= s; };
        sc(gW1); sc(gb1); sc(gWa); sc(gba); sc(gWv); sc(gB); gbv *= s;
    }

    I.adam_t += 1;
    float bc1 = 1.f - std::pow(ADAM_B1, float(I.adam_t));
    float bc2 = 1.f - std::pow(ADAM_B2, float(I.adam_t));
    I.adam(I.W1, I.mW1, I.vW1, gW1, bc1, bc2);
    I.adam(I.b1, I.mb1, I.vb1, gb1, bc1, bc2);
    I.adam(I.Wa, I.mWa, I.vWa, gWa, bc1, bc2);
    I.adam(I.ba, I.mba, I.vba, gba, bc1, bc2);
    I.adam(I.Wv, I.mWv, I.vWv, gWv, bc1, bc2);
    I.adam(I.B,  I.mB,  I.vB,  gB,  bc1, bc2);
    {   // bv scalar
        I.mbv = ADAM_B1 * I.mbv + (1.f - ADAM_B1) * gbv;
        I.vbv = ADAM_B2 * I.vbv + (1.f - ADAM_B2) * gbv * gbv;
        I.bv -= ADAM_LR * (I.mbv / bc1) / (std::sqrt(I.vbv / bc2) + ADAM_EPS);
    }

    last_kl          = float(sum_kl    * invN);
    last_policy_loss = float(sum_ploss * invN);
    last_value_loss  = float(sum_vloss * invN);

    auto t1 = std::chrono::high_resolution_clock::now();
    last_ms = std::chrono::duration<float, std::milli>(t1 - t0).count();
    return last_kl;
}

// ---------------------------------------------------------------------------
void RavenBrain::inherit_adapter(int child, int pa, int pb, float sigma_base) {
    if (!impl_) return;
    Impl& I = *impl_;
    const int dH = cfg.d_hidden, r = cfg.lora_rank;
    if (child < 0 || child >= cfg.n_agents || pa < 0 || pa >= cfg.n_agents) return;

    int cbase = I.Bblock(child), abase = I.Bblock(pa);
    for (int e = 0; e < r * dH; ++e) I.B[cbase + e] = I.B[abase + e];

    if (pb >= 0 && pb != pa && pb < cfg.n_agents) {
        int bbase = I.Bblock(pb);
        std::uniform_real_distribution<float> uni(0.f, 1.f);
        for (int k = 0; k < r; ++k)
            if (uni(I.rng) < 0.5f)
                for (int hid = 0; hid < dH; ++hid)
                    I.B[cbase + k * dH + hid] = I.B[bbase + k * dH + hid];
    }

    std::normal_distribution<float> gauss(0.f, 1.f);
    for (int e = 0; e < r * dH; ++e) {
        float b = I.B[cbase + e];
        I.B[cbase + e] = b + sigma_base * std::fabs(b) * gauss(I.rng);
        I.mB[cbase + e] = 0.f;
        I.vB[cbase + e] = 0.f;
    }
}

} // namespace corvid
