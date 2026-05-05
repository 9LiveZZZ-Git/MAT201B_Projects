// pulsar_cern.cpp — Pulsar synthesis driven by CMS Open Data dimuon collisions.
//
// Each CERN event triggers one pulsar voice. Pulsar synthesis (Curtis Roads):
//   pulsar period Pp = 1/Ff      (fundamental)
//   pulsaret duration W = N/Fc   (N cycles of formant Fc)
//   duty D = W/Pp = N * Ff / Fc  (D<1 → silence between pulsarets)
//
// Mapping:
//   invariant mass M (GeV)  → Ff  (log: 30..1500 Hz)
//   leading muon pT         → amplitude, voice life
//   muon-pair Σ|η|          → formant ratio Fc/Ff
//   muon η                  → stereo pan
//   product of charges      → waveform shape (same-sign vs opposite-sign)
//
// Build:
//   ./run.sh MAT201B_projects/pulsar_cern/pulsar_cern.cpp
//
// Real data:
//   bash MAT201B_projects/pulsar_cern/data/download_dimuon.sh
//   then either rename dimuon_full.csv → dimuon_sample.csv,
//   or set CSV path via the GUI.

#include "al/app/al_App.hpp"
#include "al/graphics/al_Shapes.hpp"
#include "al/graphics/al_VAOMesh.hpp"
#include "al/io/al_ControlNav.hpp"
#include "al/math/al_Random.hpp"
#include "al/ui/al_ControlGUI.hpp"
#include "al/ui/al_Parameter.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

using namespace al;

// -------------------------------------------------------------
// Dimuon event from CMS Open Data CSV.
// -------------------------------------------------------------
struct DimuonEvent {
  // Two muons.
  float pt1, eta1, phi1; int q1;
  float pt2, eta2, phi2; int q2;
  // 4-momentum components useful for visuals.
  float px1, py1, pz1;
  float px2, py2, pz2;
  // Invariant mass (GeV).
  float M;
};

static bool loadDimuonCSV(const std::string& path, std::vector<DimuonEvent>& out) {
  std::ifstream f(path);
  if (!f.is_open()) return false;
  std::string line;
  std::getline(f, line);  // header
  out.clear();
  while (std::getline(f, line)) {
    if (line.empty()) continue;
    std::stringstream ss(line);
    std::string cell;
    std::vector<std::string> cols;
    cols.reserve(21);
    while (std::getline(ss, cell, ',')) cols.push_back(cell);
    if (cols.size() < 21) continue;
    DimuonEvent e{};
    try {
      // Run, Event, Type1 → cols 0..2 ignored
      // E1, px1, py1, pz1, pt1, eta1, phi1, Q1
      e.px1 = std::stof(cols[4]);
      e.py1 = std::stof(cols[5]);
      e.pz1 = std::stof(cols[6]);
      e.pt1 = std::stof(cols[7]);
      e.eta1 = std::stof(cols[8]);
      e.phi1 = std::stof(cols[9]);
      e.q1 = std::stoi(cols[10]);
      // Type2, E2, px2, py2, pz2, pt2, eta2, phi2, Q2
      e.px2 = std::stof(cols[13]);
      e.py2 = std::stof(cols[14]);
      e.pz2 = std::stof(cols[15]);
      e.pt2 = std::stof(cols[16]);
      e.eta2 = std::stof(cols[17]);
      e.phi2 = std::stof(cols[18]);
      e.q2 = std::stoi(cols[19]);
      e.M = std::stof(cols[20]);
    } catch (...) { continue; }
    out.push_back(e);
  }
  return !out.empty();
}

// Synthetic fallback if CSV loading fails — keeps the app demoable offline.
static void synthDataset(std::vector<DimuonEvent>& out, int count = 80) {
  out.clear();
  rnd::Random<> rng;
  // Mix resonances + continuum.
  auto sampleMass = [&](int i) -> float {
    int b = i % 5;
    if (b == 0) return 3.10f + rng.uniformS() * 0.05f;     // J/psi
    if (b == 1) return 9.46f + rng.uniformS() * 0.4f;      // Upsilon
    if (b == 2) return 91.2f + rng.uniformS() * 1.5f;      // Z
    if (b == 3) return 0.5f + rng.uniform() * 8.0f;        // low continuum
    return 10.0f + rng.uniform() * 70.0f;                  // continuum
  };
  for (int i = 0; i < count; ++i) {
    DimuonEvent e{};
    e.M = sampleMass(i);
    e.pt1 = 5.0f + rng.uniform() * 40.0f;
    e.pt2 = 5.0f + rng.uniform() * 40.0f;
    e.eta1 = rng.uniformS() * 2.4f;
    e.eta2 = -e.eta1 + rng.uniformS() * 0.4f;
    e.phi1 = rng.uniformS() * float(M_PI);
    e.phi2 = e.phi1 + float(M_PI) + rng.uniformS() * 0.3f;
    e.q1 = (i & 1) ? 1 : -1; e.q2 = -e.q1;
    e.px1 = e.pt1 * std::cos(e.phi1); e.py1 = e.pt1 * std::sin(e.phi1);
    e.pz1 = e.pt1 * std::sinh(e.eta1);
    e.px2 = e.pt2 * std::cos(e.phi2); e.py2 = e.pt2 * std::sin(e.phi2);
    e.pz2 = e.pt2 * std::sinh(e.eta2);
    out.push_back(e);
  }
}

// -------------------------------------------------------------
// Pulsar voice. One per active CERN event.
//
// State written by sim thread BEFORE active.store(true, release).
// Audio thread does active.load(acquire) → safe to read params.
// Audio thread sets active=false when lifetime is exhausted.
// -------------------------------------------------------------
struct PulsarVoice {
  std::atomic<bool> active{false};

  // Synth params (audio-thread reads after acquire).
  float Ff = 100.f;       // fundamental (Hz)
  float Fc = 800.f;       // formant (Hz)
  float N = 3.f;          // cycles per pulsaret
  float amp = 0.4f;
  float pan = 0.f;        // -1..1
  float lifetime = 1.5f;
  int waveformShape = 0;  // 0 sine, 1 saw, 2 tri
  int windowShape = 0;    // 0 gauss, 1 hann, 2 tri
  float jitterAmt = 0.f;  // 0..1
  float maskProb = 0.f;   // 0..1
  uint32_t rngState = 1u;

  // Visual metadata (read by GL thread).
  float massGeV = 0.f;
  Vec3f muon1Dir{0,0,0}, muon2Dir{0,0,0};
  float pt1 = 0.f, pt2 = 0.f;
  int q1 = 0, q2 = 0;

  // Audio-thread-only state.
  float phase01 = 0.f;
  float age = 0.f;
  int   lastCycleIdx = -1;
  bool  cycleMasked = false;
  float cyclePhaseJitter = 0.f;

  // Cheap xorshift to avoid contending the global RNG from audio thread.
  inline float rand01() {
    rngState ^= rngState << 13; rngState ^= rngState >> 17; rngState ^= rngState << 5;
    return (rngState & 0x00FFFFFFu) / float(0x01000000);
  }

  // Returns sample for current frame, advances phase.
  // Returns 0 (and frees voice) once age > lifetime.
  float tick(float dt) {
    if (age >= lifetime) { active.store(false, std::memory_order_release); return 0.f; }
    age += dt;

    // Per-cycle decisions (mask, jitter).
    int cycleIdx = int(phase01);
    float frac = phase01 - float(cycleIdx);
    if (cycleIdx != lastCycleIdx) {
      cycleMasked = rand01() < maskProb;
      cyclePhaseJitter = (rand01() - 0.5f) * jitterAmt;
      lastCycleIdx = cycleIdx;
    }

    // Duty: pulsaret length / period.  D = N*Ff/Fc.
    float D = (Fc > 0.f) ? (N * Ff / Fc) : 1.f;
    if (D > 1.f) D = 1.f;  // formant ≥ N*Ff: continuous, no silence.

    float s = 0.f;
    if (!cycleMasked && frac < D) {
      // Position inside pulsaret: 0..1.
      float u = frac / D;
      float u_jit = u + cyclePhaseJitter;
      // Window envelope.
      float w;
      switch (windowShape) {
        case 1: w = 0.5f * (1.f - std::cos(2.f * float(M_PI) * u)); break;          // Hann
        case 2: w = 1.f - std::fabs(2.f * u - 1.f); break;                          // Triangle
        default: { float c = 2.f * u - 1.f; w = std::exp(-4.f * c * c); }           // Gaussian
      }
      // Waveform: phase advances at Fc. Within pulsaret it cycles N times.
      float wfPhase = u_jit * N;
      float wf;
      switch (waveformShape) {
        case 1: { float p = wfPhase - std::floor(wfPhase); wf = 2.f * p - 1.f; break; }    // Saw
        case 2: { float p = wfPhase - std::floor(wfPhase); wf = 1.f - 4.f * std::fabs(p - 0.5f); break; } // Tri
        default: wf = std::sin(2.f * float(M_PI) * wfPhase);                                              // Sine
      }
      s = w * wf * amp;
      // Lifetime fade in/out so voices don't click on birth/death.
      const float fadeIn = 0.05f, fadeOut = 0.4f;
      float env = 1.f;
      if (age < fadeIn) env *= age / fadeIn;
      if (age > lifetime - fadeOut) env *= std::max(0.f, (lifetime - age) / fadeOut);
      s *= env;
    }

    phase01 += dt * Ff;
    return s;
  }
};

// -------------------------------------------------------------
// App
// -------------------------------------------------------------
struct PulsarCernApp : App {
  static constexpr int    kVoices = 32;
  static constexpr float  kSampleRate = 48000.f;

  std::vector<DimuonEvent> events;
  PulsarVoice voices[kVoices];

  // Triggering / playback.
  size_t nextEventIdx = 0;
  double timeAccum = 0.0;

  // Visual state for each voice (mirrored on sim thread).
  struct VoiceVisual {
    bool active = false;
    float age = 0.f;
    float lifetime = 1.f;
    float massGeV = 0.f;
    Vec3f m1{0,0,0}, m2{0,0,0};
    int q1 = 0, q2 = 0;
  };
  VoiceVisual vvis[kVoices];

  // GUI parameters — atomic via Parameter, safe to read from audio thread.
  Parameter eventsPerSecond{"events/sec", "transport", 2.0f, 0.05f, 20.f};
  Parameter freqMin{"Ff min Hz", "synth", 40.f, 20.f, 200.f};
  Parameter freqMax{"Ff max Hz", "synth", 1200.f, 300.f, 3000.f};
  Parameter formantRatioMin{"Fc/Ff min", "synth", 1.0f, 0.5f, 8.f};
  Parameter formantRatioMax{"Fc/Ff max", "synth", 6.0f, 1.f, 32.f};
  Parameter pulsaretCycles{"N cycles", "synth", 3.f, 1.f, 12.f};
  Parameter masterGain{"gain", "synth", 0.6f, 0.f, 1.5f};
  Parameter jitterAmount{"jitter", "synth", 0.05f, 0.f, 0.5f};
  Parameter maskProb{"mask prob", "synth", 0.10f, 0.f, 0.9f};
  ParameterInt windowShape{"window", "synth", 0, 0, 2};
  Parameter voiceLife{"voice life s", "synth", 1.5f, 0.2f, 6.f};
  ParameterBool freezePlayback{"freeze", "transport", false};

  ControlGUI gui;

  // Geometry.
  VAOMesh vertexMarker, muonCone;

  // CSV path attempts (relative to bin/ which is the cwd allolib launches from).
  std::vector<std::string> csvCandidates() {
    return {
      "../data/dimuon_sample.csv",
      "../data/dimuon_full.csv",
      "../../data/dimuon_sample.csv",
      "MAT201B_projects/pulsar_cern/data/dimuon_sample.csv",
    };
  }

  void onInit() override {
    // Load dataset.
    bool loaded = false;
    for (auto& p : csvCandidates()) {
      if (loadDimuonCSV(p, events)) {
        std::cout << "[pulsar_cern] loaded " << events.size()
                  << " events from " << p << "\n";
        loaded = true; break;
      }
    }
    if (!loaded) {
      synthDataset(events);
      std::cout << "[pulsar_cern] CSV not found, using " << events.size()
                << " synthetic events.\n";
    }
  }

  void onCreate() override {
    nav().pos(0, 1.5, 8);
    nav().faceToward(Vec3d{0, 0, 0});

    // Collision vertex.
    addSphere(vertexMarker, 0.18f, 16, 16);
    vertexMarker.update();

    // Muon direction cone (drawn scaled per voice). Apex at +z=1.
    addCone(muonCone, 0.12f, Vec3f{0, 0, 1}, 12, 1);
    muonCone.update();

    // GUI.
    gui << eventsPerSecond << freezePlayback << masterGain
        << freqMin << freqMax << formantRatioMin << formantRatioMax
        << pulsaretCycles << windowShape << jitterAmount << maskProb
        << voiceLife;
    gui.init();
  }

  // -------------------------------------------------------
  // Event triggering: find an inactive voice, fill in params from a CERN
  // event, then publish via active.store(release).
  // -------------------------------------------------------
  void triggerEvent(const DimuonEvent& e) {
    int slot = -1;
    for (int i = 0; i < kVoices; ++i) {
      if (!voices[i].active.load(std::memory_order_acquire)) { slot = i; break; }
    }
    if (slot < 0) return;  // pool full, drop event.
    PulsarVoice& v = voices[slot];

    // Logarithmic mass→Ff mapping. M domain ~[0.5, 100] GeV.
    float Mc = std::max(0.5f, std::min(120.f, e.M));
    float t  = std::log(Mc / 0.5f) / std::log(120.f / 0.5f);  // 0..1
    v.Ff = freqMin.get() * std::pow(freqMax.get() / freqMin.get(), t);

    // Formant: ratio scaled by Σ|η| (more forward muons → brighter).
    float etaSum = std::fabs(e.eta1) + std::fabs(e.eta2);  // ~0..5
    float rt = std::min(1.f, etaSum / 5.f);
    float ratio = formantRatioMin.get()
                + (formantRatioMax.get() - formantRatioMin.get()) * rt;
    v.Fc = v.Ff * ratio;

    v.N = pulsaretCycles.get();

    // Amplitude from leading pT (5..50 GeV → 0.2..1.0).
    float ptLead = std::max(e.pt1, e.pt2);
    float ptN = std::min(1.f, std::max(0.f, (ptLead - 5.f) / 45.f));
    v.amp = 0.2f + 0.8f * ptN;

    // Lifetime scales with pT.
    v.lifetime = voiceLife.get() * (0.5f + ptN);

    // Pan from average η (forward muons spread wider).
    float etaAvg = 0.5f * (e.eta1 + e.eta2);
    v.pan = std::max(-1.f, std::min(1.f, etaAvg / 2.4f));

    // Same-sign muons (rare in this data) → saw, opposite-sign → sine, both forward → tri.
    if (e.q1 == e.q2) v.waveformShape = 1;
    else if (etaSum > 4.0f) v.waveformShape = 2;
    else v.waveformShape = 0;

    v.windowShape = windowShape.get();
    v.jitterAmt = jitterAmount.get();
    v.maskProb = maskProb.get();
    v.rngState = uint32_t(slot * 2654435761u + uint32_t(e.M * 1000.f) + 1u);

    // Visual data.
    v.massGeV = e.M;
    v.muon1Dir = Vec3f{e.px1, e.py1, e.pz1};
    if (v.muon1Dir.mag() > 0) v.muon1Dir.normalize();
    v.muon2Dir = Vec3f{e.px2, e.py2, e.pz2};
    if (v.muon2Dir.mag() > 0) v.muon2Dir.normalize();
    v.pt1 = e.pt1; v.pt2 = e.pt2;
    v.q1 = e.q1; v.q2 = e.q2;

    // Reset audio-thread-only state.
    v.phase01 = 0.f;
    v.age = 0.f;
    v.lastCycleIdx = -1;
    v.cycleMasked = false;
    v.cyclePhaseJitter = 0.f;

    // Mirror to visual record (sim-thread side).
    VoiceVisual& vv = vvis[slot];
    vv.active = true;
    vv.age = 0.f;
    vv.lifetime = v.lifetime;
    vv.massGeV = v.massGeV;
    vv.m1 = v.muon1Dir; vv.m2 = v.muon2Dir;
    vv.q1 = v.q1; vv.q2 = v.q2;

    // Publish — must be last, with release ordering.
    v.active.store(true, std::memory_order_release);
  }

  void onAnimate(double dt) override {
    if (!freezePlayback.get() && !events.empty()) {
      timeAccum += dt;
      double interval = 1.0 / std::max(0.05f, eventsPerSecond.get());
      while (timeAccum >= interval) {
        triggerEvent(events[nextEventIdx]);
        nextEventIdx = (nextEventIdx + 1) % events.size();
        timeAccum -= interval;
      }
    }
    // Update visual fade — read active flag (acquire) and copy age.
    for (int i = 0; i < kVoices; ++i) {
      VoiceVisual& vv = vvis[i];
      if (!vv.active) continue;
      vv.age += float(dt);
      if (!voices[i].active.load(std::memory_order_acquire) || vv.age >= vv.lifetime) {
        vv.active = false;
      }
    }
  }

  void onDraw(Graphics& g) override {
    g.clear(0.02f, 0.02f, 0.04f);
    g.lighting(false);
    g.depthTesting(true);
    g.blending(true);
    g.blendTrans();

    // Collision vertex (always at origin).
    g.color(0.8f, 0.8f, 0.9f);
    g.draw(vertexMarker);

    // Beam pipe drawn as a line along z.
    {
      Mesh m;
      m.primitive(Mesh::LINES);
      m.vertex(0, 0, -6); m.color(0.2f, 0.3f, 0.5f);
      m.vertex(0, 0,  6); m.color(0.2f, 0.3f, 0.5f);
      g.draw(m);
    }

    // Per-voice visuals.
    for (int i = 0; i < kVoices; ++i) {
      VoiceVisual& vv = vvis[i];
      if (!vv.active) continue;
      float life = vv.age / std::max(0.001f, vv.lifetime);
      float fade = 1.f - std::min(1.f, life);

      auto drawMuon = [&](const Vec3f& dir, float pt, int q) {
        // Color by charge.
        float r = (q > 0) ? 0.95f : 0.30f;
        float bb = (q > 0) ? 0.30f : 0.95f;
        float gg = 0.50f;
        // Length proportional to pT, shrinking with life.
        float len = std::min(5.f, 0.5f + 0.08f * pt) * (0.4f + 0.6f * fade);
        // Trail of segments along direction with falloff.
        Mesh trail;
        trail.primitive(Mesh::LINE_STRIP);
        const int segs = 16;
        for (int s = 0; s <= segs; ++s) {
          float t = float(s) / segs;
          Vec3f p = dir * (len * t);
          float a = (1.f - t) * fade;
          trail.vertex(p);
          trail.color(r * a, gg * a, bb * a, a);
        }
        g.draw(trail);

        // Tip marker (cone scaled).
        g.pushMatrix();
        Vec3f tip = dir * len;
        g.translate(tip);
        // Orient cone along dir — addCone points along +z.
        // Build rotation that maps +z to dir.
        Vec3f zAxis(0, 0, 1);
        Vec3f axis = cross(zAxis, dir);
        float dot01 = std::max(-1.f, std::min(1.f, zAxis.dot(dir)));
        float ang = std::acos(dot01) * 180.f / float(M_PI);
        if (axis.mag() > 1e-4f) g.rotate(ang, axis.normalize());
        float scl = 0.4f + 0.6f * fade;
        g.scale(scl, scl, scl);
        g.color(r * fade, gg * fade, bb * fade, fade);
        g.draw(muonCone);
        g.popMatrix();
      };

      drawMuon(vv.m1, voices[i].pt1, vv.q1);
      drawMuon(vv.m2, voices[i].pt2, vv.q2);

      // Mass ring at origin, radius scaled by log mass.
      Mesh ring;
      ring.primitive(Mesh::LINE_LOOP);
      float rad = 0.15f + 0.6f * std::log(1.f + vv.massGeV) / std::log(120.f);
      const int segs = 48;
      for (int s = 0; s < segs; ++s) {
        float a = 2.f * float(M_PI) * s / segs;
        float ringSize = rad * (0.6f + 0.4f * fade);
        ring.vertex(ringSize * std::cos(a), ringSize * std::sin(a), 0);
        float bright = fade * 0.8f;
        ring.color(bright, bright * 0.7f, bright * 0.4f, fade);
      }
      g.draw(ring);
    }

    gui.draw(g);
  }

  // -------------------------------------------------------
  // Audio.
  // -------------------------------------------------------
  void onSound(AudioIOData& io) override {
    const float dt = 1.f / float(io.framesPerSecond());
    const float gain = masterGain.get();
    while (io()) {
      float L = 0.f, R = 0.f;
      for (int i = 0; i < kVoices; ++i) {
        if (!voices[i].active.load(std::memory_order_acquire)) continue;
        float s = voices[i].tick(dt);
        // Equal-power pan.
        float p = 0.5f * (voices[i].pan + 1.f);  // 0..1
        float lG = std::cos(p * float(M_PI) * 0.5f);
        float rG = std::sin(p * float(M_PI) * 0.5f);
        L += s * lG;
        R += s * rG;
      }
      // Soft clip.
      auto sat = [](float x) { return std::tanh(x); };
      io.out(0) = sat(L * gain);
      io.out(1) = sat(R * gain);
    }
  }

  bool onKeyDown(const Keyboard& k) override {
    switch (k.key()) {
      case ' ': freezePlayback = !freezePlayback.get(); return true;
      case 'r': nextEventIdx = 0; return true;
      case 'm': {
        // Force trigger one event now.
        if (!events.empty()) triggerEvent(events[nextEventIdx]);
        nextEventIdx = (nextEventIdx + 1) % std::max<size_t>(1, events.size());
        return true;
      }
      default: return false;
    }
  }
};

int main() {
  PulsarCernApp app;
  app.title("Pulsar Synthesis · CMS Open Data");
  app.dimensions(1280, 800);
  app.configureAudio(48000, 512, 2, 0);
  app.start();
  return 0;
}
