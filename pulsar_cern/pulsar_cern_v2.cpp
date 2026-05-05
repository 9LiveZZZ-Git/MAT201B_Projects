// pulsar_cern_v2.cpp — Spectral pulsar synthesis with 256 voices, driven by
// CMS Open Data dimuon collisions, rendered as a 1:1 detector event display.
//
// Differences from v1:
//   * Spectral pulsarets: each voice owns a precomputed table built via
//     additive synthesis. Spectrum is derived per-event from M, pT, η, charge.
//   * 256-voice polyphony (was 32). Audio loop is a tight table lookup.
//   * Detector visualizer: nested wireframe layers (tracker/ECAL/HCAL/muon),
//     two converging proton beams along z, helical muon tracks bent by a
//     3.8 T solenoidal B-field (radius ρ = pT / (0.3·B)), bright hits at every
//     detector layer crossing, swarm particles streaming along trajectories.
//
// Build:
//   ./run.sh MAT201B_projects/pulsar_cern/pulsar_cern_v2.cpp
//
// Data:
//   bash MAT201B_projects/pulsar_cern/data/download_dimuon.sh
//   v2 prefers dimuon_full.csv when present, falls back to the sample CSV.

#include "al/app/al_App.hpp"
#include "al/graphics/al_Shapes.hpp"
#include "al/graphics/al_VAOMesh.hpp"
#include "al/io/al_ControlNav.hpp"
#include "al/math/al_Random.hpp"
#include "al/sound/al_Reverb.hpp"
#include "al/ui/al_ControlGUI.hpp"
#include "al/ui/al_Parameter.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

using namespace al;

// ---- constants ------------------------------------------------------------

static constexpr int   kVoices            = 256;
static constexpr int   kPulsaretTableSize = 1024;
static constexpr int   kMaxHarmonics      = 48;
static constexpr int   kHelixSamples      = 160;
static constexpr int   kSwarmPerMuon      = 24;

// LHC bunch structure (visualization). Each beam carries kBunches groups of
// kPerBunch particles; bunches drift toward origin and respawn at the far end.
static constexpr int   kBunches    = 6;
static constexpr int   kPerBunch   = 28;

// Real CMS sub-detector radii in metres.
//   Tracker outer:  ~1.10 m   (silicon strip)
//   ECAL inner:     ~1.29 m
//   HCAL outer:     ~2.95 m
//   Muon system outer: ~7.4 m
static constexpr float kLayerR[]   = {1.10f, 1.29f, 2.95f, 7.40f};
static constexpr int   kLayerCount = 4;
static const     float kBField     = 3.8f;       // T  (CMS solenoid)
static const     float kHelixVisualScale = 1.0f; // 1.0 = true ρ = pT / (0.3·B) metres

// ---- dimuon event ---------------------------------------------------------

struct DimuonEvent {
  float E1, E2;            // total muon energies (GeV) — drives hit-marker scale
  float pt1, eta1, phi1; int q1;
  float pt2, eta2, phi2; int q2;
  float px1, py1, pz1;
  float px2, py2, pz2;
  float M;
};

static bool loadDimuonCSV(const std::string& path, std::vector<DimuonEvent>& out) {
  std::ifstream f(path);
  if (!f.is_open()) return false;
  std::string line;
  std::getline(f, line);
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
      e.E1  = std::stof(cols[3]);
      e.px1 = std::stof(cols[4]);  e.py1 = std::stof(cols[5]);  e.pz1 = std::stof(cols[6]);
      e.pt1 = std::stof(cols[7]);  e.eta1 = std::stof(cols[8]); e.phi1 = std::stof(cols[9]);
      e.q1  = std::stoi(cols[10]);
      e.E2  = std::stof(cols[12]);
      e.px2 = std::stof(cols[13]); e.py2 = std::stof(cols[14]); e.pz2 = std::stof(cols[15]);
      e.pt2 = std::stof(cols[16]); e.eta2 = std::stof(cols[17]);e.phi2 = std::stof(cols[18]);
      e.q2  = std::stoi(cols[19]);
      e.M   = std::stof(cols[20]);
    } catch (...) { continue; }
    out.push_back(e);
  }
  return !out.empty();
}

static void synthDataset(std::vector<DimuonEvent>& out, int count = 200) {
  out.clear();
  rnd::Random<> rng;
  auto sampleMass = [&](int i) -> float {
    int b = i % 5;
    if (b == 0) return 3.10f + rng.uniformS() * 0.05f;
    if (b == 1) return 9.46f + rng.uniformS() * 0.4f;
    if (b == 2) return 91.2f + rng.uniformS() * 1.5f;
    if (b == 3) return 0.5f  + rng.uniform()  * 8.0f;
    return 10.f + rng.uniform() * 70.f;
  };
  for (int i = 0; i < count; ++i) {
    DimuonEvent e{};
    e.M   = sampleMass(i);
    e.pt1 = 5.f + rng.uniform() * 40.f;
    e.pt2 = 5.f + rng.uniform() * 40.f;
    e.eta1 = rng.uniformS() * 2.4f;
    e.eta2 = -e.eta1 + rng.uniformS() * 0.4f;
    e.phi1 = rng.uniformS() * float(M_PI);
    e.phi2 = e.phi1 + float(M_PI) + rng.uniformS() * 0.3f;
    e.q1   = (i & 1) ? 1 : -1; e.q2 = -e.q1;
    e.px1 = e.pt1 * std::cos(e.phi1); e.py1 = e.pt1 * std::sin(e.phi1);
    e.pz1 = e.pt1 * std::sinh(e.eta1);
    e.px2 = e.pt2 * std::cos(e.phi2); e.py2 = e.pt2 * std::sin(e.phi2);
    e.pz2 = e.pt2 * std::sinh(e.eta2);
    e.E1 = e.pt1 * std::cosh(e.eta1);   // ultra-relativistic muon: E ≈ |p|
    e.E2 = e.pt2 * std::cosh(e.eta2);
    out.push_back(e);
  }
}

// ---- spectral pulsaret table ---------------------------------------------

struct PulsaretTable {
  float s[kPulsaretTableSize];

  // Build a windowed sum of harmonics. K_cycles cycles fit across the table;
  // window is baked in so playback is one mul + lerp per sample.
  void build(int K_cycles, int windowShape,
             const float* amps, const float* phases, int nHarm) {
    for (int n = 0; n < kPulsaretTableSize; ++n) {
      float u = float(n) / float(kPulsaretTableSize - 1);
      float w;
      switch (windowShape) {
        case 1: w = 0.5f * (1.f - std::cos(2.f * float(M_PI) * u)); break;
        case 2: w = 1.f - std::fabs(2.f * u - 1.f); break;
        case 3: w = 0.42f - 0.5f*std::cos(2.f*float(M_PI)*u)
                  + 0.08f*std::cos(4.f*float(M_PI)*u); break;
        default: { float c = 2.f*u - 1.f; w = std::exp(-4.f*c*c); }
      }
      float wfPhase = u * float(K_cycles);
      float v = 0.f;
      for (int k = 1; k <= nHarm; ++k) {
        if (amps[k] == 0.f) continue;
        v += amps[k] * std::cos(2.f*float(M_PI)*float(k)*wfPhase + phases[k]);
      }
      s[n] = w * v;
    }
  }
};

// ---- voice ----------------------------------------------------------------

struct Voice {
  std::atomic<bool> active{false};

  PulsaretTable table;

  float Ff = 100.f;
  float dutyMax = 1.f;
  float amp = 0.4f;
  float pan = 0.f;
  float lifetime = 1.5f;

  // Doppler glide: factor goes from (1+a) → (1−a) over voice lifetime,
  // simulating a relativistic muon flying past the listener (forward muons
  // get amt > 0, backward get amt < 0).
  float dopplerAmt = 0.f;

  float phase01 = 0.f;
  float age = 0.f;

  // Visual mirror (read by GL thread).
  float massGeV = 0.f;
  float pt1 = 0.f, pt2 = 0.f;
  int   q1 = 0,  q2 = 0;
  Vec3f dir1{0,0,0}, dir2{0,0,0};

  inline float tick(float dt) {
    if (age >= lifetime) {
      active.store(false, std::memory_order_release);
      return 0.f;
    }
    age += dt;
    float frac = phase01 - std::floor(phase01);
    float out = 0.f;
    if (frac < dutyMax) {
      float u = frac / dutyMax;
      float idx = u * float(kPulsaretTableSize - 1);
      int   i0  = int(idx);
      float f   = idx - float(i0);
      int   i1  = i0 + 1;
      if (i1 >= kPulsaretTableSize) i1 = kPulsaretTableSize - 1;
      out = (table.s[i0] * (1.f - f) + table.s[i1] * f) * amp;
    }
    float t = age / std::max(lifetime, 1e-3f);
    float dop = 1.f + dopplerAmt * (1.f - 2.f * t);  // approach → recede
    phase01 += dt * Ff * dop;
    const float fadeIn = 0.04f, fadeOut = 0.35f;
    float env = 1.f;
    if (age < fadeIn)             env *= age / fadeIn;
    if (age > lifetime - fadeOut) env *= std::max(0.f, (lifetime - age) / fadeOut);
    return out * env;
  }
};

// ---- helical track in CMS B-field ----------------------------------------

struct Track {
  std::vector<Vec3f> pts;
  std::vector<int>   hitIdx;
  std::vector<int>   hitLayer;
};

static void buildTrack(Track& tr, Vec3f p3, int charge) {
  tr.pts.clear(); tr.hitIdx.clear(); tr.hitLayer.clear();
  float pmag = p3.mag();
  if (pmag < 1e-4f) return;
  float pT = std::sqrt(p3.x*p3.x + p3.y*p3.y);

  float rho = (pT / (0.3f * kBField)) * kHelixVisualScale;
  if (rho < 0.05f) rho = 0.05f;
  if (rho > 50.f)  rho = 50.f;

  float trFrac = pT / pmag;
  float lzFrac = p3.z / pmag;

  Vec2f trDir{ p3.x / std::max(pT, 1e-6f), p3.y / std::max(pT, 1e-6f) };
  Vec2f perp = (charge > 0) ? Vec2f{-trDir.y, trDir.x}
                            : Vec2f{ trDir.y,-trDir.x};
  Vec2f center = perp * rho;
  Vec2f r0 = -center;

  // Track length: muons in CMS routinely exit through the muon system at
  // r=7.4 m and beyond. Sample 18 m of path so high-η forward muons reach
  // the endcap and curving low-pT muons get to wrap.
  const float pathMax = 18.0f;
  for (int i = 0; i <= kHelixSamples; ++i) {
    float t  = float(i) / float(kHelixSamples);
    float L  = t * pathMax;
    float sxy = L * trFrac;
    float theta = sxy / rho;
    float sgn = (charge > 0) ? -1.f : 1.f;
    float c = std::cos(sgn * theta), s = std::sin(sgn * theta);
    Vec2f rRel{ r0.x*c - r0.y*s, r0.x*s + r0.y*c };
    Vec2f xy = center + rRel;
    float z  = lzFrac * L;
    tr.pts.push_back(Vec3f{xy.x, xy.y, z});
  }

  for (int i = 1; i < (int)tr.pts.size(); ++i) {
    float r0xy = std::sqrt(tr.pts[i-1].x * tr.pts[i-1].x + tr.pts[i-1].y * tr.pts[i-1].y);
    float r1xy = std::sqrt(tr.pts[i  ].x * tr.pts[i  ].x + tr.pts[i  ].y * tr.pts[i  ].y);
    for (int L = 0; L < kLayerCount; ++L) {
      float r = kLayerR[L];
      bool cross = (r0xy < r && r1xy >= r) || (r0xy > r && r1xy <= r);
      if (cross) { tr.hitIdx.push_back(i); tr.hitLayer.push_back(L); }
    }
  }
}

// ---- visual event ---------------------------------------------------------

struct VisualEvent {
  bool   active = false;
  float  age = 0.f;
  float  lifetime = 1.f;
  float  massGeV = 0.f;
  int    q1 = 0, q2 = 0;
  float  pt1 = 0, pt2 = 0;
  float  E1 = 0, E2 = 0;       // total muon energy → hit-marker scale
  Track  trk1, trk2;
  std::vector<float> swarmT1, swarmT2;
  std::vector<Vec3f> swarmJ1, swarmJ2;
};

// ---- beam bunch stream ----------------------------------------------------
// LHC beams are not continuous — they're trains of bunches separated by
// ~7.5 m at 25 ns crossing rate. We model each beam as kBunches bunches of
// kPerBunch tightly clustered particles drifting toward origin.

struct BeamSwarm {
  struct Bunch {
    float z;
    float vz;
    std::vector<Vec3f> off;  // local offsets (xy spread + a little z spread)
  };
  std::vector<Bunch> bunches;
  rnd::Random<> rng;

  void resetBunchOffsets(Bunch& b) {
    b.off.resize(kPerBunch);
    for (int i = 0; i < kPerBunch; ++i) {
      float a = rng.uniform() * 2.f * float(M_PI);
      float r = 0.05f * std::sqrt(rng.uniform());
      b.off[i] = Vec3f{r*std::cos(a), r*std::sin(a),
                       0.18f * (rng.uniform() - 0.5f)};
    }
  }

  void init() {
    bunches.clear();
    const float spacing = 2.0f;       // metres between bunch centres
    const float speed   = 4.0f;       // visualization advance speed (m/s)
    for (int side = 0; side < 2; ++side) {
      for (int b = 0; b < kBunches; ++b) {
        Bunch B;
        B.vz = side ? -speed : speed;
        // Stagger initial z so bunches don't all arrive simultaneously.
        B.z = (side ? 1.f : -1.f) * (0.4f + b * spacing
              + rng.uniform() * spacing * 0.2f);
        resetBunchOffsets(B);
        bunches.push_back(B);
      }
    }
  }
  void update(float dt) {
    const float spacing = 2.0f;
    for (auto& b : bunches) {
      b.z += b.vz * dt;
      // Past origin → respawn at far end with fresh offsets.
      bool past = (b.vz < 0 && b.z < -0.2f) ||
                  (b.vz > 0 && b.z >  0.2f);
      if (past) {
        b.z = -std::copysign(0.4f + kBunches * spacing
                             + rng.uniform() * spacing, b.vz);
        resetBunchOffsets(b);
      }
    }
  }
};

// ---- App ------------------------------------------------------------------

struct PulsarCernV2App : App {
  std::vector<DimuonEvent> events;
  Voice voices[kVoices];
  VisualEvent vvis[kVoices];

  size_t nextEventIdx = 0;
  double timeAccum = 0.0;

  Parameter eventsPerSecond{"events/sec", "transport", 3.0f, 0.05f, 30.f};
  ParameterBool freezePlayback{"freeze", "transport", false};
  Parameter masterGain{"gain", "synth", 0.5f, 0.f, 1.5f};
  Parameter freqMin{"Ff min Hz", "synth", 35.f, 20.f, 200.f};
  Parameter freqMax{"Ff max Hz", "synth", 1500.f, 300.f, 4000.f};
  Parameter formantRatioMin{"Fc/Ff min", "synth", 1.f, 0.5f, 8.f};
  Parameter formantRatioMax{"Fc/Ff max", "synth", 8.f, 1.f, 32.f};
  Parameter pulsaretCycles{"N cycles", "synth", 4.f, 1.f, 16.f};
  ParameterInt windowShape{"window 0G/1H/2T/3B", "synth", 0, 0, 3};
  Parameter spectrumTilt{"tilt", "spectrum", 0.7f, 0.1f, 2.5f};
  Parameter spectrumBright{"brightness", "spectrum", 1.0f, 0.f, 2.f};
  Parameter spectrumNoise{"phase incoherence", "spectrum", 0.4f, 0.f, 1.f};
  Parameter voiceLife{"voice life s", "synth", 1.4f, 0.2f, 6.f};

  // Doppler + reverb.
  Parameter dopplerAmount{"doppler amt", "fx", 0.35f, 0.f, 0.9f};
  Parameter reverbWet{"reverb wet", "fx", 0.30f, 0.f, 1.f};
  Parameter reverbDecay{"reverb decay", "fx", 0.88f, 0.5f, 0.99f};
  Parameter reverbDamping{"reverb damping", "fx", 0.35f, 0.f, 1.f};

  Reverb<float> reverb;
  float lastReverbDecay = -1.f, lastReverbDamping = -1.f;

  ControlGUI gui;

  VAOMesh vertexMarker, hitSphere;
  std::vector<VAOMesh> layerWires;
  BeamSwarm beam;

  std::vector<std::string> csvCandidates() {
    return {
      "../data/dimuon_full.csv",
      "../data/dimuon_sample.csv",
      "../../data/dimuon_full.csv",
      "../../data/dimuon_sample.csv",
      "MAT201B_projects/pulsar_cern/data/dimuon_full.csv",
      "MAT201B_projects/pulsar_cern/data/dimuon_sample.csv",
    };
  }

  void onInit() override {
    bool loaded = false;
    for (auto& p : csvCandidates()) {
      if (loadDimuonCSV(p, events)) {
        std::cout << "[pulsar_cern_v2] loaded " << events.size()
                  << " events from " << p << "\n";
        loaded = true; break;
      }
    }
    if (!loaded) {
      synthDataset(events);
      std::cout << "[pulsar_cern_v2] CSV not found; using "
                << events.size() << " synthetic events.\n";
    }
  }

  void buildLayerWires() {
    layerWires.clear();
    layerWires.resize(kLayerCount);
    for (int L = 0; L < kLayerCount; ++L) {
      VAOMesh& m = layerWires[L];
      m.primitive(Mesh::LINES);
      const float r = kLayerR[L];
      // CMS detector half-length scales with radius (rough endcap shape).
      const float halfZ = std::min(11.0f, r * 1.6f + 2.0f);
      const int   nAround = 64;
      const int   nAxial  = 4;
      for (int a = 0; a <= nAxial; ++a) {
        float z = -halfZ + (2.f * halfZ) * a / nAxial;
        for (int i = 0; i < nAround; ++i) {
          float t0 = 2.f*float(M_PI) * i / nAround;
          float t1 = 2.f*float(M_PI) * (i+1) / nAround;
          m.vertex(r*std::cos(t0), r*std::sin(t0), z);
          m.vertex(r*std::cos(t1), r*std::sin(t1), z);
        }
      }
      for (int i = 0; i < 12; ++i) {
        float t = 2.f*float(M_PI) * i / 12;
        m.vertex(r*std::cos(t), r*std::sin(t), -halfZ);
        m.vertex(r*std::cos(t), r*std::sin(t),  halfZ);
      }
      m.update();
    }
  }

  void onCreate() override {
    // Pull camera back to fit the muon system (r ≈ 7.4 m).
    nav().pos(0, 8, 22);
    nav().faceToward(Vec3d{0, 0, 0});

    addSphere(vertexMarker, 0.18f, 16, 16);
    vertexMarker.update();
    addSphere(hitSphere, 0.06f, 8, 8);
    hitSphere.update();

    buildLayerWires();
    beam.init();

    gui << eventsPerSecond << freezePlayback << masterGain
        << freqMin << freqMax << formantRatioMin << formantRatioMax
        << pulsaretCycles << windowShape
        << spectrumTilt << spectrumBright << spectrumNoise
        << voiceLife
        << dopplerAmount << reverbWet << reverbDecay << reverbDamping;
    gui.init();
  }

  // -----------------------------------------------------------------------
  // Trigger event: find a free voice, build spectral table, publish.
  // -----------------------------------------------------------------------
  void triggerEvent(const DimuonEvent& e) {
    int slot = -1;
    for (int i = 0; i < kVoices; ++i) {
      if (!voices[i].active.load(std::memory_order_acquire)) { slot = i; break; }
    }
    if (slot < 0) return;
    Voice& v = voices[slot];

    float Mc = std::max(0.5f, std::min(120.f, e.M));
    float t  = std::log(Mc / 0.5f) / std::log(120.f / 0.5f);
    v.Ff = freqMin.get() * std::pow(freqMax.get() / freqMin.get(), t);

    float etaSum = std::fabs(e.eta1) + std::fabs(e.eta2);
    float rt = std::min(1.f, etaSum / 5.f);
    float ratio = formantRatioMin.get()
                + (formantRatioMax.get() - formantRatioMin.get()) * rt;
    float Fc = v.Ff * ratio;
    int N = std::max(1, int(pulsaretCycles.get()));
    v.dutyMax = std::min(1.f, float(N) * v.Ff / std::max(Fc, 1.f));

    float ptLead = std::max(e.pt1, e.pt2);
    float ptN = std::min(1.f, std::max(0.f, (ptLead - 5.f) / 45.f));
    v.amp = 0.18f + 0.55f * ptN;
    v.lifetime = voiceLife.get() * (0.6f + ptN);

    float etaAvg = 0.5f * (e.eta1 + e.eta2);
    v.pan = std::max(-1.f, std::min(1.f, etaAvg / 2.4f));

    // Spectrum: derive harmonic content from event data.
    float amps[kMaxHarmonics + 1]   = {0};
    float phases[kMaxHarmonics + 1] = {0};
    float bright = std::min(1.f, std::max(0.f, ptLead / 50.f)) * spectrumBright.get();
    int K = std::max(2, int(4.f + bright * float(kMaxHarmonics - 6)));
    if (K > kMaxHarmonics) K = kMaxHarmonics;
    float tilt = spectrumTilt.get() * (0.5f + std::min(1.f, e.M / 100.f) * 1.5f);
    bool sameSign = (e.q1 == e.q2);
    float noiseAmt = spectrumNoise.get() * std::min(1.f, etaSum / 5.f);

    rnd::Random<> rng;
    rng.seed(uint32_t(slot * 2654435761u
                    + uint32_t(e.M * 1000.f)
                    + uint32_t(ptLead * 100.f) + 1u));
    float total = 0.f;
    for (int k = 1; k <= K; ++k) {
      if (sameSign && (k % 2) == 0) continue;
      float a = std::pow(float(k), -tilt);
      amps[k] = a;
      phases[k] = noiseAmt * rng.uniform() * 2.f * float(M_PI);
      total += std::fabs(a);
    }
    if (total > 0.f) {
      for (int k = 1; k <= K; ++k) amps[k] /= total;
    }

    v.table.build(N, windowShape.get(), amps, phases, K);

    v.massGeV = e.M;
    v.pt1 = e.pt1; v.pt2 = e.pt2;
    v.q1  = e.q1;  v.q2  = e.q2;
    v.dir1 = Vec3f{e.px1, e.py1, e.pz1};
    v.dir2 = Vec3f{e.px2, e.py2, e.pz2};
    if (v.dir1.mag() > 0) v.dir1.normalize();
    if (v.dir2.mag() > 0) v.dir2.normalize();

    // Doppler glide amount: average dz of the muon-pair direction.
    // Forward pair (dz>0) → blue→red glide; backward → red→blue glide.
    float meanDz = 0.5f * (v.dir1.z + v.dir2.z);
    v.dopplerAmt = dopplerAmount.get() * meanDz;

    v.phase01 = 0.f;
    v.age = 0.f;

    VisualEvent& vv = vvis[slot];
    vv.active = true;
    vv.age = 0.f;
    vv.lifetime = v.lifetime;
    vv.massGeV = e.M;
    vv.q1 = e.q1; vv.q2 = e.q2;
    vv.pt1 = e.pt1; vv.pt2 = e.pt2;
    vv.E1 = e.E1; vv.E2 = e.E2;
    buildTrack(vv.trk1, Vec3f{e.px1, e.py1, e.pz1}, e.q1);
    buildTrack(vv.trk2, Vec3f{e.px2, e.py2, e.pz2}, e.q2);

    vv.swarmT1.assign(kSwarmPerMuon, 0.f);
    vv.swarmJ1.assign(kSwarmPerMuon, Vec3f{0,0,0});
    vv.swarmT2.assign(kSwarmPerMuon, 0.f);
    vv.swarmJ2.assign(kSwarmPerMuon, Vec3f{0,0,0});
    rnd::Random<> srng;
    srng.seed(uint32_t(slot * 92837111u + uint32_t(e.M * 10.f) + 7u));
    for (int i = 0; i < kSwarmPerMuon; ++i) {
      vv.swarmT1[i] = srng.uniform() * 0.3f;
      vv.swarmJ1[i] = Vec3f{srng.uniformS(), srng.uniformS(), srng.uniformS()} * 0.06f;
      vv.swarmT2[i] = srng.uniform() * 0.3f;
      vv.swarmJ2[i] = Vec3f{srng.uniformS(), srng.uniformS(), srng.uniformS()} * 0.06f;
    }

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
    beam.update(float(dt));
    for (int i = 0; i < kVoices; ++i) {
      VisualEvent& vv = vvis[i];
      if (!vv.active) continue;
      vv.age += float(dt);
      if (!voices[i].active.load(std::memory_order_acquire) || vv.age >= vv.lifetime) {
        vv.active = false;
        continue;
      }
      float adv = float(dt) / std::max(0.05f, vv.lifetime);
      for (int s = 0; s < kSwarmPerMuon; ++s) {
        vv.swarmT1[s] = std::min(1.f, vv.swarmT1[s] + adv);
        vv.swarmT2[s] = std::min(1.f, vv.swarmT2[s] + adv);
      }
    }
  }

  void drawDetector(Graphics& g) {
    Mesh axis;
    axis.primitive(Mesh::LINES);
    axis.vertex(0, 0, -14.f); axis.color(0.18f, 0.32f, 0.55f);
    axis.vertex(0, 0,  14.f); axis.color(0.18f, 0.32f, 0.55f);
    g.draw(axis);

    static const float layerCol[kLayerCount][3] = {
      {0.30f, 0.55f, 0.70f},
      {0.85f, 0.65f, 0.20f},
      {0.60f, 0.30f, 0.55f},
      {0.45f, 0.55f, 0.45f},
    };
    for (int L = 0; L < kLayerCount; ++L) {
      g.color(layerCol[L][0]*0.55f, layerCol[L][1]*0.55f, layerCol[L][2]*0.55f, 0.9f);
      g.draw(layerWires[L]);
    }

    // Beam bunches: each bunch is kPerBunch tightly clustered points.
    Mesh bp;
    bp.primitive(Mesh::POINTS);
    const float farZ = kBunches * 2.0f + 2.f;
    for (const auto& B : beam.bunches) {
      float fade = 1.f - std::min(1.f, std::fabs(B.z) / farZ);
      // Brighter near the centre of the bunch in z (Gaussian-ish via fade).
      for (const auto& off : B.off) {
        bp.vertex(off.x, off.y, B.z + off.z);
        bp.color(1.f, 0.92f, 0.55f, 0.35f + 0.65f * fade);
      }
    }
    g.draw(bp);

    g.color(0.85f, 0.9f, 1.f);
    g.draw(vertexMarker);
  }

  void drawMuon(Graphics& g, const Track& tr, int q, float fade, float energyGeV,
                const std::vector<float>& swarmT,
                const std::vector<Vec3f>& swarmJ) {
    if (tr.pts.empty()) return;
    float r = (q > 0) ? 0.95f : 0.30f;
    float b = (q > 0) ? 0.30f : 0.95f;
    float gC = 0.50f;

    Mesh trail;
    trail.primitive(Mesh::LINE_STRIP);
    int n = (int)tr.pts.size();
    for (int i = 0; i < n; ++i) {
      float t = float(i) / float(n - 1);
      float a = (1.f - 0.85f * t) * fade;
      trail.vertex(tr.pts[i]);
      trail.color(r * a, gC * a, b * a, a);
    }
    g.draw(trail);

    // Per-hit scale: log(E/1 GeV) so a 100 GeV muon dot is ~2x a 5 GeV dot.
    // (CSV gives total muon E only; per-layer deposit isn't in this dataset.)
    float eScale = 0.5f + 0.7f * std::log(1.f + std::max(0.f, energyGeV))
                                   / std::log(120.f);
    for (size_t k = 0; k < tr.hitIdx.size(); ++k) {
      int idx = tr.hitIdx[k];
      if (idx < 0 || idx >= n) continue;
      g.pushMatrix();
      g.translate(tr.pts[idx]);
      float scl = eScale * (0.6f + 0.4f * fade);
      g.scale(scl, scl, scl);
      g.color(r * fade, (gC + 0.3f) * fade, b * fade, fade);
      g.draw(hitSphere);
      g.popMatrix();
    }

    Mesh swarm;
    swarm.primitive(Mesh::POINTS);
    for (size_t s = 0; s < swarmT.size(); ++s) {
      float t = swarmT[s];
      if (t >= 1.f) continue;
      int i0 = std::min(n - 2, int(t * float(n - 1)));
      float fr = t * float(n - 1) - float(i0);
      Vec3f p = tr.pts[i0] * (1.f - fr) + tr.pts[i0 + 1] * fr + swarmJ[s];
      swarm.vertex(p);
      float a = (1.f - t) * fade;
      swarm.color(r * a + 0.2f * a, gC * a + 0.2f * a, b * a + 0.2f * a, a);
    }
    g.draw(swarm);
  }

  void onDraw(Graphics& g) override {
    g.clear(0.015f, 0.018f, 0.03f);
    g.lighting(false);
    g.depthTesting(true);
    g.blending(true);
    g.blendTrans();
    g.pointSize(3.0f);

    drawDetector(g);

    for (int i = 0; i < kVoices; ++i) {
      VisualEvent& vv = vvis[i];
      if (!vv.active) continue;
      float life = vv.age / std::max(0.001f, vv.lifetime);
      float fade = 1.f - std::min(1.f, life);

      if (vv.age < 0.12f) {
        float k = 1.f - vv.age / 0.12f;
        g.pushMatrix();
        float r = 0.18f + 0.6f * (1.f - k);
        g.scale(r, r, r);
        g.color(1.f, 0.95f * k, 0.7f * k, 0.6f * k);
        g.draw(vertexMarker);
        g.popMatrix();
      }

      drawMuon(g, vv.trk1, vv.q1, fade, vv.E1, vv.swarmT1, vv.swarmJ1);
      drawMuon(g, vv.trk2, vv.q2, fade, vv.E2, vv.swarmT2, vv.swarmJ2);

      Mesh ring;
      ring.primitive(Mesh::LINE_LOOP);
      float rad = 0.12f + 0.6f * std::log(1.f + vv.massGeV) / std::log(120.f);
      const int segs = 64;
      for (int s = 0; s < segs; ++s) {
        float a = 2.f * float(M_PI) * float(s) / float(segs);
        float rr = rad * (0.6f + 0.4f * fade);
        ring.vertex(rr * std::cos(a), rr * std::sin(a), 0.f);
        float bC = fade * 0.7f;
        ring.color(bC, bC * 0.7f, bC * 0.4f, fade);
      }
      g.draw(ring);
    }

    gui.draw(g);
  }

  void onSound(AudioIOData& io) override {
    const float dt = 1.f / float(io.framesPerSecond());
    const float gain = masterGain.get();
    const float wet  = reverbWet.get();
    const float dry  = 1.f - wet;
    // Update reverb params once per buffer (they're not sample-rate-critical).
    float dec = reverbDecay.get();
    float dmp = reverbDamping.get();
    if (dec != lastReverbDecay)   { reverb.decay(dec);   lastReverbDecay = dec; }
    if (dmp != lastReverbDamping) { reverb.damping(dmp); lastReverbDamping = dmp; }

    while (io()) {
      float dryL = 0.f, dryR = 0.f, mono = 0.f;
      for (int i = 0; i < kVoices; ++i) {
        if (!voices[i].active.load(std::memory_order_acquire)) continue;
        float s = voices[i].tick(dt);
        float p = 0.5f * (voices[i].pan + 1.f);
        float lG = std::cos(p * float(M_PI) * 0.5f);
        float rG = std::sin(p * float(M_PI) * 0.5f);
        dryL += s * lG;
        dryR += s * rG;
        mono += s;
      }
      // Dattorro plate reverb: mono in, stereo wet out.
      float wL = 0.f, wR = 0.f;
      reverb(mono * 0.5f, wL, wR, 0.6f);
      auto sat = [](float x) { return std::tanh(x); };
      io.out(0) = sat((dryL * dry + wL * wet) * gain);
      io.out(1) = sat((dryR * dry + wR * wet) * gain);
    }
  }

  bool onKeyDown(const Keyboard& k) override {
    // WASD: allolib's default uses 'x' for back and 's' for halt; remap 's'
    // to backwards motion. Match allolib's velocity scaling so feel is consistent.
    double v = 0.25;
    if (k.alt())   v *= 10.0;
    if (k.shift()) v *= 0.1;
    switch (k.key()) {
      case ' ': freezePlayback = !freezePlayback.get(); return true;
      case 'r': nextEventIdx = 0; return true;
      case 'm':
        if (!events.empty()) triggerEvent(events[nextEventIdx]);
        nextEventIdx = (nextEventIdx + 1) % std::max<size_t>(1, events.size());
        return true;
      case 's': case 'S':
        nav().moveF(-v);
        return true;  // consume to suppress allolib's default halt-on-s
      default: return false;
    }
  }

  bool onKeyUp(const Keyboard& k) override {
    switch (k.key()) {
      case 's': case 'S':
        nav().moveF(0);
        return true;
      default: return false;
    }
  }
};

int main() {
  // Heap-allocate: the App contains Voice[256] with 4KB tables each (~1MB),
  // which exceeds Windows' default main-thread stack of 1 MB.
  auto app = std::make_unique<PulsarCernV2App>();
  app->title("Pulsar Synthesis v2 . CMS Detector Display");
  app->dimensions(1480, 880);
  app->configureAudio(48000, 512, 2, 0);
  app->start();
  return 0;
}
