#pragma once
// AudioEngine (v3) — World of Shadow Work's generative score.
//
// MIX PHILOSOPHY (v3): the WORD-WHISPER is the lead voice and the focus — it is foregrounded and
// DUCKS everything else; a KICK DRUM carries the Euclidean beat; every other layer (pad, melody,
// grains, Shepard, bass) is a quiet, blended BACKGROUND BED. Subtle, coherent, spacious. The
// Conductor's live walk is the score; the depth tide + a slow STOCHASTIC mood scheduler give a
// long, varied, non-repeating form. Synthesis of three composer designs (Xenakis stochastics +
// Kuchera-Morin harmonic-series/JI + dramaturgical Shepard/Euclidean/bass arc).
//
// LAYERS: 0 pluck (melody/image), 1 bell (word/accent), 2 grain, 3 trace, 4 WHISPER (lead),
//         5 organ pedal (low harmonic glue, on cluster change), 6 KICK (the beat, on Euclidean pulses).
// CC0 samples (assets/audio, VCSL) with per-role nearest-pitch + round-robin so timbres vary.
//
// Threading: PRIMARY-only; lock-free SPSC ring + atomics; audio thread owns all voices; no
// alloc/lock in the callback (reviewed-correct core retained; all review fixes folded in).
#include "al/io/al_AudioIOData.hpp"
#include "al/sound/al_Reverb.hpp"

#include <array>
#include <atomic>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace wosw {

class AudioEngine {
 public:
  void init(const std::string& assetDir, double sampleRate);
  bool ready() const { return ready_.load(std::memory_order_acquire); }

  void onArrival(int node, int type, float density, int cluster);
  void igniteArp(const std::vector<std::pair<int, float>>& nbrs, int cluster);
  void traceOn(int slot, int node, float pan);
  void traceOff(int slot);
  void whisper(const std::string& word, int node, float pan);
  void update(float dt, float hesitation, float depth, float progress, float focusPan);

  void render(al::AudioIOData& io);

 private:
  // ---------- scale ----------
  static constexpr int MAXMODE = 16;
  int   modeBuf_[MAXMODE] = {0, 2, 3, 5, 7, 8, 10};
  int   modeN_ = 7;
  float root_ = 220.f;
  float degHz(int degree) const;
  int   nodeDegree(int node, int cluster) const;

  // ---------- CC0 sample bank (per-role lists; nearest-pitch + round-robin) ----------
  struct Sample { std::vector<float> data; float srcSR = 44100.f; float rootHz = 220.f; bool pitched = true; };
  struct Role   { int idx[24]; int n = 0; int rr = 0; };
  std::vector<Sample> samples_;
  Role rBass_, rPluck_, rBell_, rTrace_;
  void loadSamples(const std::string& assetDir);
  int  pickSample(Role& r, float hz);          // audio thread: choose a sample index (-1 = none)

  // ---------- lock-free event ring (SPSC) ----------
  struct Ev { int kind; float hz, amp, pan; int layer, islot; float a, b, c; };
  enum { EV_NOTE = 0, EV_TRACE_ON = 1, EV_TRACE_OFF = 2, EV_WHISPER = 3, EV_RHYTHM = 4 };
  static constexpr int RING = 1024;
  std::array<Ev, RING>  ring_{};
  std::atomic<uint32_t> rHead_{0}, rTail_{0};
  bool push(const Ev& e); bool pop(Ev& e);

  // ---------- continuous controls (atomic, sim -> audio) ----------
  std::atomic<float> cHesitation_{0.f}, cDepth_{1.f}, cFocusPan_{0.f}, cActivity_{0.f};
  std::atomic<float> cClusterRoot_{220.f};
  std::atomic<int>   cClusterId_{0}, cVisits_{0};
  std::atomic<float> cMoodBright_{0.5f}, cMoodDens_{0.3f}, cMoodWet_{0.3f};   // slow random mood

  // ---------- sim-side state ----------
  std::vector<std::pair<float, int>> mel_;
  std::size_t melIdx_ = 0;
  float       melTimer_ = 0.f, melPan_ = 0.f;
  int         melDeg_ = 0; bool lastLeap_ = false;
  uint32_t    mprng_ = 0x1234567u;
  float       activity_ = 0.f, simHes_ = 0.f, simDepth_ = 1.f, simDensity_ = 0.5f;
  int         visits_ = 0, euK_ = -1, euN_ = -1, curNode_ = 0, clusterCur_ = 0;
  // stochastic mood scheduler
  uint32_t    sprng_ = 0x55AA55AAu;
  float       moodTimer_ = 0.f, moodBright_ = 0.5f, moodDens_ = 0.3f, moodWet_ = 0.3f;
  float       mbTgt_ = 0.5f, mdTgt_ = 0.3f, mwTgt_ = 0.3f;

  // ---------- audio-side voices ----------
  struct Voice {
    bool  on = false; int layer = 0;
    float hz = 220, amp = 0, pan = 0, age = 0, life = 1, atk = 0.02f;
    float phase = 0, lp = 0;
    int   tslot = -1;
    const float* smp = nullptr; int smpLen = 0; float smpPos = 0, smpRate = 1.f;
    int   K = 5; float pm[6] = {1,2,3,4,5,6}, pa[6] = {0}; float pnorm = 1.f;
    float fmt[3] = {500, 1500, 2500}, fcoef[3][3] = {}, fz_[3][2] = {};
    int   syl = 2; float body = 0.6f; uint32_t grng = 0x2545F491u;
  };
  static constexpr int NVOX = 128;
  std::array<Voice, NVOX> vox_{};
  int  allocVoice();
  int  allocCapped(int layer, int cap);        // steal-oldest within a layer cap
  void trigger(const Ev& e);
  float tickVoice(Voice& v, float isr, float depth, float bright);

  // ---------- always-on layers ----------
  static constexpr int PADN = 4;
  float padHz_[PADN] = {220, 330, 440, 550}, padPhase_[PADN] = {}, padVib_[PADN] = {}, padLp_[PADN] = {}, padAmp_[PADN] = {};
  float beatPhase_ = 0.f;                       // second, detuned fifth -> gentle beating
  float subPhase_ = 0.f, subAmp_ = 0.f, subLp_ = 0.f, subHz_ = 55.f;
  static constexpr int SHEP = 6;
  float shepPhase_ = 0.f, shepPh_[SHEP] = {}, shepRate_ = 0.04f, shepRateTgt_ = 0.04f, shepGain_ = 0.f;
  float nzLp_ = 0.f, nzLp2_ = 0.f;              // airy noise bed
  uint32_t euMask_ = 0x49u; int euLen_ = 8, euStep_ = 0; float euPhase_ = 0.f;
  float grainTimer_ = 0.f; uint32_t rng_ = 0x9E3779B9u; float frand();
  int   lastClusterId_ = -999;                  // organ pedal on cluster change
  float duckGain_ = 1.f;                        // bed ducks under the whisper lead
  float panLFO_ = 0.f;                          // slow spatial drift
  float lastDepth_ = 1.f, moodLP_ = 1.f, tension_ = 0.f, padBloom_ = 1.f, reverbWet_ = 0.3f;

  al::Reverb<float> reverb_;
  float  lastDecay_ = -1.f, lastDamp_ = -1.f;
  double sr_ = 44100.0;
  std::atomic<bool> ready_{false};

  void loadManifest(const std::string& assetDir);
  void fireKick();
  float jiBassHz() const;
};

}  // namespace wosw
