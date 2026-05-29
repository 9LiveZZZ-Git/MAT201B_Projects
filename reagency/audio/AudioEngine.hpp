#pragma once
// AudioEngine (v3.2) — World of Shadow Work generative score.
//
// Mix: the WORD-WHISPER is the lead (blended, ducks the bed AND the low end); a KICK + a Risset
// eternal-accelerando tick layer + a pitched-TIMPANI 2nd Euclidean line carry a varied,
// never-4/4 rhythm; everything else (moving JI drone, melody, grains, Shepard) is a quiet bed.
// Master tilts toward a pink reference (+10 dB <300 Hz shelf + HF roll-off). Every onset gets a
// random 2-156 ms micro-timing offset (humanized). Occasional synth INDUSTRIAL clangs as FX.
//
// LAYERS: 0 pluck, 1 bell, 2 grain/tick, 3 trace, 4 WHISPER(lead), 5 organ pedal, 6 KICK,
//         7 TIMPANI (pitched 2nd-Euclidean), 8 INDUSTRIAL clang.
// Threading: PRIMARY-only; lock-free SPSC ring + atomics; audio thread owns all voices + a
// pending-event scheduler; no alloc/lock in the callback.
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

  // ---------- CC0 sample bank ----------
  struct Sample { std::vector<float> data; float srcSR = 44100.f; float rootHz = 220.f; bool pitched = true; };
  struct Role   { int idx[24]; int n = 0; int rr = 0; };
  std::vector<Sample> samples_;
  Role rBass_, rPluck_, rBell_, rTrace_, rTimp_, rMetal_;
  void loadSamples(const std::string& assetDir);
  int  pickSample(Role& r, float hz);

  // ---------- lock-free event ring (SPSC) ----------
  struct Ev { int kind; float hz, amp, pan; int layer, islot; float a, b, c; };
  enum { EV_NOTE = 0, EV_TRACE_ON = 1, EV_TRACE_OFF = 2, EV_WHISPER = 3, EV_RHYTHM = 4, EV_RHYTHM2 = 5 };
  static constexpr int RING = 1024;
  std::array<Ev, RING>  ring_{};
  std::atomic<uint32_t> rHead_{0}, rTail_{0};
  bool push(const Ev& e); bool pop(Ev& e);

  // pending-event scheduler (audio thread): random 2-156 ms micro-timing on every onset
  struct Pend { Ev e; int left = 0; bool used = false; };
  std::array<Pend, 256> pend_{};
  void schedule(const Ev& e);

  // ---------- continuous controls (atomic, sim -> audio) ----------
  std::atomic<float> cHesitation_{0.f}, cDepth_{1.f}, cFocusPan_{0.f}, cActivity_{0.f};
  std::atomic<float> cClusterRoot_{220.f};
  std::atomic<int>   cClusterId_{0}, cVisits_{0};
  std::atomic<float> cMoodBright_{0.5f}, cMoodDens_{0.3f}, cMoodWet_{0.3f};
  // arrangement gates (sim sets per-section targets; audio glides) + dubstep growl amount
  std::atomic<float> cDroneGate_{1.f}, cWhisperGate_{1.f}, cGrainGate_{1.f}, cGrowl_{0.f};
  std::atomic<float> cFmt0_{500.f}, cFmt1_{1500.f}, cFmt2_{2500.f};   // formants of the last whispered word

  // ---------- sim-side state ----------
  std::vector<std::pair<float, int>> mel_;
  std::size_t melIdx_ = 0;
  float       melTimer_ = 0.f, melPan_ = 0.f;
  int         melDeg_ = 0; bool lastLeap_ = false;
  uint32_t    mprng_ = 0x1234567u;
  float       activity_ = 0.f, simHes_ = 0.f, simDepth_ = 1.f, simDensity_ = 0.5f;
  int         visits_ = 0, euK_ = -1, euN_ = -1, euK2_ = -1, euN2_ = -1, curNode_ = 0, clusterCur_ = 0;
  uint32_t    sprng_ = 0x55AA55AAu;
  float       moodTimer_ = 0.f, moodBright_ = 0.5f, moodDens_ = 0.3f, moodWet_ = 0.3f;
  float       mbTgt_ = 0.5f, mdTgt_ = 0.3f, mwTgt_ = 0.3f;
  // arrangement section + recognizable public-domain melody player (sim thread)
  int         section_ = 0, melTune_ = 0, melNote_ = 0; bool melodyOn_ = false; float melNoteTimer_ = 0.f;

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
  static constexpr int NVOX = 160;
  std::array<Voice, NVOX> vox_{};
  int  allocVoice();
  int  allocCapped(int layer, int cap);
  void trigger(const Ev& e);
  float tickVoice(Voice& v, float isr, float depth, float bright);

  // ---------- always-on layers ----------
  static constexpr int PADN = 4;
  float padHz_[PADN] = {220, 330, 440, 550}, padPhase_[PADN] = {}, padVib_[PADN] = {}, padLp_[PADN] = {}, padAmp_[PADN] = {};
  float padTrem_[PADN] = {}, padDrift_[PADN] = {};     // moving-drone LFOs
  float beatPhase_ = 0.f;
  float subPhase_ = 0.f, subAmp_ = 0.f, subLp_ = 0.f, subHz_ = 55.f;
  static constexpr int SHEP = 6;
  float shepPhase_ = 0.f, shepPh_[SHEP] = {}, shepRate_ = 0.04f, shepRateTgt_ = 0.04f, shepGain_ = 0.f;
  float nzLp_ = 0.f, nzLp2_ = 0.f;
  uint32_t euMask_ = 0x49u; int euLen_ = 8, euStep_ = 0; float euPhase_ = 0.f;
  uint32_t euMask2_ = 0x15u; int euLen2_ = 5, euStep2_ = 0; float euPhase2_ = 0.f; int timpDeg_ = 0;
  static constexpr int RIS = 4;
  float risP_ = 0.f, risPh_[RIS] = {};                 // Risset eternal-accelerando tick streams
  float fxTimer_ = 6.f;                                // industrial-clang scheduler
  float grainTimer_ = 0.f; uint32_t rng_ = 0x9E3779B9u; float frand();
  int   lastClusterId_ = -999;
  float duckGain_ = 1.f, lowDuck_ = 1.f;               // bed + low-end duck under the whisper
  // arrangement gates (glided) + dubstep growl LFO/filter + ping-pong delay (grains+whispers)
  float droneGate_ = 1.f, whisperGate_ = 1.f, grainGate_ = 1.f, growl_ = 0.f;
  float wobPhase_ = 0.f, growlLp_ = 0.f, growlBp_ = 0.f, wobMult_ = 1.f, wobChange_ = 0.f;
  float growlDuty_ = 1.f, growlDutyTgt_ = 1.f;         // sometimes off -> clean drone
  float growlFmtCoef_[3][3] = {}, growlFz_[3][2] = {}, lastFmt_[3] = {0.f, 0.f, 0.f};  // talking-growl formants
  std::vector<float> ppL_, ppR_; int ppN_ = 0, ppPos_ = 0;
  float panLFO_ = 0.f;
  float lastDepth_ = 1.f, moodLP_ = 1.f, tension_ = 0.f, padBloom_ = 1.f, reverbWet_ = 0.3f;
  float masterLoL_ = 0.f, masterLoR_ = 0.f, masterHiL_ = 0.f, masterHiR_ = 0.f;

  al::Reverb<float> reverb_;
  float  lastDecay_ = -1.f, lastDamp_ = -1.f;
  double sr_ = 44100.0;
  std::atomic<bool> ready_{false};

  void loadManifest(const std::string& assetDir);
  void fireKick();
  void fireTimp();          // pitched timpani on the 2nd Euclidean
  void fireClang();         // industrial metallic FX
  void fireAnd();           // subtle off-beat "and" tick accenting the main beat
  float jiBassHz() const;
};

}  // namespace wosw
