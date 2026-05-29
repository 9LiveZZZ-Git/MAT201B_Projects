#pragma once
// AudioEngine (v2) — World of Shadow Work's generative score. The Conductor's live graph-walk
// IS the music; the ~80s depth tide is the macro-clock of an endless extraction cycle.
//
// THE STORY (told in sound): the machine descends into its core to extract (bass swells, dense
// grain-mass), blooms a brief just-intonation "magic" out of the crushed material, then a Shepard
// tone keeps rising forever — labor that accumulates and never resolves. Synthesis of three
// composer designs (Xenakis stochastics + sieves + clouds; Kuchera-Morin harmonic-series/JI +
// data-as-score; dramaturgical Shepard/Euclidean/bass/mood arc).
//
// LAYERS (all allolib + plain C++; gam not required — hand-rolled DSP + a tiny WAV loader):
//   - BASS (sampled organ + procedural sub): JI, rhythmic on the Euclidean clock, swells when sunk.
//   - MELODY (kalimba/agogo samples or additive): stochastic Markov walk, NOT a fixed arp.
//   - PAD: just-intonation partials gliding to the cluster's tonal centre, blooms with depth.
//   - GRAIN CLOUD: Poisson pulsarets, density/hesitation-driven (Xenakis mass).
//   - SHEPARD: octave-stacked sines under a Gaussian log-freq window — the endless rise.
//   - WHISPER: formant breath of each classified word, dissolving into grains.
//   - TRACE: bowed-psaltery (sampled) warm presence at surfacing photos.
//
// Threading: PRIMARY-only. Sim thread (onAnimate) -> lock-free SPSC ring (events) + atomics
// (continuous controls). Audio thread (render) drains them and owns all voices. No alloc/lock in
// the audio callback. (Reviewed-correct lock-free core retained; ready_ now atomic w/ release.)
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

  // ---- PRIMARY sim thread (onAnimate) ----
  void onArrival(int node, int type, float density, int cluster);
  void igniteArp(const std::vector<std::pair<int, float>>& nbrs, int cluster);  // -> Markov phrase
  void traceOn(int slot, int node, float pan);
  void traceOff(int slot);
  void whisper(const std::string& word, int node, float pan);
  void update(float dt, float hesitation, float depth, float progress, float focusPan);

  // ---- AUDIO thread (onSound) ----
  void render(al::AudioIOData& io);

 private:
  // ===================== scale / tuning =====================
  static constexpr int MAXMODE = 16;
  int   modeBuf_[MAXMODE] = {0, 2, 3, 5, 7, 8, 10};   // POD scale (audio reads this, not a vector)
  int   modeN_ = 7;
  float root_ = 220.f;                                 // A3
  float degHz(int degree) const;                       // equal-tempered scale degree -> Hz
  int   nodeDegree(int node, int cluster) const;       // stable pitch per node
  int   snapDegree(int d) const { return d; }          // (degrees are already scale indices)

  // ===================== CC0 sample bank =====================
  struct Sample { std::vector<float> data; float srcSR = 44100.f; float rootHz = 220.f; bool pitched = true; };
  std::vector<Sample> samples_;
  int sBass_ = -1, sPluck_ = -1, sBell_ = -1, sTrace_ = -1;   // role -> sample index (-1 = synth)
  void loadSamples(const std::string& assetDir);

  // ===================== lock-free event ring (SPSC) =====================
  struct Ev { int kind; float hz, amp, pan; int layer, islot; float a, b, c; };
  enum { EV_NOTE = 0, EV_TRACE_ON = 1, EV_TRACE_OFF = 2, EV_WHISPER = 3, EV_RHYTHM = 4 };
  static constexpr int RING = 1024;
  std::array<Ev, RING>  ring_{};
  std::atomic<uint32_t> rHead_{0}, rTail_{0};
  bool push(const Ev& e);
  bool pop(Ev& e);

  // ===================== continuous controls (atomic, sim -> audio) =====================
  std::atomic<float> cHesitation_{0.f}, cDepth_{1.f}, cFocusPan_{0.f}, cActivity_{0.f};
  std::atomic<float> cClusterRoot_{220.f};            // current cluster tonal centre (Hz)
  std::atomic<int>   cClusterId_{0}, cVisits_{0};

  // ===================== sim-side state (sim thread only) =====================
  std::vector<std::pair<float, int>> mel_;   // pending melody notes: (hz, layer)
  std::size_t melIdx_ = 0;
  float       melTimer_ = 0.f, melPan_ = 0.f;
  int         melDeg_ = 0;                    // Markov running degree
  bool        lastLeap_ = false;
  uint32_t    mprng_ = 0x1234567u;            // melodic PRNG (seeded per node)
  float       activity_ = 0.f, simHes_ = 0.f, simDepth_ = 1.f, simDensity_ = 0.5f;
  int         visits_ = 0, euK_ = -1, euN_ = -1;     // last-pushed Euclidean (k,n)
  int         curNode_ = 0, clusterCur_ = 0;         // context for the Markov seed

  // ===================== audio-side voices =====================
  struct Voice {
    bool  on = false; int layer = 0;          // 0 pluck,1 bell,2 grain,3 trace,4 whisper,5 bass
    float hz = 220, amp = 0, pan = 0, age = 0, life = 1, atk = 0.02f;
    float phase = 0, lp = 0;
    int   tslot = -1;
    // sampled playback (smp != null)
    const float* smp = nullptr; int smpLen = 0; float smpPos = 0, smpRate = 1.f;
    // additive partials (precomputed at trigger)
    int   K = 5; float pm[6] = {1,2,3,4,5,6}, pa[6] = {0}; float pnorm = 1.f;
    // whisper formants
    float fmt[3] = {500, 1500, 2500}, fcoef[3][3] = {}, fz_[3][2] = {};
    int   syl = 2; float body = 0.6f; uint32_t grng = 0x2545F491u;
  };
  static constexpr int NVOX = 128;
  std::array<Voice, NVOX> vox_{};
  int  allocVoice();
  void trigger(const Ev& e);
  float tickVoice(Voice& v, float isr, float depth, float bright);

  // ===================== always-on layers (audio thread) =====================
  // JI pad: up to 4 partials gliding to cluster-root just ratios
  static constexpr int PADN = 4;
  float padHz_[PADN] = {220, 330, 440, 550}, padPhase_[PADN] = {}, padVib_[PADN] = {}, padLp_[PADN] = {}, padAmp_[PADN] = {};
  // sub bass pedal (procedural floor under the sampled bass)
  float subPhase_ = 0.f, subAmp_ = 0.f, subLp_ = 0.f, subHz_ = 55.f;
  // Shepard ladder
  static constexpr int SHEP = 6;
  float shepPhase_ = 0.f, shepPh_[SHEP] = {}, shepRate_ = 0.0333f, shepRateTgt_ = 0.0333f, shepGain_ = 0.f;
  // Euclidean clock
  uint32_t euMask_ = 0x49u; int euLen_ = 8, euStep_ = 0; float euPhase_ = 0.f;
  // grain spawner
  float grainTimer_ = 0.f; uint32_t rng_ = 0x9E3779B9u; float frand();
  // smoothed/derived
  float lastDepth_ = 1.f, moodLP_ = 1.f, tension_ = 0.f, padBloom_ = 1.f, reverbWet_ = 0.25f;

  al::Reverb<float> reverb_;
  float  lastDecay_ = -1.f, lastDamp_ = -1.f;
  double sr_ = 44100.0;
  std::atomic<bool> ready_{false};

  void loadManifest(const std::string& assetDir);
  void fireBass();                            // audio: a bass onset on a Euclidean pulse
  float jiBassHz() const;                     // current JI bass fundamental
};

}  // namespace wosw
