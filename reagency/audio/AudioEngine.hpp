#pragma once
// AudioEngine (v3.2) — World of Shadow Work generative score.
//
// Mix: the WORD-WHISPER is the lead (sits on top at its own gain — NO bed/low-end ducking); a KICK + a Risset
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
#include "al/sound/al_Spatializer.hpp"   // al::Spatializer base — dome placement (Stage 1)
#include "al/sound/al_Speaker.hpp"       // al::Speakers layout
#include <memory>
#if defined(WOSW_HAVE_DECORR)
#include "al_ext/spatialaudio/al_Decorrelation.hpp"   // al::Decorrelation — diffuse bed across the dome (Stage 2)
#endif

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
  // LATER tier (SIEVES): overload that ALSO bakes the pitch-sieves from a plain int[nClusters]
  // galaxy-cluster size histogram (main_reagency computes it via field.clusterOf). clusterCounts may
  // be null (procedural galaxy) -> a fixed fallback triple is used. Builds sieves BEFORE ready_ release.
  void init(const std::string& assetDir, double sampleRate, const int* clusterCounts, int nClusters);
  bool ready() const { return ready_.load(std::memory_order_acquire); }

  void onArrival(int node, int type, float density, int cluster);
  void igniteArp(const std::vector<std::pair<int, float>>& nbrs, int cluster);
  void traceOn(int slot, int node, float pan);
  void traceOff(int slot);
  // 2a/2b: whisper() gains optional per-voice chorus detune (ratio, 1=center) + pan-spread fraction (-1..1);
  // defaults keep every existing whisper() callsite source-compatible.
  void whisper(const std::string& word, int node, float pan, float detune = 1.f, float spreadFrac = 0.f);
  void whisperChorus(const std::string& word, int node, float pan, int n, float spread);   // 2b: n<=0 -> use cChorusN_; 1..5 explicit
  // 2c: emergePhase/emergeActive carry the SYNCED runtime emergence (WoSWState.emergePhase / emergeDream>=0)
  // from the SIM thread into cEmerge_. The audio thread never reads WoSWState. Defaulted -> source-compatible.
  void update(float dt, float hesitation, float depth, float progress, float focusPan, int act = 1,
              float emergePhase = 0.f, bool emergeActive = false);

  void render(al::AudioIOData& io);

  // ---- dome spatialization (Stage 1: Ambisonics placement) ----
  // Build the spatializer ONCE off the audio thread (call from onCreate). dome=false leaves the
  // spatializer NULL => the legacy stereo path runs UNCHANGED (Mac dev). Env WSW_FORCE_SPAT=1
  // forces a StereoPanner spatial path on for local testing.
  void initSpatial(bool dome, int blockSize);
  // Sim-thread: push listener-relative source directions (lead = THEM focus node, bed = front).
  void setSpatialDirs(float lx, float ly, float lz, float bx, float by, float bz) {
    cLeadX_.store(lx, std::memory_order_relaxed); cLeadY_.store(ly, std::memory_order_relaxed);
    cLeadZ_.store(lz, std::memory_order_relaxed); cBedX_.store(bx, std::memory_order_relaxed);
    cBedY_.store(by, std::memory_order_relaxed);  cBedZ_.store(bz, std::memory_order_relaxed);
  }

 private:
  // ---------- scale ----------
  static constexpr int MAXMODE = 16;
  int   modeBuf_[MAXMODE] = {0, 2, 3, 5, 7, 8, 10};
  int   modeN_ = 7;
  float root_ = 220.f;
  float degHz(int degree) const;
  int   nodeDegree(int node, int cluster) const;
  // LATER tier (SIEVES, Step 1): three Xenakis pitch-sieves (unions of residual classes mod 12),
  // baked once in init() from the real points.bin galaxy-cluster SIZES, read-only forever on the
  // audio thread (benign publish via the ready_ acquire/release). sieveScale_[row][pc]!=0 => that
  // pitch-class is LIT. When cSieve_>0.5 (Acts III-V) degHz() reinterprets the integer `degree` as
  // SEMITONES-mod-12 and snaps to the nearest lit pc of the active row (cClusterId_%3). The remap IS
  // the dramaturgy (the diatonic sweetness curdles into machine-math) -- a future editor must NOT
  // "fix" it back to mode-steps. cSieve_ is a hard gate stepped in update() off the VISUAL act + within-act
  // f ramp per AV-sync (a1) (0 Acts I-II; 1 from III; late + incomplete revert in the looping Act V), NOT off
  // the free-running secT_ wall-clock. sieveScale_ stays decoupled from ParticleField: built from a plain
  // int[3] histogram passed into the init() overload.
  int8_t sieveScale_[3][12] = {};
  std::atomic<float> cSieve_{0.f};
  void  buildSievesFromCounts(const int* clusterCounts, int nClusters);

  // ---------- CC0 sample bank ----------
  struct Sample { std::vector<float> data; float srcSR = 44100.f; float rootHz = 220.f; bool pitched = true; };
  struct Role   { int idx[64]; int n = 0; int rr = 0; };
  std::vector<Sample> samples_;
  Role rBass_, rPluck_, rBell_, rTrace_, rTimp_, rMetal_, rVoice_;   // 2a: rVoice_ = real ghostly-voice/speech WAVs (THEM made literal)
  void loadSamples(const std::string& assetDir);
  int  pickSample(Role& r, float hz);
  void loadWords(const std::string& assetDir);            // real on-theme words for the whisper
  std::vector<std::string> wordbank_;
  uint32_t wprng_ = 0xC0FFEEu;

  // ---------- lock-free event ring (SPSC) ----------
  struct Ev { int kind; float hz, amp, pan; int layer, islot; float a, b, c; int vow = 0; };  // vow = packed vowel sequence
  enum { EV_NOTE = 0, EV_TRACE_ON = 1, EV_TRACE_OFF = 2, EV_WHISPER = 3, EV_RHYTHM = 4, EV_RHYTHM2 = 5, EV_GLITCH = 6 };  // 2c: per-emergence-step master glitch (no voice). EV_LOOP=6 never existed; 6 is the next free value.
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
  std::atomic<float> cGrowl_{0.f};
  // per-element DROPOUT gates: each element fades/cuts in & out independently over ~10 min.
  enum { G_DRONE, G_WHISPER, G_GRAIN, G_KICK, G_BASS, G_MEL, G_BELL, G_TIMP, NGATE };
  std::array<std::atomic<float>, NGATE> cGate_, cGateTau_;   // target + glide time-constant (fade vs sudden)
  std::atomic<float> cFmt0_{500.f}, cFmt1_{1500.f}, cFmt2_{2500.f};   // formants of the last whispered word
  // ---------- AVANT-GARDE controls (sim -> audio) ----------
  static constexpr int PADN = 8;                                      // spectral drone: 8 partials (was 4)
  std::atomic<float> cSpectrum_[PADN];                               // per-partial TARGET ratio (harmonic<->inharmonic)
  std::atomic<float> cBeat_{0.f};                                    // difference-tone detune (audible beating/roughness)
  std::atomic<float> cTopCut_{0.f};                                  // V: fade the upper partials -> detuned ghost
  std::atomic<float> cCloudLam_{6.f}, cCloudSpread_{0.f}, cQuant_{1.f};   // Xenakis grain cloud: rate, gliss span, quantize
  std::atomic<float> cGranDens_{0.f}, cGranSize_{0.06f}, cGranScat_{0.f}, cGranRev_{0.f}, cGranPitch_{0.f};  // whisper granulator
  std::atomic<float> cCut_{1.f}, cFreeze_{0.f};                      // IV TURN: structural CUT to silence + reverb freeze
  // STORY SPINE (THEM: one named worker per phrase) + CC0 sample CHOIR (US: comfort atop their voice)
  std::atomic<float> cStoryDeg_{7.f}, cCostBeat_{0.f};               // worker year->register, wage->difference-tone roughness
  // LOW-END INTEREST: the sub PEDAL (the "bass drone") stops being a static sine locked to the pad —
  // per-act beat-PULSE depth, per-act PRESENCE (extra silencing, decoupled from the pad's G_DRONE), and a
  // per-phrase low NOTE ostinato. Sim-set in update()/the phrase block, audio-read relaxed in render().
  std::atomic<float> cSubPulse_{0.f}, cSubLevel_{1.f}, cSubRatio_{1.f};
  // LATER tier (GENDYN, Step 2): true 2nd-order Xenakis Dynamic Stochastic Synthesis (layer 10). THEM made
  // HARSH -- the worker's difference-tone roughness given a body, alternating with WHISPER via cVoiceSel_ so the
  // THEM foreground is always exactly ONE soloist. cGendyn_ = conductor envelope (orchestration scalar), cGendynHarsh_
  // = wage-driven harshness (== st.costN, the SAME number that feeds cCostBeat_->cBeat_), cVoiceSel_ = phraseIdx_&1
  // mux (0 = whisper's turn, 1 = GENDYN's turn). Sim sets per phrase in update(); audio reads at voice-birth only.
  std::atomic<float> cGendyn_{0.f}, cGendynHarsh_{0.f}; std::atomic<int> cVoiceSel_{0};
  static constexpr int GSEG = 6;                                     // K=6 breakpoint segments (fits pm[6]/pa[6] exactly)
  void  fireGendyn();                                                // schedules a layer-10 EV_NOTE (THEM harm-tone soloist)
  // LATER tier (ARBORESCENCES, Step 3): Metastaseis branching of the layer-2 grain cloud. The moment of
  // fan-out / capture -- a single attended worker explodes into a sheaf of glissando trajectories (one
  // becomes a statistical mass). NO new DSP primitive: a per-sample glide of a layer-2 grain's v.hz toward
  // a stored TARGET (kept in the free v.body slot, sentinel tslot=-9) IS a glissando. cArbor_ = conductor
  // envelope (peaks Act III). Spread reuses the existing cCloudSpread_ atomic (no second knob). Sim sets
  // cArbor_ per buffer; the audio fan emitter (a 2nd Poisson clock, arborTimer_) reads it lock-free.
  std::atomic<float> cArbor_{0.f};
  std::atomic<float> cChoir_{0.f}, cChoirLam_{0.f};                  // washed granular sample-choir amp + density
  std::atomic<float> cSparsity_{0.6f};                              // Eno loop gating: higher = fewer loops survive
  // ---------- PHASE 2 controls (sim -> audio); all read with memory_order_relaxed on the audio thread ----------
  std::atomic<int>   cChorusN_{1};                                  // 2b: ghosts per on-image whisper (1..5); peaks in Act V
  std::atomic<float> cChorusDet_{0.f};                             // 2b: max per-voice detune span (semitones)
  std::atomic<float> cFmtShift_{1.f};                             // 2b: place->throat formant scale
  std::atomic<float> cVoiceRough_{0.f};                           // 2b: wage->strain (widens formant bandwidths)
  std::atomic<float> cGranSmear_{0.f};                            // 2b: extra scatter added on the voice-shred read
  // 2c DREAM-EMERGENCE: cEmerge_ is NOT in Phase 1 (grep-verified) -> declared here. Synced runtime emergence
  // (sim sets it in update() from WoSWState.emergePhase + a conductor envelope). Read INVERSELY in render():
  // shepGain*=(1-emg), cGranScat*=(1-emg), pad-floor*=emg -> crossfade Shepard-noise -> harmonic pad.
  std::atomic<float> cEmerge_{0.f};
  // 2c per-emergence-STEP glitch params (sim sets, audio glitch handler latches): cStutLen_ shrinks +
  // cCrush_ falls as emg->1 so the image resolves out of glitch into clarity.
  std::atomic<float> cStutLen_{0.f}, cCrush_{0.f};
  // LATER tier (GLITCH AXIS, Step 4): EV_GLITCH=6 already exists (Phase-2 owns islot 0: the per-emergence-step
  // master stutter+bitcrush window). This axis owns islot 1-5 (1=bitcrush, 2=stutter, 3=reverse-slam, 4=freeze,
  // 5=micro-gate) and fires GRID-LOCKED on the audio thread (NOT via the ring) so the edits read as a producer's
  // splices, never as failure. Sim sets per-act target floats in update() (keyed off the `act` arg + the within-
  // section `f` ramp, AV-sync decision a1); audio owns the density clock. cGlitchDens_ = P(op|grid hit) envelope;
  // cGateAmt_ = micro-gate depth (GLITCH_GATE); cGlitchOp_ = the per-act stutter probability x1000 (GLITCH_STUT,
  // published as an int to stay within the declare-once set -- a slight repurpose of the "op encoder", documented).
  // cCrush_/cStutLen_ are REUSED from Phase-2 (declared above) -- NOT re-declared. cCrush_ biases the bitcrush
  // depth (harsher when cCostBeat_ is high); cStutLen_ the stutter window. Glitch ducks under the whisper (half
  // density) and the big ops (reverse-slam/freeze) gate on the downbeat (euStep_==0). The cliff = a dense
  // reverse-slam+freeze cluster at the visual III->IV edge (cFreeze_ latched), near-silent through the cut.
  std::atomic<float> cGlitchDens_{0.f}, cGateAmt_{0.f};
  std::atomic<int>   cGlitchOp_{0};

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
  float       dropT_[NGATE] = {}, dropTgt_[NGATE] = {1, 1, 1, 1, 1, 1, 1, 1};   // per-element dropout scheduler
  uint32_t    dprng_ = 0xDEADBEEFu;
  int         lastActEdge_ = 1; float cutHold_ = 0.f;   // detect the III->IV edge; hold the cut ~0.35 s
  // loaded labor stories: each phrase pops one (the THEM spine)
  struct Story { float yearN = 0.5f, costN = 0.5f; int era = 0; };
  std::vector<Story> stories_;
  std::size_t storyIdx_ = 0; float phraseT_ = 0.f; int phraseIdx_ = 0;
  void loadStories(const std::string& assetDir);
  float       actElapsed_ = 0.f; int prevConductAct_ = 1;   // within-section build (evolve, not slam)
  int         lastEmergeStep_ = -1;                         // 2c: sim-thread edge-detect so EV_GLITCH fires once per discrete emergence step
  // ---------- LATER tier: time-authoritative conductor (Step 0; sim-thread only) ----------
  // conduct(dt) accumulates secT_ (the FIBSEC golden-mean clock, cut at index 7 = 555 s), derives the
  // FIBSEC sub-section index -> conductAct_ (advisory), and a within-section ramp conductF_ (0..1). Per the
  // AV-sync decision (a1) the SHIPPED 5-act palette still keys off the `act` arg + the existing `f` ramp;
  // secT_/conductAct_/conductF_ are the within-time clock the LATER techniques (GENDYN/SIEVE/ARBOR/GLITCH)
  // will read. They make NO sound on their own. Read on the sim thread in update(); a renderer may read them
  // via WoSWState later if AV lockstep needs it.
  float       secT_ = 0.f; int conductAct_ = 1; float conductF_ = 0.f;
  void        conduct(float dt);                            // advances secT_; sets conductAct_/conductF_

  // ---------- audio-side voices ----------
  // LATER tier (GENDYN, Step 2) layer-10 ALIAS MAP (no struct growth): layer-10 voices NEVER enter the
  // formant (layer 4) or generic-additive paths, so they freely REUSE existing scratch fields ---
  //   breakpoint durations d[k]   -> pm[k]           (k in [0,GSEG))
  //   breakpoint amplitudes a[k]  -> pa[k]
  //   time-velocities      vT[k]  -> fcoef[k/3][k%3] (9 free slots; GSEG=6 fits)
  //   amp-velocities       vA[k]  -> fz_[k/2][k%2]   (6 free slots; GSEG=6 fits)
  //   segment count               -> syl
  //   fundamental (Hz)            -> hz
  //   walk / read phase           -> phase
  //   harshness h                 -> body
  //   foldback drive (cached)     -> lp
  // Invariant: a layer==10 voice must be dispatched by the layer-10 tickVoice() branch ONLY (placed above the
  // generic additive path); it must never fall through to the formant/additive code that would clobber these.
  struct Voice {
    bool  on = false; int layer = 0;
    float hz = 220, amp = 0, pan = 0, age = 0, life = 1, atk = 0.02f;
    float phase = 0, lp = 0;
    int   tslot = -1;
    const float* smp = nullptr; int smpLen = 0; float smpPos = 0, smpRate = 1.f;
    int   K = 5; float pm[6] = {1,2,3,4,5,6}, pa[6] = {0}; float pnorm = 1.f;
    float fmt[3] = {500, 1500, 2500}, fcoef[3][3] = {}, fz_[3][2] = {};
    int   syl = 2; float body = 0.6f; uint32_t grng = 0x2545F491u;
    int   vowels[5] = {5, 5, 5, 5, 5}; int curSyl = -1;   // word's vowel sequence + current syllable
  };
  static constexpr int NVOX = 160;
  std::array<Voice, NVOX> vox_{};
  int  allocVoice();
  int  allocCapped(int layer, int cap);
  void trigger(const Ev& e);
  float tickVoice(Voice& v, float isr, float depth, float bright);

  // ---------- always-on layers ----------
  // PADN now declared with the avant-garde controls above (8 partials).
  float padHz_[PADN] = {}, padPhase_[PADN] = {}, padVib_[PADN] = {}, padLp_[PADN] = {}, padAmp_[PADN] = {};
  float padTrem_[PADN] = {}, padDrift_[PADN] = {}, padHp_[PADN] = {};   // moving-drone LFOs + high-pass (off the sub-bass)
  float padBeatPh_[PADN] = {}, padRatioCur_[PADN] = {1,2,3,4,5,6,7,8};  // beating LFO phase + glided per-partial ratio
  // spectral granulator capture ring (audio-thread only) + glided cut/freeze
  static constexpr int CAPN = 1 << 16;
  std::array<float, CAPN> cap_{}; int capPos_ = 0; float granTimer_ = 0.f, cut_ = 1.f, freeze_ = 0.f;
  float choirTimer_ = 0.f;                             // Poisson clock for the sample choir
  float arborTimer_ = 0.f;                             // LATER (ARBOR): 2nd Poisson clock for the Metastaseis fan emitter
  // Eno generative loops: 8 coprime Fibonacci-length step-loops that phase against each other (never repeat
  // in 14 min). loopStep_ advances one per beat; loop fires when its step wraps, gated by cSparsity_.
  static constexpr int NLOOP = 8;
  int loopStep_[NLOOP] = {0, 1, 2, 3, 4, 5, 6, 7};     // staggered starts so they don't all hit the downbeat
  void fireLoop(int L);
  float triPhase_ = 0.f, triHz_ = 55.f;                // pure mono triangle SUB-BASS
  float beatPhase_ = 0.f;
  float subPhase_ = 0.f, subAmp_ = 0.f, subLp_ = 0.f, subHz_ = 55.f;
  float subRatioCur_ = 1.f;                            // glided per-phrase ostinato ratio (no zipper between notes)
  static constexpr int SHEP = 6;
  float shepPhase_ = 0.f, shepPh_[SHEP] = {}, shepBp_[SHEP] = {}, shepRate_ = 0.04f, shepRateTgt_ = 0.04f, shepGain_ = 0.f;  // shepPh_/shepBp_ = noise-bandpass SVF states (L)
  float shepPhR_[SHEP] = {}, shepBpR_[SHEP] = {};      // right-channel SVF bank (independent noise -> stereo Shepard)
  float nzLp_ = 0.f, nzLp2_ = 0.f;
  uint32_t euMask_ = 0x49u; int euLen_ = 8, euStep_ = 0; float euPhase_ = 0.f;
  uint32_t euMask2_ = 0x15u; int euLen2_ = 5, euStep2_ = 0; float euPhase2_ = 0.f; int timpDeg_ = 0;
  static constexpr int RIS = 4;
  float risP_ = 0.f, risPh_[RIS] = {};                 // Risset eternal-accelerando tick streams
  float fxTimer_ = 6.f;                                // industrial-clang scheduler
  float grainTimer_ = 0.f; uint32_t rng_ = 0x9E3779B9u; float frand();
  int   lastClusterId_ = -999;
  // 2c emergence + glitch (audio-side, ALL fixed/preallocated — zero alloc on the audio thread):
  float emerge_ = 0.f;                                  // glided cEmerge_ (one-pole ~0.4 s, click-free)
  static constexpr int STUTN = 1 << 12;                 // 4096 frames (~93 ms @ 44.1k) stutter capture
  float stutBufL_[STUTN] = {}, stutBufR_[STUTN] = {}; int stutWr_ = 0;
  int   glitchT_ = 0, glitchLen_ = 0, stutPeriod_ = 0, stutRead_ = 0;
  float crushAmt_ = 0.f, crushHoldL_ = 0.f, crushHoldR_ = 0.f; int crushPhase_ = 0, crushStride_ = 1;
  // LATER tier (GLITCH AXIS, Step 4) audio-owned state (no atomics; lives only on the audio thread). The Step-4
  // CONTINUOUS bitcrush (islot 1) sample-and-holds the master L/R on crushPh_ (re-quantizing at a crush-rate <
  // sr_); crushHoldL_/crushHoldR_ are REUSED from Phase-2's window crush (the two crush domains are partitioned:
  // Step-4 crush is skipped while Phase-2's islot-0 window is active -> no fight). crushBits_ is REPURPOSED as a
  // per-sample ARMED COUNTDOWN (>0 = crushing; re-armed on a bitcrush grid hit, decremented per sample); the live
  // bit-depth is read from cCrush_/cCostBeat_ each hold. microGate_/microGateTgt_ = the glided master micro-gate
  // (islot 5; target re-rolled per grid hit). stutOrigin_ (cap_ window start) + stutPlay_ (window length in
  // samples) drive the stutter/freeze-loop wrap in tickVoice() for tslot==-11 cap_ voices. freezeHold_ = a grid-
  // step countdown that forces the SINGLE existing reverb freeze_ high (NO 2nd reverb-freeze, NO race on cFreeze_).
  // capRms_ = a one-pole RMS of the cap_ (whisper-lead) writes -- the silence guard: stutter/reverse/freeze fire
  // only when capRms_>threshold (crush/micro-gate operate on the full master bus, which always has material).
  float crushPh_ = 0.f, microGate_ = 1.f, microGateTgt_ = 1.f, stutPlay_ = 0.f, capRms_ = 0.f;
  int   stutOrigin_ = 0, freezeHold_ = 0, crushBits_ = 0;
  float kickDuck_ = 1.f;   // sub-bass sidechain vs the kick (voice ducking removed: bed/low no longer dip under voices)
  // arrangement gates (glided) + dubstep growl LFO/filter + ping-pong delay (grains+whispers)
  float growl_ = 0.f, gate_[NGATE] = {};               // smoothed per-element dropout gates (audio)
  float wobPhase_ = 0.f, growlLp_ = 0.f, growlBp_ = 0.f, wobMult_ = 1.f, wobChange_ = 0.f;
  float growlDuty_ = 1.f, growlDutyTgt_ = 1.f, growlOutLp_ = 0.f;   // sometimes off -> clean drone; LP keeps it off the bass
  float growlFmtCoef_[3][3] = {}, growlFz_[3][2] = {}, lastFmt_[3] = {0.f, 0.f, 0.f};  // talking-growl formants
  std::vector<float> ppL_, ppR_; int ppN_ = 0, ppPos_ = 0;
  float panLFO_ = 0.f;
  float lastDepth_ = 1.f, moodLP_ = 1.f, tension_ = 0.f, padBloom_ = 1.f, reverbWet_ = 0.3f;
  float masterLoL_ = 0.f, masterLoR_ = 0.f, masterHiL_ = 0.f, masterHiR_ = 0.f, masterTopL_ = 0.f, masterTopR_ = 0.f;

  // ---- dome spatialization state (Stage 1) ----
  std::atomic<float> cLeadX_{0.f}, cLeadY_{0.f}, cLeadZ_{-4.f};   // THEM focus point (listener-relative dir)
  std::atomic<float> cBedX_{0.f},  cBedY_{0.f},  cBedZ_{-3.f};    // bed anchor (in front of the listener)
  std::unique_ptr<al::Spatializer> spat_;   // Ambisonics (dome) / StereoPanner (forced); NULL => legacy stereo
  al::Speakers speakers_;
  static constexpr int SPATFR = 4096;        // fixed scratch upper bound (>= framesPerBuffer); no RT alloc
  float stemLead_[SPATFR]{}, stemBed_[SPATFR]{}, stemLow_[SPATFR]{}, stemRev_[SPATFR]{}, stemFx_[SPATFR]{};
  bool  spatPrepared_ = false;   // al::Lbap allocs in prepare() -> call it ONCE (not per-buffer, RT)
  float subLfeLp_      = 0.f;     // ~100 Hz one-pole state for the dome SUB (device ch 47)
  float washGain_      = 2.0f;    // dome: boost the decorrelated mains wash (the music) -- env WSW_WASH_GAIN
  float subTrim_       = 0.4f;    // dome: trim the LFE/sub level (it was dominating the mains) -- env WSW_SUB_TRIM
#if defined(WOSW_HAVE_DECORR)
  std::unique_ptr<al::Decorrelation> decorr_;   // Stage 2: diffuse the BED across all speakers (envelopment)
  int decorrN_ = 0;                              // decorrelation output count (= speaker count)
#endif

  al::Reverb<float> reverb_;
  al::Reverb<float> sampRev_;       // short mono reverb to blend the CC0 samples
  float  lastDecay_ = -1.f, lastDamp_ = -1.f;
  double sr_ = 48000.0;   // overwritten by init() from configureAudio; default matches the piece's 48 kHz
  std::atomic<bool> ready_{false};

  void loadManifest(const std::string& assetDir);
  void fireKick();
  void fireTimp();          // pitched timpani on the 2nd Euclidean
  void fireClang();         // industrial metallic FX
  void fireAnd();           // subtle off-beat "and" tick accenting the main beat
  void fireGlitch(float activity, bool whisperOn);   // LATER (GLITCH AXIS): grid-locked edit bus (islot 1-5)
  float jiBassHz() const;
};

}  // namespace wosw
