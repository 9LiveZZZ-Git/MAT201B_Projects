#include "audio/AudioEngine.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dirent.h>

namespace wosw {

static constexpr float kPI  = 3.14159265358979f;
static constexpr float kTAU = 6.28318530717959f;
static const float kJI[8] = {1.f, 16.f / 15.f, 6.f / 5.f, 5.f / 4.f, 4.f / 3.f, 3.f / 2.f, 7.f / 4.f, 2.f};
// Eno generative loops (1d): coprime Fibonacci lengths, pitched layers, degree offsets, amps.
static const int   LOOPLEN[8]   = {3, 5, 8, 13, 21, 34, 55, 89};
static const int   LOOPLAYER[8] = {1, 0, 3, 5, 2, 1, 0, 5};            // bell/pluck/trace/organ/grain
static const int   LOOPDEG[8]   = {0, 7, 4, -3, 9, 12, 2, -5};         // an evolving chord over the worker's register
static const float LOOPAMP[8]   = {0.060f, 0.050f, 0.050f, 0.070f, 0.045f, 0.045f, 0.050f, 0.060f};

// Recognizable PUBLIC-DOMAIN melodies (semitone offsets from tonic, + beat-durations). The top
// grains occasionally coalesce into one of these (a familiar quote inside the machine's texture).
struct Tune { const int* s; const int* d; int n; };
static const int TW_S[] = {0,0,7,7,9,9,7, 5,5,4,4,2,2,0};            // Twinkle Twinkle / Ah vous dirai-je
static const int TW_D[] = {1,1,1,1,1,1,2, 1,1,1,1,1,1,2};
static const int OJ_S[] = {4,4,5,7,7,5,4,2,0,0,2,4,4,2,2};          // Ode to Joy (Beethoven)
static const int OJ_D[] = {1,1,1,1,1,1,1,1,1,1,1,1,1,2,2};
static const int FJ_S[] = {0,2,4,0,0,2,4,0,4,5,7,4,5,7};            // Frere Jacques
static const int FJ_D[] = {1,1,1,1,1,1,1,1,1,1,2,1,1,2};
static const Tune TUNES[3] = {{TW_S, TW_D, 14}, {OJ_S, OJ_D, 15}, {FJ_S, FJ_D, 14}};

// LATER tier — time-authoritative conductor (Step 0). FIBSEC = cumulative section END times (s);
// the golden-mean structural cut is index 7 = 555 s. FIBACT maps each sub-section -> an advisory act.
// secT_ walks FIBSEC to derive conductAct_ + a within-section ramp conductF_ (see AudioEngine::conduct).
static const float FIBSEC[12] = {89.f, 144.f, 233.f, 322.f, 377.f, 466.f, 521.f, 555.f, 576.f, 631.f, 720.f, 843.f};
static const int   FIBACT[12] = {1, 1, 1, 2, 2, 3, 3, 3, 4, 4, 5, 5};

static inline float dn(float x) { return (std::fabs(x) < 1e-15f) ? 0.f : x; }
static inline float clamp01(float x) { return x < 0.f ? 0.f : (x > 1.f ? 1.f : x); }
static inline float gco(float tau, int nf, double sr) { return 1.f - std::exp(-float(nf) / (float(sr) * tau)); }

// ===================== 16-bit PCM WAV loader =====================
static bool loadWav16(const std::string& path, std::vector<float>& out, float& srcSR) {
  FILE* f = std::fopen(path.c_str(), "rb"); if (!f) return false;
  char riff[4]; if (std::fread(riff, 1, 4, f) != 4 || std::memcmp(riff, "RIFF", 4)) { std::fclose(f); return false; }
  std::fseek(f, 4, SEEK_CUR);
  char wave[4]; if (std::fread(wave, 1, 4, f) != 4 || std::memcmp(wave, "WAVE", 4)) { std::fclose(f); return false; }
  int ch = 1, bits = 16; uint32_t rate = 32000; char id[4]; uint32_t sz = 0; bool gotFmt = false, gotData = false;
  while (std::fread(id, 1, 4, f) == 4 && std::fread(&sz, 4, 1, f) == 1) {
    if (!std::memcmp(id, "fmt ", 4)) {
      uint16_t fmt = 1, c = 1, b = 16; uint32_t r = 32000;
      std::fread(&fmt, 2, 1, f); std::fread(&c, 2, 1, f); std::fread(&r, 4, 1, f); std::fseek(f, 6, SEEK_CUR); std::fread(&b, 2, 1, f);
      ch = c; bits = b; rate = r; gotFmt = true; long skip = long(sz) - 16; if (skip > 0 && skip < (1 << 20)) std::fseek(f, skip, SEEK_CUR);
    } else if (!std::memcmp(id, "data", 4)) {
      if (!gotFmt || bits != 16 || ch < 1 || ch > 8 || rate == 0 || sz == 0 || sz > 200u * 1024u * 1024u) { std::fclose(f); return false; }
      long n = long(sz) / 2; std::vector<int16_t> raw(n);
      if (long(std::fread(raw.data(), 2, n, f)) != n) n = 0;
      int frames = (ch > 0) ? int(n / ch) : 0; out.resize(frames);
      for (int i = 0; i < frames; ++i) { int acc = 0; for (int c = 0; c < ch; ++c) acc += raw[i * ch + c]; out[i] = float(acc) / (ch * 32768.f); }
      srcSR = float(rate); gotData = true; break;
    } else std::fseek(f, int(sz) + (int(sz) & 1), SEEK_CUR);
  }
  std::fclose(f); return gotData && !out.empty();
}
static bool parseNoteHz(const std::string& name, float& hz) {
  static const int semi[7] = {9, 11, 0, 2, 4, 5, 7};
  for (size_t i = 0; i + 1 < name.size(); ++i) {
    char c = char(std::toupper((unsigned char)name[i])); if (c < 'A' || c > 'G') continue;
    if (i > 0 && std::isalpha((unsigned char)name[i - 1])) continue;
    size_t j = i + 1; int sharp = 0; if (name[j] == '#' || name[j] == 's') { sharp = 1; ++j; }
    if (j >= name.size() || !std::isdigit((unsigned char)name[j])) continue;
    int oct = name[j] - '0', midi = (oct + 1) * 12 + semi[c - 'A'] + sharp; hz = 440.f * std::pow(2.f, (midi - 69) / 12.f); return true;
  }
  return false;
}
void AudioEngine::loadSamples(const std::string& assetDir) {
  const std::string bases[] = { assetDir + "/audio", "assets/audio", "../../assets/audio", "reagency/assets/audio", "MAT201B_Projects/reagency/assets/audio" };
  auto add = [](Role& r, int idx) { if (r.n < 64) r.idx[r.n++] = idx; };
  for (const auto& b : bases) {
    DIR* d = opendir(b.c_str()); if (!d) continue;
    struct dirent* e;
    while ((e = readdir(d)) != nullptr) {
      std::string fn = e->d_name; if (fn.size() < 5) continue;
      std::string lc = fn; for (auto& ch : lc) ch = char(std::tolower((unsigned char)ch));
      if (lc.substr(lc.size() - 4) != ".wav") continue;
      Sample s; if (!loadWav16(b + "/" + fn, s.data, s.srcSR)) continue;
      auto has = [&](const char* k) { return lc.find(k) != std::string::npos; };
      s.pitched = parseNoteHz(fn, s.rootHz);
      if (has("agogo") || has("bowl") || has("chime") || has("crotale") || has("tubular")) s.pitched = false;
      int idx = int(samples_.size());
      if (has("timpani")) { s.pitched = true; s.rootHz = has("timpani2") ? 98.f : 73.f; samples_.push_back(std::move(s)); add(rTimp_, idx); continue; }
      samples_.push_back(std::move(s));
      if (has("cymbal") || has("crash") || has("tam") || has("/gong") || has("metal") || has("anvil") || has("brake")) add(rMetal_, idx);
      else if (has("organ") || has("pedal") || has("tuba") || has("bassoon") || has("contrabass") || has("trombone")) add(rBass_, idx);
      else if (has("gong") || has("/bell") || has("_bell")) add(rBell_, idx);
      else if (has("speech") || has("ghost") || (has("voice") && !has("voicing"))) { samples_.back().pitched = false; add(rVoice_, idx); }   // 2a: real ghostly voices (THEM) -> NON-pitched
      else if (has("psaltery") || has("bowed") || has("cello") || has("violin") || has("viola") || has("choir") || has("clarinet")) add(rTrace_, idx);
      else add(rPluck_, idx);
    }
    closedir(d);
    if (!samples_.empty()) { std::fprintf(stderr, "[wosw audio] %zu CC0 samples from %s/ (bass=%d pluck=%d bell=%d trace=%d timp=%d metal=%d voice=%d)\n",
                                          samples_.size(), b.c_str(), rBass_.n, rPluck_.n, rBell_.n, rTrace_.n, rTimp_.n, rMetal_.n, rVoice_.n); return; }
  }
  std::fprintf(stderr, "[wosw audio] no assets/audio samples — fully procedural synthesis\n");
}
int AudioEngine::pickSample(Role& r, float hz) {
  if (r.n == 0) return -1;
  float best = 1e9f;
  for (int i = 0; i < r.n; ++i) { const Sample& s = samples_[r.idx[i]]; float dd = s.pitched ? std::fabs(std::log2(s.rootHz / hz)) : 0.f; if (dd < best) best = dd; }
  int cand[64], nc = 0;
  for (int i = 0; i < r.n; ++i) { const Sample& s = samples_[r.idx[i]]; float dd = s.pitched ? std::fabs(std::log2(s.rootHz / hz)) : 0.f; if (dd <= best + 0.34f && nc < 64) cand[nc++] = r.idx[i]; }
  if (nc == 0) return r.idx[0];
  return cand[(r.rr++) % nc];
}

// ===================== manifest =====================
void AudioEngine::loadManifest(const std::string& assetDir) {
  const std::string bases[] = { assetDir, "assets", "../../assets", "reagency/assets", "MAT201B_Projects/reagency/assets" };
  for (const auto& b : bases) {
    FILE* f = std::fopen((b + "/manifest.json").c_str(), "rb"); if (!f) continue;
    std::string s; char buf[4096]; size_t n; while ((n = std::fread(buf, 1, sizeof(buf), f)) > 0) s.append(buf, n); std::fclose(f);
    auto pos = s.find("\"root_hz\""); if (pos != std::string::npos) { double v = std::atof(s.c_str() + s.find(':', pos) + 1); if (v > 20.0 && v < 4000.0) root_ = float(v); }
    pos = s.find("\"mode\"");
    if (pos != std::string::npos) { size_t lb = s.find('[', pos), rb = s.find(']', lb);
      if (lb != std::string::npos && rb != std::string::npos) { int m[MAXMODE], cnt = 0; const char* p = s.c_str() + lb + 1; const char* end = s.c_str() + rb;
        while (p < end && cnt < MAXMODE) { while (p < end && (*p == ' ' || *p == ',')) ++p; if (p >= end) break; m[cnt++] = std::atoi(p); while (p < end && *p != ',') ++p; }
        if (cnt >= 3) { modeN_ = cnt; for (int i = 0; i < cnt; ++i) modeBuf_[i] = m[i]; } } }
    std::fprintf(stderr, "[wosw audio] scale: root=%.1f Hz, %d degrees from %s/manifest.json\n", root_, modeN_, b.c_str()); return;
  }
  std::fprintf(stderr, "[wosw audio] manifest.json not found — default A minor (root 220)\n");
}
// LATER tier (SIEVES): when cSieve_>0.5 (Acts III-V) the integer `degree` is REINTERPRETED as
// semitones-mod-12 and snapped to the nearest LIT pitch-class of the active sieve row (cClusterId_%3,
// nearest-lit search outward up to +-6). This is INTENTIONAL: the same gestures land on the alien
// galaxy-derived lattice instead of the diatonic mode -- the sweetness curdles. Do NOT "fix" the
// semitone reinterpretation; the remap IS the dramaturgy. Otherwise the byte-for-byte UNCHANGED
// diatonic modeBuf_ path. Called on BOTH threads; the atomic load + immutable sieveScale_ is the
// correct race-free discipline (sieveScale_ published once via ready_ before any audio runs).
float AudioEngine::degHz(int degree) const {
  if (cSieve_.load(std::memory_order_relaxed) > 0.5f) {
    int oct = degree / 12, pc = degree % 12; if (pc < 0) { pc += 12; oct -= 1; }   // semitone reinterpretation
    int sc = int(cClusterId_.load(std::memory_order_relaxed)) % 3; if (sc < 0) sc = 0;   // cluster -1 -> row 0
    if (!sieveScale_[sc][pc]) {                                                   // not lit -> nearest-lit +-6
      int best = pc;
      for (int d = 1; d <= 6; ++d) {
        int up = ((pc + d) % 12 + 12) % 12, dn2 = ((pc - d) % 12 + 12) % 12;
        if (sieveScale_[sc][up]) { best = pc + d; break; }                        // tie -> prefer upward
        if (sieveScale_[sc][dn2]) { best = pc - d; break; }
      }
      pc = best;
    }
    return root_ * std::pow(2.f, (pc + 12 * oct) / 12.f);
  }
  int idx = degree % modeN_, oct = degree / modeN_; if (idx < 0) { idx += modeN_; oct -= 1; } return root_ * std::pow(2.f, (modeBuf_[idx] + 12 * oct) / 12.f);
}
// LATER tier (SIEVES) NOTE: nodeDegree returns a mode-STEP index. Under cSieve_ it is reinterpreted by
// degHz() as a semitone class (see degHz). That semitone remap is the intended dramaturgy, not a bug.
int AudioEngine::nodeDegree(int node, int cluster) const { unsigned h = unsigned(node) * 2654435761u; int d = int(h % unsigned(modeN_ * 3)); if (cluster > 0 && (cluster % 2 == 1)) d += modeN_; return d; }
float AudioEngine::jiBassHz() const { return cClusterRoot_.load(std::memory_order_relaxed) * 0.5f; }
// LATER tier Step 0 — make the conductor time-authoritative. Accumulates secT_ (FIBSEC golden-mean clock),
// derives the current FIBSEC sub-section -> conductAct_ (advisory) + a within-section ramp conductF_ (0..1).
// Pure sim-thread bookkeeping: adds NO sound. The shipped 5-act palette still keys off the `act` arg + the
// existing `f` ramp (AV-sync decision a1); these members are the within-time clock the LATER techniques read.
void AudioEngine::conduct(float dt) {
  secT_ += dt;
  int s = 0; while (s < 11 && secT_ >= FIBSEC[s]) ++s;        // current FIBSEC sub-section index [0..11]
  conductAct_ = FIBACT[s];                                    // advisory act for this sub-section
  float segStart = (s > 0) ? FIBSEC[s - 1] : 0.f;             // this sub-section's start time
  float segLen = FIBSEC[s] - segStart;
  conductF_ = (segLen > 1e-4f) ? clamp01((secT_ - segStart) / segLen) : 1.f;   // within-section ramp 0..1
  // NOTE: the SIEVE gate is NOT stepped here anymore. secT_ is a free-running, never-wrapping wall-clock;
  // keying the gate off it desynced the alien lattice from the LOOPING visual piece (the sieve never
  // returned on later loops, drifted under hesitation, and ignored the 1-5 act-audition jumps which move
  // only the visual clock). Per AV-sync (a1) the gate now lives in update(), keyed off the visual `act`
  // arg + within-act `f` ramp -- exactly like every other LATER technique (GENDYN/ARBOR/GLITCH). See cSieve_.
}

bool AudioEngine::push(const Ev& e) { uint32_t h = rHead_.load(std::memory_order_relaxed), nx = (h + 1) % RING; if (nx == rTail_.load(std::memory_order_acquire)) return false; ring_[h] = e; rHead_.store(nx, std::memory_order_release); return true; }
bool AudioEngine::pop(Ev& e) { uint32_t t = rTail_.load(std::memory_order_relaxed); if (t == rHead_.load(std::memory_order_acquire)) return false; e = ring_[t]; rTail_.store((t + 1) % RING, std::memory_order_release); return true; }

void AudioEngine::loadWords(const std::string& assetDir) {
  const std::string bases[] = { assetDir, "assets", "../../assets", "reagency/assets", "MAT201B_Projects/reagency/assets" };
  for (const auto& b : bases) {
    FILE* f = std::fopen((b + "/words.txt").c_str(), "r"); if (!f) continue;
    char line[256];
    while (std::fgets(line, sizeof(line), f)) { std::string w(line); while (!w.empty() && (w.back() == '\n' || w.back() == '\r' || w.back() == ' ')) w.pop_back(); if (w.size() >= 2) wordbank_.push_back(w); }
    std::fclose(f);
    if (!wordbank_.empty()) { std::fprintf(stderr, "[wosw audio] %zu whisper words from %s/words.txt\n", wordbank_.size(), b.c_str()); return; }
  }
  std::fprintf(stderr, "[wosw audio] words.txt not found — whisper uses per-node pseudo-vowels\n");
}

void AudioEngine::loadStories(const std::string& assetDir) {
  // The 142 verified labor stories (THEM). Pretty-printed JSON -> parse line-by-line, one record per
  // "oneLiner". Map year->register, the $ wage->dissonance, era->timbre family. No deps, sim-thread only.
  const std::string bases[] = { assetDir + "/../v2", "../../v2", "reagency/v2", "MAT201B_Projects/reagency/v2" };
  for (const auto& b : bases) {
    FILE* f = std::fopen((b + "/stories.json").c_str(), "r"); if (!f) continue;
    char line[1024]; Story cur; bool have = false;
    auto flush = [&]() { if (have) stories_.push_back(cur); };
    while (std::fgets(line, sizeof(line), f)) {
      if (std::strstr(line, "\"oneLiner\"")) { flush(); cur = Story(); have = true; }
      if (const char* y = std::strstr(line, "\"year\"")) { const char* c = std::strchr(y, ':');
        if (c) { while (*c && (*c < '0' || *c > '9')) ++c; int yr = std::atoi(c); if (yr > 1000) cur.yearN = clamp01((float(yr) - 1770.f) / 255.f); } }
      if (const char* g = std::strstr(line, "\"figure\"")) { const char* d = std::strchr(g, '$');
        if (d) { double v = std::atof(d + 1); if (v > 0.0) cur.costN = clamp01(1.f - std::log10(std::max(0.5, v)) / 5.4f); } }
      if (std::strstr(line, "\"era\"")) { if (std::strstr(line, "historical")) cur.era = 1; else if (std::strstr(line, "contemporary")) cur.era = 0; }
    }
    flush(); std::fclose(f);
    if (!stories_.empty()) { std::fprintf(stderr, "[wosw audio] %zu labor stories (THEM spine) from %s/stories.json\n", stories_.size(), b.c_str()); return; }
  }
  std::fprintf(stderr, "[wosw audio] stories.json not found — THEM spine inert\n");
}

// LATER tier (SIEVES, Step 1) — bake the three pitch-sieves from the galaxy-cluster SIZE histogram.
// Deterministic + reproducible from sizes (so identical on every dome node; pure function of points.bin).
// For each cluster i of size s: M1=3+(s%5), R1=s%M1, M2=2+((s/7)%4), R2=(s/13)%M2; a pc is LIT when
// pc%M1==R1 || pc%M2==R2. If <4 pcs lit, force {0,7,3,10} (tonic/fifth/minor-third/minor-seventh anchor)
// so a row never degenerates. Validate 4-7 notes per row + print them. Runs in init() BEFORE ready_
// release (sim/main thread); sieveScale_ is read-only on the audio thread thereafter.
void AudioEngine::buildSievesFromCounts(const int* clusterCounts, int nClusters) {
  static const int kFallback[3] = {30000, 8000, 6000};   // procedural-galaxy fallback (no points.bin)
  for (int i = 0; i < 3; ++i) {
    int s = (clusterCounts && i < nClusters && clusterCounts[i] > 0) ? clusterCounts[i] : kFallback[i];
    int M1 = 3 + (s % 5), R1 = s % M1, M2 = 2 + ((s / 7) % 4), R2 = (s / 13) % M2;
    int lit = 0;
    for (int pc = 0; pc < 12; ++pc) {
      sieveScale_[i][pc] = (pc % M1 == R1 || pc % M2 == R2) ? int8_t(1) : int8_t(0);
      lit += sieveScale_[i][pc] ? 1 : 0;
    }
    if (lit < 4) {                                        // degenerate -> force the tonal anchor
      for (int pc = 0; pc < 12; ++pc) sieveScale_[i][pc] = 0;
      const int anchor[4] = {0, 7, 3, 10};
      for (int k = 0; k < 4; ++k) sieveScale_[i][anchor[k]] = 1;
      lit = 4;
    }
    char buf[64]; int bn = 0;
    for (int pc = 0; pc < 12; ++pc) if (sieveScale_[i][pc]) bn += std::snprintf(buf + bn, sizeof(buf) - bn, "%d ", pc);
    std::fprintf(stderr, "[wosw audio] sieve row %d (size=%d, M1=%d R1=%d M2=%d R2=%d): %d notes { %s}%s\n",
                 i, s, M1, R1, M2, R2, lit, buf, (lit < 4 || lit > 7) ? "  <-- WARN out of 4..7" : "");
  }
}

void AudioEngine::init(const std::string& assetDir, double sampleRate) {
  init(assetDir, sampleRate, nullptr, 0);   // delegate; null counts -> sieve fallback triple
}

void AudioEngine::init(const std::string& assetDir, double sampleRate, const int* clusterCounts, int nClusters) {
  sr_ = sampleRate > 0 ? sampleRate : 44100.0;
  loadManifest(assetDir); loadSamples(assetDir); loadWords(assetDir); loadStories(assetDir);
  buildSievesFromCounts(clusterCounts, nClusters);   // LATER (SIEVES): bake BEFORE ready_ release
  reverb_.bandwidth(0.9f); reverb_.damping(0.45f); reverb_.decay(0.9f);
  sampRev_.bandwidth(0.9f); sampRev_.damping(0.5f); sampRev_.decay(0.62f);   // room for the samples (longer)
  cClusterRoot_.store(root_, std::memory_order_relaxed); subHz_ = root_ * 0.25f;
  for (int p = 0; p < PADN; ++p) { cSpectrum_[p].store(float(p + 1)); padRatioCur_[p] = float(p + 1); padHz_[p] = root_ * float(p + 1); }
  ppN_ = int(sr_ * 0.55); if (ppN_ < 64) ppN_ = 64;    // ping-pong delay buffers (allocated once)
  ppL_.assign(ppN_, 0.f); ppR_.assign(ppN_, 0.f); ppPos_ = 0;
  for (int i = 0; i < NGATE; ++i) { cGate_[i].store(1.f); cGateTau_[i].store(0.5f); gate_[i] = 1.f; }
  ready_.store(true, std::memory_order_release);
}

static void formantFor(char vowel, float& f0, float& f1, float& f2) {
  switch (std::tolower((unsigned char)vowel)) {
    case 'a': f0 = 800; f1 = 1150; f2 = 2900; break; case 'e': f0 = 500; f1 = 1800; f2 = 2500; break;
    case 'i': f0 = 320; f1 = 2300; f2 = 3000; break; case 'o': f0 = 450; f1 = 850; f2 = 2800; break;
    case 'u': f0 = 325; f1 = 700; f2 = 2600; break; default: f0 = 500; f1 = 1500; f2 = 2500; break; }
}
static int vowelIndex(char c) { switch (std::tolower((unsigned char)c)) { case 'a': return 0; case 'e': return 1; case 'i': return 2; case 'o': return 3; case 'u': return 4; default: return -1; } }
static void formantForIdx(int i, float& f0, float& f1, float& f2) { static const char v[6] = {'a', 'e', 'i', 'o', 'u', 'x'}; formantFor(v[(i < 0 || i > 5) ? 5 : i], f0, f1, f2); }
static uint32_t euclidMask(int k, int n) { if (n <= 0) return 1u; if (k < 1) k = 1; if (k > n) k = n; uint32_t mask = 0; int prev = -1; for (int i = 0; i < n; ++i) { int cur = (i * k) / n; if (cur != prev) { mask |= 1u << i; prev = cur; } } return mask; }
static float slotRate(float depth, float hes, float act) { float r = (1.6f + 1.3f * depth) * (1.f - 0.28f * hes) * (1.f + 0.20f * act); return r < 0.5f ? 0.5f : (r > 4.5f ? 4.5f : r); }

// ===================== sim-thread API =====================
void AudioEngine::onArrival(int node, int type, float density, int cluster) {
  if (!ready()) return;
  curNode_ = node; clusterCur_ = cluster; simDensity_ = density < 0 ? 0.5f : density;
  ++visits_; cVisits_.store(visits_, std::memory_order_relaxed); cClusterId_.store(cluster, std::memory_order_relaxed);
  int croot = (cluster >= 0) ? (cluster % modeN_) : 0; cClusterRoot_.store(degHz(croot + modeN_), std::memory_order_relaxed);
  activity_ = std::min(1.f, activity_ + 0.5f);
  Ev e{}; e.kind = EV_NOTE; e.layer = (type == 1) ? 1 : 0; e.hz = degHz(nodeDegree(node, cluster)); e.amp = 0.10f + 0.10f * simDensity_; e.pan = cFocusPan_.load(std::memory_order_relaxed); push(e);
  melDeg_ = nodeDegree(node, cluster);
}
void AudioEngine::igniteArp(const std::vector<std::pair<int, float>>& nbrs, int cluster) {
  if (!ready()) return;
  mprng_ = unsigned(curNode_) * 2654435761u ^ unsigned(visits_) * 40503u;
  mel_.clear(); melIdx_ = 0; melTimer_ = 0.f; melPan_ = cFocusPan_.load(std::memory_order_relaxed);
  const float hes = simHes_, depth = simDepth_; const int octShift = (depth > 0.6f) ? modeN_ : (depth < 0.35f ? -modeN_ : 0);
  const int L = 3 + int(4.f * simDensity_); lastLeap_ = false;
  int gravDeg = melDeg_; float bestW = -1.f; for (const auto& nb : nbrs) if (nb.second > bestW) { bestW = nb.second; gravDeg = nodeDegree(nb.first, cluster); }
  static const int dStep[13] = {-7, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 7};
  auto frnd = [&]() { mprng_ ^= mprng_ << 13; mprng_ ^= mprng_ >> 17; mprng_ ^= mprng_ << 5; return float(mprng_) / 4294967296.f; };
  for (int step = 0; step < L; ++step) {
    float w[13], sum = 0.f;
    for (int i = 0; i < 13; ++i) { int dd = dStep[i]; float ww = std::exp(-std::fabs(float(dd)) / 3.f); ww += hes * 0.6f * (std::abs(dd) >= 4 ? 1.f : 0.f);
      if ((melDeg_ + dd) % modeN_ == gravDeg % modeN_) ww += 0.8f; if (lastLeap_ && std::abs(dd) > 2) ww = 0.f; w[i] = ww; sum += ww; }
    float r = frnd() * sum, acc = 0.f; int pick = 6; for (int i = 0; i < 13; ++i) { acc += w[i]; if (r <= acc) { pick = i; break; } }
    int dd = dStep[pick]; lastLeap_ = std::abs(dd) > 2; melDeg_ += dd; melDeg_ = std::max(-modeN_, std::min(modeN_ * 3, melDeg_));
    float edgeW = nbrs.empty() ? 0.5f : nbrs[step % nbrs.size()].second; bool accent = (step == 0) || (melDeg_ % modeN_ == gravDeg % modeN_); int layer = (edgeW > 0.80f) ? 1 : 0;
    mel_.push_back({degHz(melDeg_ + octShift), accent ? (layer + 10) : layer});
  }
}
void AudioEngine::traceOn(int slot, int node, float pan) { if (!ready()) return; Ev e{}; e.kind = EV_TRACE_ON; e.layer = 3; e.hz = degHz(nodeDegree(node, 0)) * 0.5f; e.amp = 0.12f; e.pan = pan; e.islot = slot; push(e); }
void AudioEngine::traceOff(int slot) { if (ready()) { Ev e{}; e.kind = EV_TRACE_OFF; e.islot = slot; push(e); } }
void AudioEngine::whisper(const std::string& word, int node, float pan, float detune, float spreadFrac) {
  if (!ready()) return;
  std::string w = word;                                       // a real word — given per-image label, or picked
  if (w.empty() && !wordbank_.empty()) { wprng_ ^= wprng_ << 13; wprng_ ^= wprng_ >> 17; wprng_ ^= wprng_ << 5; w = wordbank_[wprng_ % wordbank_.size()]; }
  int vidx[5], nv = 0, firstV = -1;
  for (char c : w) { int vi = vowelIndex(c); if (vi >= 0) { if (firstV < 0) firstV = vi; if (nv < 5) vidx[nv++] = vi; } }
  if (nv == 0) { int v0 = (node >= 0 ? node : 0) % 5; nv = (node >= 0 ? node : 0) % 3 + 1; for (int i = 0; i < nv; ++i) vidx[i] = v0; firstV = v0; }
  int syl = nv < 1 ? 1 : (nv > 5 ? 5 : nv), vow = 0;
  for (int i = 0; i < syl; ++i) vow |= (vidx[i] & 7) << (3 * i);
  float f0, f1, f2; formantForIdx(firstV, f0, f1, f2);
  cFmt0_.store(f0, std::memory_order_relaxed); cFmt1_.store(f1, std::memory_order_relaxed); cFmt2_.store(f2, std::memory_order_relaxed);  // -> talking growl
  // 2a: if real-voice WAVs are loaded, push NEGATIVE islot (=-syl) so trigger() loads a real-speech sample on layer 4
  //      (shares the duck + cap_ capture + granulator). Empty bucket -> positive syl -> synth whisper (graceful fallback).
  int islot = (rVoice_.n > 0) ? -syl : syl;
  // 2b: e.a now carries the per-voice CHORUS DETUNE ratio (1=center); pan already includes the chorus spread.
  Ev e{}; e.kind = EV_WHISPER; e.layer = 4; e.amp = 0.15f; e.pan = pan; e.islot = islot; e.vow = vow; e.a = (detune > 0.01f ? detune : 1.f); e.b = f1; e.c = f2; push(e);   // -10 dB
  (void)spreadFrac;   // pan-spread is applied by the caller (whisperChorus) before this; param kept for symmetry/clarity
}
void AudioEngine::whisperChorus(const std::string& word, int node, float pan, int n, float spread) {
  if (!ready()) return;
  if (n <= 0) n = cChorusN_.load(std::memory_order_relaxed);   // n<=0 -> conductor-driven count
  if (n < 1) n = 1; if (n > 5) n = 5;
  const float det = cChorusDet_.load(std::memory_order_relaxed);   // semitone span (set by the conductor in update)
  for (int i = 0; i < n; ++i) {
    float frac = (n > 1) ? (float(i) / float(n - 1) * 2.f - 1.f) : 0.f;        // -1..1 across the chorus
    float ratio = std::pow(2.f, frac * det / 12.f);                            // per-voice detune ratio
    whisper(word, node, pan + spread * frac, ratio, frac);                     // pan-spread + detune passed DIRECTLY (no ring patch)
  }
}
void AudioEngine::update(float dt, float hesitation, float depth, float progress, float focusPan, int act,
                        float emergePhase, bool emergeActive) {
  if (!ready()) return;
  conduct(dt);   // LATER tier Step 0: advance the time-authoritative conductor FIRST (secT_/conductAct_/conductF_)
  depth = clamp01(depth);
  simHes_ = hesitation; simDepth_ = depth;
  cHesitation_.store(hesitation, std::memory_order_relaxed); cDepth_.store(depth, std::memory_order_relaxed);
  cFocusPan_.store(focusPan < -1 ? -1 : (focusPan > 1 ? 1 : focusPan), std::memory_order_relaxed);
  activity_ *= std::pow(0.5f, dt / 0.8f); cActivity_.store(activity_, std::memory_order_relaxed); (void)progress;

  // ---- 5-ACT ORCHESTRATION: the act sets the palette; the dropout scheduler textures WITHIN it ----
  const int a = (act >= 1 && act <= 5) ? act : 1;
  static const float ACTG[6][NGATE] = {                       // [G_DRONE,WHISPER,GRAIN,KICK,BASS,MEL,BELL,TIMP]
    {1, 1, 1, 1, 1, 1, 1, 1},                                 //   fallback
    {1.00f, 0.40f, 0.30f, 0.40f, 1.00f, 1.00f, 1.00f, 0.40f}, // I   SEDUCTION: lush bed + the tune
    {1.00f, 1.00f, 0.40f, 0.50f, 1.00f, 0.30f, 0.70f, 0.50f}, // II  READING: the whisper leads
    {0.70f, 0.85f, 1.00f, 1.00f, 1.00f, 0.00f, 0.20f, 1.00f}, // III EXTRACTION: grinding, dense, the cede
    {0.00f, 1.00f, 1.00f, 1.00f, 0.30f, 0.00f, 0.00f, 0.30f}, // IV  TURN: STRIP to bare pulse (kick+grain+whisper)
    {0.50f, 0.50f, 0.60f, 0.40f, 0.70f, 0.30f, 0.40f, 0.40f}, // V   INTEGRATION: thinned, haunted (comfort-FAIL)
  };
  static const float ACTM[6][3] = { {0.5f, 0.5f, 0.4f},       // mood: bright, density, wet
    {0.70f, 0.40f, 0.55f}, {0.55f, 0.45f, 0.50f}, {0.35f, 0.85f, 0.35f}, {0.30f, 0.50f, 0.30f}, {0.45f, 0.35f, 0.45f} };
  const float* aGate = ACTG[a];
  section_ = a;
  // LOW-END INTEREST (per-act): the sub PEDAL gets its own beat-PULSE depth + PRESENCE so it diverges
  // from the sustained pad (both rode G_DRONE before -> too similar). I/II breathe gently, III grinds,
  // IV is gone (like G_DRONE), V is sparse + slow.       fb     I      II     III    IV     V
  static const float SUBPULSE[6] = { 0.f, 0.25f, 0.40f, 0.85f, 0.00f, 0.55f };   // beat-pulse depth (0 steady..1 gated between beats)
  static const float SUBLEVEL[6] = { 1.f, 0.70f, 0.55f, 1.00f, 0.00f, 0.45f };   // sub-pedal presence (decoupled from the pad)
  cSubPulse_.store(SUBPULSE[a], std::memory_order_relaxed);
  cSubLevel_.store(SUBLEVEL[a], std::memory_order_relaxed);
  mbTgt_ = ACTM[a][0]; mdTgt_ = ACTM[a][1]; mwTgt_ = ACTM[a][2];
  cGrowl_.store(a == 3 ? 1.f : (a == 4 ? 0.3f : 0.f), std::memory_order_relaxed);   // the grind in extraction
  if (a == 1) { if (!melodyOn_) { melodyOn_ = true; melNote_ = 0; melTune_ = (melTune_ + 1) % 3; melNoteTimer_ = 0.f; } }
  else melodyOn_ = false;                                     // the tune is the SEDUCTION only
  float mc = std::min(1.f, dt * 0.25f); moodBright_ += (mbTgt_ - moodBright_) * mc; moodDens_ += (mdTgt_ - moodDens_) * mc; moodWet_ += (mwTgt_ - moodWet_) * mc;
  cMoodBright_.store(moodBright_, std::memory_order_relaxed); cMoodDens_.store(moodDens_, std::memory_order_relaxed); cMoodWet_.store(moodWet_, std::memory_order_relaxed);

  // ---- AVANT-GARDE per-act dispatch: living spectrum, Xenakis cloud, voice granulator, the TURN cut ----
  // 1a within-section BUILD: f ramps 0->1 across the section so sections EVOLVE rather than slam.
  if (a != prevConductAct_) { actElapsed_ = 0.f; prevConductAct_ = a; }
  actElapsed_ += dt;
  static const float SEGDUR[6] = {200.f, 210.f, 150.f, 150.f, 90.f, 240.f};
  const float f = clamp01(actElapsed_ / SEGDUR[a]);
  static const float SPARS[6] = {0.6f, 0.55f, 0.50f, 0.10f, 0.85f, 0.70f};   // higher -> fewer Eno loops survive
  cSparsity_.store(SPARS[a], std::memory_order_relaxed);
  static const float ACT_INH[6]    = {0.f, 0.030f, 0.080f, 0.55f,  0.22f, 0.15f};   // inharmonic stretch (audible from act I)
  static const float ACT_BEAT[6]   = {0.f, 0.022f, 0.032f, 0.060f, 0.045f, 0.035f}; // difference-tone beating (clearly rough)
  static const float ACT_TOPCUT[6] = {0.f, 0.f,    0.f,    0.f,    0.f,   1.f};      // V: thin the upper partials -> ghost
  cBeat_.store(ACT_BEAT[a] * (0.6f + 0.4f * f) + 0.04f * cCostBeat_.load(std::memory_order_relaxed), std::memory_order_relaxed);  // builds in; cheap wage -> rougher
  cTopCut_.store(ACT_TOPCUT[a], std::memory_order_relaxed);
  for (int p = 0; p < PADN; ++p) {
    float stretch = float(p + 1) * std::pow(1.f + ACT_INH[a], float(p));            // upper partials drift sharp
    cSpectrum_[p].store(stretch + 0.012f * (kJI[(p * 3) & 7] - 1.f) * (a == 3 ? 3.f : 1.f), std::memory_order_relaxed);
  }
  static const float CL[6] = {6, 6, 4, 40, 2, 7};                                   // grain cloud rate per act
  static const float SP[6] = {0, 0.28f, 0.42f, 0.90f, 0.f, 0.55f};                  // glissando span (audible gliss from act I)
  static const float QU[6] = {1, 1.f, 0.80f, 0.15f, 1.f, 0.60f};                    // P(quantized) vs continuous gliss
  cCloudLam_.store(CL[a] * (0.5f + 0.5f * f), std::memory_order_relaxed); cCloudSpread_.store(SP[a], std::memory_order_relaxed); cQuant_.store(QU[a], std::memory_order_relaxed);   // swarm builds across the section
  // LATER (ARBORESCENCES, Step 3): conductor envelope for the Metastaseis fan-out. AV-sync (a1) -> key off the
  // `act` arg + the within-section `f` ramp. Peaks in Act III (EXTRACT: one worker -> statistical mass), builds
  // within-section via f; hairline branches in II, ghost branches in V. Spread reuses cCloudSpread_ (SP[a]).
  static const float ARB[6] = {0.f, 0.f, 0.15f, 1.0f, 0.10f, 0.20f};
  cArbor_.store(ARB[a] * (0.5f + 0.5f * f), std::memory_order_relaxed);
  static const float GD[6]  = {0, 4.f, 14.f, 90.f, 24.f, 6.f};                       // granulator density (a grain halo even in act I)
  static const float GS[6]  = {0.06f, 0.06f, 0.09f, 0.018f, 0.03f, 0.11f};           // grain length (s)
  static const float GSC[6] = {0, 0.f, 0.f, 0.85f, 0.30f, 1.0f};                     // scatter into the past
  static const float GRV[6] = {0, 0.f, 0.f, 0.40f, 0.f, 0.50f};                      // P(reverse)
  static const float GPT[6] = {0, 0.f, 0.f, 7.f, 0.f, 0.f};                          // pitch-shatter +-semis
  cGranDens_.store(GD[a] * (0.6f + 0.8f * moodDens_), std::memory_order_relaxed);
  cGranSize_.store(GS[a], std::memory_order_relaxed); cGranScat_.store(GSC[a], std::memory_order_relaxed);
  cGranRev_.store(GRV[a], std::memory_order_relaxed); cGranPitch_.store(GPT[a], std::memory_order_relaxed);
  // IV TURN: latch a structural CUT to silence + reverb freeze on the III->IV edge, hold ~0.35 s
  if (a == 4 && lastActEdge_ != 4) cutHold_ = 0.35f;
  lastActEdge_ = a;
  if (cutHold_ > 0.f) { cutHold_ -= dt; cCut_.store(0.f, std::memory_order_relaxed); cFreeze_.store(1.f, std::memory_order_relaxed); }
  else { cCut_.store(1.f, std::memory_order_relaxed); cFreeze_.store(0.f, std::memory_order_relaxed); }
  // sample CHOIR density/amp per act (the washed background)
  static const float CHOIR_LAM[6] = {0.f, 2.0f, 3.0f, 1.0f, 0.f, 4.0f};
  static const float CHOIR_AMP[6] = {0.f, 0.8f, 1.0f, 0.4f, 0.f, 1.0f};
  cChoirLam_.store(CHOIR_LAM[a], std::memory_order_relaxed); cChoir_.store(CHOIR_AMP[a], std::memory_order_relaxed);
  // LATER (GENDYN, Step 2): conductor envelope for the harm-tone soloist. AV-sync (a1) -> key off the `act` arg +
  // the existing within-section `f` ramp (NOT a literal secT_ cut). Design-spine weights: I:0 II:0.20 III:0.70
  // IV:0.50 V:0.30. ABSENT in EMERGE; under the arbor grind in EXTRACT (demoted to affect); EXPOSED in the TURN.
  static const float GENDYN_ENV[6] = {0.f, 0.f, 0.20f, 0.70f, 0.50f, 0.30f};
  cGendyn_.store(GENDYN_ENV[a] * f, std::memory_order_relaxed);
  // LATER tier (SIEVES): hard gate the machine-math pitch alphabet. Per AV-sync (a1) the gate keys off the
  // VISUAL `act` arg + the within-act `f` ramp (NOT the free-running secT_ wall-clock) so it lands with the
  // looping visual III->IV/V structure and tracks the 1-5 act-audition jumps + hesitation. OFF for Acts I-II
  // (diatonic), ON from Act III (EXTRACT) -- the sweetness curdles. In Act V the lattice resolves back toward
  // diatonic only LATE and INCOMPLETELY (f ramps 0..1 across the visual Act V) but NEVER fully leaves: it
  // floors at 0.6 > the degHz() lattice threshold (0.5), so the V comfort sits ON the alien lattice -- the
  // never-reverting sieve is the audio twin of the monotonic extractionDebt haunt. Makes no sound of its own.
  float sieve = (a >= 3) ? 1.f : 0.f;
  if (a == 5) { float fr = clamp01((f - 0.6f) / 0.4f); sieve = std::max(0.6f, 1.f - fr); }
  cSieve_.store(sieve, std::memory_order_relaxed);
  // LATER (GLITCH AXIS, Step 4): per-act target floats for the grid-locked edit bus (islot 1-5). Stored here in
  // update() keyed off the `act` arg + the within-section `f` ramp (AV-sync decision a1) -- consistent with every
  // other LATER envelope above; the audio thread owns the actual density CLOCK (grid-locked, no ring push). Density
  // crescendos through III, peaks early IV (each glitch an isolated edit in the stripped TURN), resolves to 0 in V.
  static const float GLITCH_DENS[6]  = {0.f, 0.f, 0.04f, 0.30f, 0.35f, 0.f};   // {fallback,I,II,III,IV,V}: P(op | grid hit); III crescendo via f, IV TURN loudest
  static const float GLITCH_CRUSH[6] = {0.f, 0.f, 0.20f, 0.75f, 0.55f, 0.f};   // 0..1 destroyed (resolving in V)
  static const float GLITCH_STUT[6]  = {0.f, 0.f, 0.f,   0.60f, 0.40f, 0.f};   // P(stutter | op)
  static const float GLITCH_GATE[6]  = {0.f, 0.f, 0.10f, 0.45f, 0.35f, 0.f};   // master micro-gate depth
  // III rises across the section (crescendo into the cliff); IV is an aftershock; V ramps to 0 (resolves out).
  float glDensF = (a == 3) ? (0.4f + 0.6f * f) : (a == 5 ? (1.f - f) : 1.f);
  cGlitchDens_.store(GLITCH_DENS[a] * glDensF, std::memory_order_relaxed);
  cCrush_.store(GLITCH_CRUSH[a], std::memory_order_relaxed);                    // also drives the per-sample bitcrush depth
  cStutLen_.store(0.18f - 0.13f * f, std::memory_order_relaxed);               // stutter window tightens across the build
  cGateAmt_.store(GLITCH_GATE[a], std::memory_order_relaxed);
  cGlitchOp_.store(int(GLITCH_STUT[a] * 1000.f + 0.5f), std::memory_order_relaxed);   // P(stutter|op) x1000 (audio reads back /1000)
  // ---- PHASE 2b: drive the haunted chorus from the act + within-section build f + the popped worker. Peaks in Act V. ----
  static const int   CHORN[6]   = {1, 1, 2, 3, 2, 5};                               // ghosts per on-image whisper: all at once in V
  static const float CHORDET[6] = {0.f, 0.06f, 0.12f, 0.20f, 0.10f, 0.35f};         // detune span (semitones), builds with the act
  cChorusN_.store(CHORN[a], std::memory_order_relaxed);
  cChorusDet_.store(CHORDET[a] * (0.5f + 0.5f * f), std::memory_order_relaxed);
  { float regN = clamp01(cStoryDeg_.load(std::memory_order_relaxed) / float(modeN_ * 2));   // worker register -> throat
    cFmtShift_.store(0.82f + 0.36f * regN, std::memory_order_relaxed);                       // historical(low)->darker, contemporary->brighter
    cVoiceRough_.store(clamp01(cCostBeat_.load(std::memory_order_relaxed) * 1.2f), std::memory_order_relaxed); }   // cheap wage -> rougher voice
  // ---- PHASE 2c DREAM EMERGENCE: fold the SYNCED dream-attend phase with a CONDUCTOR envelope (opening ~89 s of
  // Act I + the Act-V recap) so the Shepard-noise->harmonic-pad crossfade also shapes the opening/recap, not only
  // when a dream is attended. Both inputs derive from the same synced clock (emergePhase from WoSWState; a/actElapsed_/f). ----
  { float condEmg = 0.f;
    if (a == 1) condEmg = clamp01(actElapsed_ / 89.f);                  // the opening ~89 s forms once
    else if (a == 5) condEmg = clamp01(0.30f + 0.70f * f);             // the Act-V recap re-forms
    float attendEmg = emergeActive ? clamp01(emergePhase) : 0.f;       // the attended/approached dream forming
    float emg = std::max(condEmg, attendEmg);                          // whichever forms harder drives the bed
    cEmerge_.store(emg, std::memory_order_relaxed);                    // SINGLE writer of cEmerge_
    // Each emergence STEP fires ONE EV_GLITCH (islot=0). Quantize emg into 8 denoise steps; fire only on the
    // RISING edge. cStutLen_ shrinks (60->8 ms) + cCrush_ falls (1->0) as emg->1 -> glitch resolves to clarity.
    if (emg > 0.001f) {
      int step = int(emg * 8.f); if (step > 7) step = 7;
      if (step > lastEmergeStep_) {
        cStutLen_.store(0.060f - 0.052f * emg, std::memory_order_relaxed);
        cCrush_.store(1.f - emg, std::memory_order_relaxed);
        Ev g{}; g.kind = EV_GLITCH; g.islot = 0; g.a = 0.060f - 0.052f * emg; g.b = 1.f - emg; push(g);
      }
      lastEmergeStep_ = step;
    } else { lastEmergeStep_ = -1; } }                                  // emergence cleared -> re-arm step 0
  // STORY SPINE (THEM): pop one named worker every Fibonacci phrase; year->register, wage->roughness, era->pan
  if (!stories_.empty()) {
    phraseT_ -= dt;
    if (phraseT_ <= 0.f) {
      static const float FIBPH[6] = {3.f, 5.f, 8.f, 5.f, 3.f, 2.f};            // phrase length in BEATS (Fibonacci)
      float beat = 1.f / std::max(0.5f, slotRate(depth, hesitation, activity_));
      phraseT_ = FIBPH[phraseIdx_ % 6] * beat; ++phraseIdx_;
      // LOW-END INTEREST: move the bass NOTE per phrase (root / P5 / P4 / min7), scaled by how active the
      // act is so the SEDUCTION stays near root and EXTRACTION roams.   fb   I     II     III    IV    V
      static const float SUBOST[4]  = { 1.0f, 1.5f, 1.3333f, 1.7818f };
      static const float SUBMOVE[6] = { 0.f, 0.0f, 0.25f, 1.0f, 0.0f, 0.6f };
      cSubRatio_.store(1.f + SUBMOVE[a] * (SUBOST[phraseIdx_ % 4] - 1.f), std::memory_order_relaxed);
      const Story& st = stories_[storyIdx_ % stories_.size()]; ++storyIdx_;
      cCostBeat_.store(st.costN, std::memory_order_relaxed);
      // historical->low, contemporary->high. Under the SIEVE (cSieve_ live) degHz reads `deg` as
      // SEMITONES, not mode-steps, so compute the register in semitones (+-12 + 12) to keep the
      // year->register spread audible across the alphabet swap; else the existing mode-step formula.
      int deg = (cSieve_.load(std::memory_order_relaxed) > 0.5f)
                  ? int((st.yearN * 2.f - 1.f) * 12.f) + 12
                  : int((st.yearN * 2.f - 1.f) * float(modeN_)) + modeN_;
      cStoryDeg_.store(float(deg), std::memory_order_relaxed);
      Ev e{}; e.kind = EV_NOTE; e.layer = 5; e.hz = degHz(deg) * 0.5f; e.amp = 0.055f; e.pan = 0.6f * (st.era ? -1.f : 1.f); push(e);   // the worker's sustained TONE
      // LATER (GENDYN, Step 2): the THEM foreground is ALWAYS exactly one soloist -- mux GENDYN xor WHISPER via
      // cVoiceSel_ (anti-mud rule 1). The SAME wage (st.costN) that drives cBeat_ roughness also sets the GENDYN
      // harshness, so one worker has two coupled symptoms (pad roughness + soloist grit). Toggle the mux per phrase.
      cGendynHarsh_.store(st.costN, std::memory_order_relaxed);
      // per-act GENDYN-vs-WHISPER bias (spec §3.1: I/II off, READ ~1-in-3, EXTRACT/TURN even, V rare/ghostly),
      // instead of a flat 50/50. Sim-thread mprng_ (same RNG fireGendyn uses) -> deterministic, lock-free.
      static const float GSEL[6] = {0.f, 0.f, 0.33f, 0.5f, 0.5f, 0.2f};   // index = act a (1..5; 0 fallback)
      mprng_ ^= mprng_ << 13; mprng_ ^= mprng_ >> 17; mprng_ ^= mprng_ << 5;
      cVoiceSel_.store((float(mprng_) / 4294967296.f) < GSEL[a] ? 1 : 0, std::memory_order_relaxed);
      if (cVoiceSel_.load(std::memory_order_relaxed) == 1 && cGendyn_.load(std::memory_order_relaxed) > 0.02f) fireGendyn();   // GENDYN's turn -> harm-tone soloist (only when its envelope is live; ABSENT in EMERGE/Act I)
      else if (!wordbank_.empty()) whisper(wordbank_[(storyIdx_ * 1103515245u + 12345u) % wordbank_.size()], curNode_, 0.5f * (st.era ? -1.f : 1.f));   // envelope dead OR whisper's turn -> whisper
    }
  }
  // ARRANGEMENT: each element follows ONLY its per-act palette gate. The per-element RANDOM dropout
  // texture (drone pause / ensemble drop-outs) is REMOVED at the artist's request — the in/out gating
  // distracted from the composition. Act transitions still glide via cGateTau_ (0.5 s init); the sub
  // pulse, conductor, techniques, Eno loops + grain cloud keep the bed alive without random silences.
  for (int i = 0; i < NGATE; ++i) cGate_[i].store(aGate[i], std::memory_order_relaxed);
  // recognizable PD melody: emit the next note ~once per beat; grains gated down so it emerges
  if (melodyOn_) {
    melNoteTimer_ -= dt;
    if (melNoteTimer_ <= 0.f) {
      const Tune& t = TUNES[melTune_ % 3];
      if (melNote_ >= t.n) { melodyOn_ = false; }
      else { float beat = 1.f / std::max(0.5f, slotRate(depth, hesitation, activity_)); melNoteTimer_ = beat * float(t.d[melNote_]);
        // pitched glockenspiel/kalimba (pluck pool repitches; agogo bells would not) one octave up
        Ev e{}; e.kind = EV_NOTE; e.layer = 0; e.hz = root_ * std::pow(2.f, float(t.s[melNote_] + 12) / 12.f); e.amp = 0.18f; e.pan = 0.f; push(e); ++melNote_; }
    }
  }

  // primary Euclidean (kick) + 2nd coprime Euclidean (timpani)
  int k = 2 + std::min(6, visits_ / 20);
  int n = (depth > 0.7f) ? ((clusterCur_ & 1) ? 12 : 8) : (depth > 0.4f) ? ((clusterCur_ & 1) ? 9 : 7) : ((clusterCur_ & 1) ? 13 : 5);
  if (k > n - 1) k = n - 1; if (k < 1) k = 1;
  if (k != euK_ || n != euN_) { euK_ = k; euN_ = n; Ev e{}; e.kind = EV_RHYTHM; e.islot = int(euclidMask(k, n)); e.layer = n; push(e); }
  int n2 = (clusterCur_ & 1) ? 7 : 5; if (n2 == n) n2 = (n2 == 5 ? 7 : 5);
  int k2 = 2 + std::min(4, visits_ / 30); if (k2 > n2 - 1) k2 = n2 - 1; if (k2 < 1) k2 = 1;
  if (k2 != euK2_ || n2 != euN2_) { euK2_ = k2; euN2_ = n2; Ev e{}; e.kind = EV_RHYTHM2; e.islot = int(euclidMask(k2, n2)); e.layer = n2; push(e); }

  if (melIdx_ < mel_.size()) {
    float period = 1.f / (1.6f * slotRate(depth, hesitation, activity_)); melTimer_ += dt;
    while (melIdx_ < mel_.size() && melTimer_ >= period) { melTimer_ -= period; int lay = mel_[melIdx_].second; bool accent = lay >= 10; lay %= 10;
      Ev e{}; e.kind = EV_NOTE; e.layer = lay; e.hz = mel_[melIdx_].first; e.amp = (accent ? 0.11f : 0.06f); e.pan = melPan_ + 0.18f * (float(melIdx_ % 5) - 2.f); push(e); ++melIdx_; }
  }
}

// ===================== audio-thread =====================
float AudioEngine::frand() { rng_ ^= rng_ << 13; rng_ ^= rng_ >> 17; rng_ ^= rng_ << 5; return float(rng_) / 4294967296.f; }
static inline float vrand(uint32_t& s) { s ^= s << 13; s ^= s >> 17; s ^= s << 5; return float(s) / 4294967296.f; }
int AudioEngine::allocVoice() { for (int i = 0; i < NVOX; ++i) if (!vox_[i].on) return i; return -1; }
int AudioEngine::allocCapped(int layer, int cap) { int cnt = 0, oldest = -1; float oldAge = -1.f;
  for (int i = 0; i < NVOX; ++i) if (vox_[i].on && vox_[i].layer == layer) { ++cnt; if (vox_[i].age > oldAge) { oldAge = vox_[i].age; oldest = i; } }
  if (cnt < cap) { int f = allocVoice(); if (f >= 0) return f; } return oldest >= 0 ? oldest : allocVoice(); }

// micro-timing: every onset delayed a random 2-156 ms (humanized)
void AudioEngine::schedule(const Ev& e) {
  int d = int((0.002f + 0.154f * frand()) * float(sr_)); if (d < 1) d = 1;
  for (auto& p : pend_) if (!p.used) { p.e = e; p.left = d; p.used = true; return; }
  trigger(e);   // pending full -> fire now
}

void AudioEngine::trigger(const Ev& e) {
  if (e.kind == EV_RHYTHM)  { euMask_  = uint32_t(e.islot); euLen_  = e.layer < 1 ? 1 : e.layer; return; }
  if (e.kind == EV_RHYTHM2) { euMask2_ = uint32_t(e.islot); euLen2_ = e.layer < 1 ? 1 : e.layer; return; }
  if (e.kind == EV_TRACE_OFF) { for (auto& v : vox_) if (v.on && v.layer == 3 && v.tslot == e.islot) if (v.life > v.age + 1.0f) v.life = v.age + 1.0f; return; }
  if (e.kind == EV_GLITCH) {   // 2c: arm a brief master-domain stutter+bitcrush window. NO voice, NO alloc.
    int len = int((e.a > 0.f ? e.a : cStutLen_.load(std::memory_order_relaxed)) * float(sr_));
    if (len < 64) len = 64; if (len > STUTN) len = STUTN;
    glitchLen_ = len; glitchT_ = len;
    stutPeriod_ = len / 3; if (stutPeriod_ < 16) stutPeriod_ = 16;            // loop the last ~1/3 of the window
    stutRead_ = (stutWr_ - stutPeriod_ + STUTN) & (STUTN - 1);
    crushAmt_ = clamp01(e.b);
    crushStride_ = 1 + int(crushAmt_ * 14.f);                                 // sample-rate decimation 1..15x
    crushPhase_ = 0; crushHoldL_ = 0.f; crushHoldR_ = 0.f;
    return;
  }
  int i; if (e.layer == 0 || e.layer == 1) i = allocCapped(e.layer, 4); else if (e.layer == 10) i = allocCapped(10, 2); else i = allocVoice(); if (i < 0) return;   // LATER (GENDYN): max 2 concurrent harm-tone voices
  Voice& v = vox_[i]; v = Voice{};
  v.on = true; v.layer = e.layer; v.hz = e.hz; v.amp = e.amp; v.pan = e.pan; v.age = 0.f; v.grng = 0x2545F491u ^ (uint32_t(i) * 2654435761u);
  auto setSample = [&](int sidx) { const Sample& s = samples_[sidx]; v.smp = s.data.data(); v.smpLen = int(s.data.size()); v.smpPos = 0.f;
    float ssr = s.srcSR > 0.f ? s.srcSR : float(sr_), ratio = s.pitched ? (v.hz / s.rootHz) : 1.f; while (ratio > 2.f) ratio *= 0.5f; while (ratio < 0.5f) ratio *= 2.f; v.smpRate = ratio * (ssr / float(sr_)); };
  auto setPartials = [&](int K, float inh) { v.K = K > 6 ? 6 : K; v.pnorm = 0.f; for (int k = 1; k <= v.K; ++k) { v.pm[k-1] = k*(1.f+inh*k*k); v.pa[k-1] = 1.f/std::pow(float(k),1.3f); v.pnorm += v.pa[k-1]; } if (v.pnorm <= 0.f) v.pnorm = 1.f; };
  switch (e.layer) {
    case 0: { v.life = 0.9f; v.atk = 0.008f; int s = pickSample(rPluck_, v.hz); if (s >= 0) setSample(s); else setPartials(5, 0.0008f); break; }
    case 1: { v.life = 1.2f; v.atk = 0.004f; int s = pickSample(rBell_, v.hz); if (s >= 0) setSample(s); else setPartials(6, 0.004f); break; }
    case 3: { v.life = 7.5f; v.atk = 0.9f; v.tslot = e.islot; int s = pickSample(rTrace_, v.hz); if (s >= 0) { setSample(s); v.life = std::min(7.5f, float(v.smpLen)/float(sr_)); } else setPartials(4, 0.0006f); break; }
    case 5: { int s = pickSample(rBass_, v.hz); if (s >= 0) { setSample(s); v.life = std::min(5.0f, float(v.smpLen)/float(sr_)); } else { v.life = 4.0f; setPartials(3, 0.0003f); } v.atk = 0.25f; break; }
    case 6: { v.life = 0.26f; v.atk = 0.001f; kickDuck_ = 0.12f; break; }   // kick sidechains the sub-bass down
    case 7: { v.life = 0.9f; v.atk = 0.003f; int s = pickSample(rTimp_, v.hz); if (s >= 0) setSample(s); else setPartials(4, 0.02f); break; }   // pitched timpani
    case 8: { v.life = 1.7f; v.atk = 0.001f; int s = pickSample(rMetal_, v.hz);                                                                  // industrial clang
              if (s >= 0 && frand() < 0.7f) setSample(s);                                                                                        // 70% bias to real industrial samples
              else { v.K = 5; float mr[5] = {1.f, 2.76f, 5.40f, 8.93f, 13.3f}; v.pnorm = 0.f; for (int k = 0; k < 5; ++k) { v.pm[k] = mr[k]; v.pa[k] = 1.f / std::pow(k + 1.f, 0.8f); v.pnorm += v.pa[k]; } } break; }
    case 4: { int syl = e.islot < 0 ? -e.islot : e.islot; v.syl = syl < 1 ? 1 : (syl > 5 ? 5 : syl); v.life = 1.5f + 0.55f * v.syl; v.body = 0.62f; v.atk = 0.09f; v.curSyl = -1;
              for (int i = 0; i < 5; ++i) v.vowels[i] = (e.vow >> (3 * i)) & 7;
              v.hz *= (e.a > 0.01f ? e.a : 1.f);                                     // 2b: per-voice chorus detune (e.a = ratio)
              if (e.islot < 0 && rVoice_.n > 0) {                                    // 2a: real ghostly voice -> load a speech SAMPLE on layer 4
                int sidx = pickSample(rVoice_, v.hz);                               //     (rVoice_ is non-pitched -> rate = source/host SR only)
                if (sidx >= 0) { const Sample& smp = samples_[sidx]; v.smp = smp.data.data(); v.smpLen = int(smp.data.size()); v.smpPos = 0.f;
                  v.smpRate = (smp.srcSR > 0.f ? smp.srcSR : float(sr_)) / float(sr_); v.life = std::min(2.6f, float(v.smpLen) / float(sr_)); v.atk = 0.05f; }
              }                                                                      //     v.smp set -> tickVoice takes the sample branch (still layer 4: duck/cap_/granulator apply)
              break; }   // synth formant whisper runs when no rVoice_ sample (graceful fallback)
    case 10: {   // LATER (GENDYN): seed K=GSEG breakpoints for the 2nd-order random walk (alias map @ Voice struct).
      v.syl = GSEG;                                                     // segCount -> syl
      float h = std::pow(clamp01(cGendynHarsh_.load(std::memory_order_relaxed)), 0.6f);   // gamma widens audible spread
      v.body = h;                                                       // harshness -> body (cached for the walk)
      v.lp = 1.f + 3.f * h;                                             // foldback drive -> lp (tanh(drive*x))
      for (int k = 0; k < GSEG; ++k) {
        v.pm[k] = 0.3f + 1.7f * vrand(v.grng);                          // durations d[k] in [DMIN=0.3, DMAX=2.0]
        v.pa[k] = 2.f * vrand(v.grng) - 1.f;                            // amplitudes a[k] in [-1,1]
        v.fcoef[k / 3][k % 3] = 0.f;                                    // time-velocities vT[k] = 0
        v.fz_[k / 2][k % 2] = 0.f;                                      // amp-velocities  vA[k] = 0
      }
      v.phase = 0.f; v.atk = 0.03f; v.life = 2.5f + 1.5f * vrand(v.grng);   // fundamental v.hz already = e.hz
      break; }
    default: v.life = 0.15f; v.atk = 0.01f; setPartials(2, 0.f); break;
  }
}

float AudioEngine::tickVoice(Voice& v, float isr, float depth, float bright) {
  v.age += isr; if (v.age >= v.life) { v.on = false; return 0.f; } const float u = v.age / v.life;
  if (v.layer == 6) { float fk = 36.f + 80.f * std::exp(-v.age / 0.032f); v.phase += fk * isr; if (v.phase >= 1.f) v.phase -= 1.f;
    float click = (v.age < 0.004f) ? (1.f - v.age / 0.004f) * (vrand(v.grng) * 2.f - 1.f) * 0.6f : 0.f; return (std::sin(kTAU * v.phase) * std::exp(-v.age / 0.10f) + click) * v.amp; }
  if (v.layer == 9) { float n = vrand(v.grng) * 2.f - 1.f; v.lp = dn(v.lp + 0.5f * (n - v.lp)); return (n - v.lp) * std::exp(-v.age / 0.0035f) * v.amp; }   // off-beat "and" click (high-passed)
  // LATER (GLITCH AXIS, Step 4): a glitch cap_ voice (tslot=-11) with a POSITIVE rate is a STUTTER/freeze-loop --
  // wrap smpPos back to the captured window [stutOrigin_, stutOrigin_+stutPlay_) so the slice re-triggers (the
  // machine re-editing the worker's recorded voice). Negative-rate -11 voices are REVERSE-SLAMs -> no wrap (the
  // smpPos<0 check below terminates them). The window is kept inside the ring at arm time, so reads stay in bounds.
  if (v.tslot == -11 && v.smpRate > 0.f && v.smpPos >= v.pm[0] + v.pm[1]) v.smpPos = v.pm[0];   // per-voice loop window (pm[0]=origin, pm[1]=len; free scratch -- cap_ voices return at L619 before any pm[] read)
  if (v.smp) { if (v.smpPos >= v.smpLen - 2 || v.smpPos < 0.f) { v.on = false; return 0.f; } int i0 = int(v.smpPos); float fr = v.smpPos - i0; float s = v.smp[i0]*(1.f-fr) + v.smp[i0+1]*fr; v.smpPos += v.smpRate;
    float env; if (v.layer == 3 || v.layer == 5) env = (u < (v.atk/v.life)) ? (u/std::max(1e-4f, v.atk/v.life)) : (u < 0.6f ? 1.f : std::max(0.f, 1.f-(u-0.6f)/0.4f));
    else { float a = v.atk/v.life; env = (u < a) ? (u/std::max(1e-4f,a)) : (u > 0.85f ? (1.f-(u-0.85f)/0.15f) : 1.f); }
    v.phase += 64.f * isr; if (v.phase >= 1.f) v.phase -= 1.f;                          // GRANULATION (deeper -> grain cloud):
    float gmod = 0.40f + 0.60f * (0.5f - 0.5f * std::cos(kTAU * v.phase));              // chop the sample into a grain stream
    return s * env * v.amp * gmod; }
  if (v.layer == 4) {
    // step the formant resonators through the word's vowels (speech-like synthesis)
    int sk = (u < v.body) ? int((u / v.body) * float(v.syl)) : (v.syl - 1); if (sk < 0) sk = 0; if (sk >= v.syl) sk = v.syl - 1;
    if (sk != v.curSyl) { v.curSyl = sk; formantForIdx(v.vowels[sk], v.fmt[0], v.fmt[1], v.fmt[2]);
      const float fsh = cFmtShift_.load(std::memory_order_relaxed), rough = cVoiceRough_.load(std::memory_order_relaxed);   // 2b: place->throat, wage->strain
      float bw[3] = {90.f, 110.f, 140.f}; for (int k = 0; k < 3; ++k) { v.fmt[k] *= fsh; bw[k] *= (1.f + 0.5f * rough); float r = std::exp(-kPI*bw[k]/float(sr_)); v.fcoef[k][0]=(1.f-r); v.fcoef[k][1]=-2.f*r*std::cos(kTAU*v.fmt[k]/float(sr_)); v.fcoef[k][2]=r*r; } }
    float env, cons = 0.f;
    if (u < v.body) { float t = u/v.body, ph = t*float(v.syl), frac = ph-std::floor(ph); env = std::pow(0.5f-0.5f*std::cos(kTAU*frac), 0.6f)*std::sin(kPI*t);
      // consonant ONSET: a brief high-passed noise burst (fricative) at each syllable start -> reads as speech
      float cnz = vrand(v.grng)*2.f-1.f; v.lp = dn(v.lp + 0.55f*(cnz - v.lp)); float chp = cnz - v.lp;
      if (frac < 0.16f) cons = chp * (1.f - frac/0.16f) * 0.6f * std::sin(kPI*t); }
    else { float t = (u-v.body)/(1.f-v.body), gp = t*(6.f+10.f*t), gf = gp-std::floor(gp), duty = 0.6f*(1.f-t); float grain = (gf<duty)?(0.5f-0.5f*std::cos(kTAU*gf/std::max(1e-3f,duty))):0.f; env = grain*(1.f-t); }
    float nz = vrand(v.grng)*2.f-1.f, s = 0.f; const float w[3] = {1.f, 0.7f, 0.45f};
    for (int k = 0; k < 3; ++k) { float y = v.fcoef[k][0]*nz - v.fcoef[k][1]*v.fz_[k][0] - v.fcoef[k][2]*v.fz_[k][1]; v.fz_[k][1]=dn(v.fz_[k][0]); v.fz_[k][0]=dn(y); s += w[k]*y; } return (s * env + cons) * v.amp * 1.25f; }
  if (v.layer == 10) {   // LATER (GENDYN): true 2nd-order Xenakis DSS. Alias map documented at the Voice struct.
    // advance the read phase one fundamental period; the K=GSEG breakpoints are re-walked ONCE per period.
    float pPrev = v.phase; v.phase += v.hz * isr;
    if (v.phase >= 1.f) {   // ---- period boundary: 2nd-order random walk + elastic-mirror barriers (once/period) ----
      v.phase -= std::floor(v.phase);
      const float h = v.body;                                          // gamma-mapped harshness (cached @ trigger)
      const float STEPT = 0.02f + 0.10f * h, VTMAX = 0.15f + 0.35f * h;   // time: step + velocity cap
      const float STEPA = 0.04f + 0.30f * h, VAMAX = 0.08f + 0.50f * h;   // amp:  step + velocity cap
      for (int k = 0; k < GSEG; ++k) {
        float& vT = v.fcoef[k / 3][k % 3];                             // time-velocity vT[k]
        float& vA = v.fz_[k / 2][k % 2];                               // amp-velocity  vA[k]
        // duration walk in box [DMIN,DMAX] = [0.3,2.0] with elastic mirror (excess reflects, velocity negates)
        vT += (2.f * vrand(v.grng) - 1.f) * STEPT; if (vT > VTMAX) vT = VTMAX; else if (vT < -VTMAX) vT = -VTMAX;
        float d = v.pm[k] + vT;
        if (d > 2.0f) { d = 4.0f - d; vT = -vT; } else if (d < 0.3f) { d = 0.6f - d; vT = -vT; }
        if (d > 2.0f) d = 2.0f; else if (d < 0.3f) d = 0.3f;           // clamp residual (double-reflect)
        v.pm[k] = d;
        // amplitude walk in box [-1,+1] with elastic mirror
        vA += (2.f * vrand(v.grng) - 1.f) * STEPA; if (vA > VAMAX) vA = VAMAX; else if (vA < -VAMAX) vA = -VAMAX;
        float a = v.pa[k] + vA;
        if (a > 1.f) { a = 2.f - a; vA = -vA; } else if (a < -1.f) { a = -2.f - a; vA = -vA; }
        if (a > 1.f) a = 1.f; else if (a < -1.f) a = -1.f;
        v.pa[k] = a;
      }
    }
    // per-sample read: locate the breakpoint segment for v.phase (durations normalize the cycle), linear interp.
    float dsum = 0.f; for (int k = 0; k < GSEG; ++k) dsum += v.pm[k]; if (dsum < 1e-4f) dsum = 1.f;
    float target = v.phase * dsum, acc = 0.f; int seg = 0;
    for (int k = 0; k < GSEG; ++k) { if (target < acc + v.pm[k] || k == GSEG - 1) { seg = k; break; } acc += v.pm[k]; }
    float fseg = (v.pm[seg] > 1e-6f) ? (target - acc) / v.pm[seg] : 0.f; if (fseg < 0.f) fseg = 0.f; else if (fseg > 1.f) fseg = 1.f;
    float a0 = v.pa[seg], a1 = v.pa[(seg + 1) % GSEG], x = a0 + (a1 - a0) * fseg;
    float xf = std::tanh(v.lp * x);                                    // v.lp = foldback drive (1..4); soft saw->slammed fold
    float a = v.atk / v.life, env = (u < a) ? (u / std::max(1e-4f, a)) : (u > 0.88f ? (1.f - (u - 0.88f) / 0.12f) : 1.f);
    return xf * env * v.amp;
  }
  // LATER (ARBORESCENCES, Step 3): layer-2 grains marked tslot=-9 glissando their pitch toward the stored
  // target v.body. Constant glide coef (12*isr), NOT v.lp (which stays the additive output one-pole below) ->
  // reaches the target over ~one grain life. Applied BEFORE the additive phase advance so the swept hz feeds it.
  if (v.layer == 2 && v.tslot == -9) { v.hz += (v.body - v.hz) * (12.f * isr); }
  v.phase += v.hz * isr; if (v.phase >= 1.f) v.phase -= std::floor(v.phase);
  float s = 0.f; const float nyq = 0.45f * float(sr_);
  for (int k = 0; k < v.K; ++k) { float pf = v.hz * v.pm[k]; if (pf >= nyq) break; s += v.pa[k] * std::sin(kTAU * v.phase * v.pm[k]); } s /= v.pnorm;
  float env; if (v.layer == 2) { float c = u*2.f-1.f; env = std::exp(-c*c*6.f); }
  else if (v.layer == 3) env = (u < 0.12f) ? (u/0.12f) : (u < 0.55f ? 1.f : std::max(0.f, 1.f-(u-0.55f)/0.45f));
  else { float a = v.atk/v.life; env = (u < a) ? (u/std::max(1e-4f,a)) : std::exp(-3.5f*(u-a)/(1.f-a)); }
  float out = s * env * v.amp; float cut = (v.layer == 1 || v.layer == 8) ? (0.55f + 0.40f*bright) : (0.22f + 0.45f*bright);
  v.lp = dn(v.lp + cut * (out - v.lp)); return v.lp;
}

void AudioEngine::fireKick() { Ev e{}; e.kind = EV_NOTE; e.layer = 6; e.hz = jiBassHz(); e.amp = ((euStep_ == 0) ? 0.55f : 0.38f) * gate_[G_KICK]; e.pan = 0.f; schedule(e); }
void AudioEngine::fireTimp() { timpDeg_ = (timpDeg_ + 1 + int(frand() * 2.f)) % (modeN_ * 2); Ev e{}; e.kind = EV_NOTE; e.layer = 7; e.hz = degHz(timpDeg_) * 0.5f; e.amp = 0.24f; e.pan = 0.18f * (2.f * frand() - 1.f); schedule(e); }
void AudioEngine::fireClang() { Ev e{}; e.kind = EV_NOTE; e.layer = 8; e.hz = 60.f + 200.f * frand(); e.amp = 0.14f + 0.10f * frand(); e.pan = 0.8f * (2.f * frand() - 1.f); schedule(e); }
// LATER (GLITCH AXIS, Step 4): the grid-locked edit bus. Called on the AUDIO thread DIRECTLY inside the kick
// euStep block right AFTER fireKick(), only on a lit grid step ((euMask_>>euStep_)&1). Bypasses schedule()/pend_
// (which would smear the grid-lock by 2-156 ms) -- exactly like fireKick. Owns islot 1-5; NEVER fires islot 0
// (Phase-2's emergence window). Arbiter: suppress act-glitch while cEmerge_ is mid-transition (islot-0 owner).
// Big ops (reverse-slam=3, freeze=4) gate on the downbeat (euStep_==0); crush(1)/micro-gate(5) on any hit. The
// cliff: when cFreeze_ is latched (visual III->IV edge) force a dense reverse-slam+freeze cluster on the downbeat.
// Glitch cap_-voices (stutter/reverse/freeze-loop, all tslot=-11) capped <=6 via a census inside the layer-2 28-cap.
void AudioEngine::fireGlitch(float activity, bool whisperOn) {
  const float emg = cEmerge_.load(std::memory_order_relaxed);
  if (emg > 0.02f && emg < 0.98f) return;                          // arbiter: Phase-2 islot-0 owns the bus mid-transition
  const float frzLatched = cFreeze_.load(std::memory_order_relaxed);   // cliff: the III->IV cut is held -> guillotine cluster
  const bool downbeat = (euStep_ == 0);

  // --- THE CLIFF: dense reverse-slam + freeze on the downbeat while the cut is latched (near-silent through it) ---
  if (frzLatched > 0.5f) {
    if (downbeat && capRms_ > 0.0006f) {
      int gc = 0; for (auto& v : vox_) if (v.on && v.layer == 2 && v.tslot == -11) ++gc;
      if (gc < 6) {                                                // reverse-slam off cap_ (the worker's recorded voice)
        int gi = allocCapped(2, 28); if (gi >= 0) { Voice& g = vox_[gi]; g = Voice{};
          g.on = true; g.layer = 2; g.tslot = -11; g.smp = cap_.data(); g.smpLen = CAPN;
          g.smpPos = float((capPos_ - 64 + CAPN) & (CAPN - 1)); g.smpRate = -1.f;   // negative rate -> reverse (no wrap)
          float revAvail = g.smpPos / float(sr_);                                   // reverse audio available before smpPos<0
          g.life = std::min(1.0f, revAvail); g.atk = std::min(0.30f, 0.30f * g.life);  // length-bound: a ring-wrap no longer leaves a dead 1.0s/0.3s-atk envelope
          g.amp = 0.12f; g.pan = 0.f; g.grng = rng_ ^ uint32_t(gi * 2654435761u); }
      }
    }
    freezeHold_ = 2;                                               // hold the SINGLE reverb freeze ~2 grid steps
    return;
  }

  const float dens = cGlitchDens_.load(std::memory_order_relaxed);
  if (dens <= 0.f) { microGateTgt_ = 1.f; return; }                // axis OFF (Acts I/V) -> let the master gate re-open
  microGateTgt_ = (frand() < cGateAmt_.load(std::memory_order_relaxed)) ? 0.f : 1.f;   // islot 5: re-roll per LIT GRID HIT (decoupled from the op-density p) -- gate breathes on every hit (AUDIO_LATER_PLAN.md L197/L216)
  float p = dens * (0.4f + 0.6f * activity) * (whisperOn ? 0.5f : 1.f);   // duck under the whisper (edits fall in the gaps)
  if (frand() >= p) return;

  const float crush = cCrush_.load(std::memory_order_relaxed);
  const float stutPr = float(cGlitchOp_.load(std::memory_order_relaxed)) / 1000.f;   // P(stutter | op)
  const bool capLive = capRms_ > 0.0006f;
  int gc = 0; for (auto& v : vox_) if (v.on && v.layer == 2 && v.tslot == -11) ++gc;   // census: glitch cap_-voices <=6

  // op pick. stutter (2)/reverse (3)/freeze (4) need a live cap_; reverse+freeze are BIG (downbeat only). crush
  // (1) + micro-gate (5) operate on the full master bus (always have material). The micro-gate target is re-rolled
  // above on EVERY LIT GRID HIT (decoupled from this op-density p) so the gate breathes across the section even
  // when no op is chosen -- per AUDIO_LATER_PLAN.md L197 "crush/micro-gate on any hit" + L216 "re-rolled per grid hit".
  float r = frand();
  if (capLive && gc < 6 && r < stutPr) {                           // islot 2: STUTTER (re-trigger loop over cap_)
    float beat = float(sr_) / std::max(0.5f, slotRate(cDepth_.load(std::memory_order_relaxed), cHesitation_.load(std::memory_order_relaxed), activity));
    int win = int(clamp01(cStutLen_.load(std::memory_order_relaxed)) * beat); if (win < 64) win = 64; if (win > CAPN / 4) win = CAPN / 4;
    stutPlay_ = float(win);
    stutOrigin_ = (capPos_ - win + CAPN) & (CAPN - 1); if (stutOrigin_ + win >= CAPN) stutOrigin_ = CAPN - win - 1;   // keep window inside the ring
    int gi = allocCapped(2, 28); if (gi >= 0) { Voice& g = vox_[gi]; g = Voice{};
      g.on = true; g.layer = 2; g.tslot = -11; g.smp = cap_.data(); g.smpLen = CAPN; g.smpPos = float(stutOrigin_); g.smpRate = 1.f;
      g.pm[0] = float(stutOrigin_); g.pm[1] = stutPlay_;          // per-voice loop window (no cross-contamination with a later arm)
      g.life = 0.10f + 0.45f * frand(); g.atk = 0.01f; g.amp = 0.10f; g.pan = 0.18f * (2.f * frand() - 1.f); g.grng = rng_ ^ uint32_t(gi * 40503u); }
  } else if (capLive && gc < 6 && downbeat && r < stutPr + 0.30f) {   // islot 3: REVERSE-SLAM (negative rate; downbeat)
    int gi = allocCapped(2, 28); if (gi >= 0) { Voice& g = vox_[gi]; g = Voice{};
      g.on = true; g.layer = 2; g.tslot = -11; g.smp = cap_.data(); g.smpLen = CAPN;
      g.smpPos = float((capPos_ - 64 + CAPN) & (CAPN - 1)); g.smpRate = -1.f;
      float revAvail = g.smpPos / float(sr_);
      g.life = std::min(1.0f, revAvail); g.atk = std::min(0.30f, 0.30f * g.life);   // length-bound the reverse-slam (see cliff arm)
      g.amp = 0.10f; g.pan = 0.f; g.grng = rng_ ^ uint32_t(gi * 2246822519u); }
  } else if (capLive && gc < 6 && downbeat && r < stutPr + 0.45f) {   // islot 4: FREEZE (latched looped window + reverb freeze)
    int win = CAPN / 16; stutPlay_ = float(win);                    // ~0.09 s window @ 44.1k
    stutOrigin_ = (capPos_ - win + CAPN) & (CAPN - 1); if (stutOrigin_ + win >= CAPN) stutOrigin_ = CAPN - win - 1;
    freezeHold_ = 1 + int(frand() * 2.f);                           // 1-2 grid steps of the SINGLE reverb freeze
    int gi = allocCapped(2, 28); if (gi >= 0) { Voice& g = vox_[gi]; g = Voice{};
      g.on = true; g.layer = 2; g.tslot = -11; g.smp = cap_.data(); g.smpLen = CAPN; g.smpPos = float(stutOrigin_); g.smpRate = 1.f;
      g.pm[0] = float(stutOrigin_); g.pm[1] = stutPlay_;          // per-voice loop window (no cross-contamination with a later arm)
      g.life = 0.4f + 0.3f * float(freezeHold_); g.atk = 0.03f; g.amp = 0.09f; g.pan = 0.f; g.grng = rng_ ^ uint32_t(gi * 668265263u); }
  } else {                                                          // islot 1: BITCRUSH (arm the per-sample sample-and-hold)
    crushBits_ = int(0.20f * float(sr_));                          // armed ~0.2 s (decremented per sample before the tanh)
    crushPh_ = 1.f;                                                // force re-latch on sample 0: crushPh_ += inc crosses 1.f -> crushHoldL_/R_ re-quantize the live L/R (no stale-hold click)
    (void)crush;
  }
}
void AudioEngine::fireGendyn() {   // LATER (GENDYN): the THEM harm-tone soloist (low/mid, anti-alias clamped). Reads atomics once.
  // Called on the SIM thread (story-pop) -> PUSH the ring (render pops -> schedule) and use the sim-side mprng_
  // for the center-ish pan jitter (NEVER the audio-thread frand()/rng_ or pend_).
  float hz = degHz(int(cStoryDeg_.load(std::memory_order_relaxed))); if (hz < 70.f) hz = 70.f; else if (hz > 440.f) hz = 440.f;
  mprng_ ^= mprng_ << 13; mprng_ ^= mprng_ >> 17; mprng_ ^= mprng_ << 5;
  Ev e{}; e.kind = EV_NOTE; e.layer = 10; e.hz = hz;
  e.amp = 0.06f + 0.12f * cGendyn_.load(std::memory_order_relaxed);   // scale off the conductor envelope (was a near-inert 0.10+0.04*; peak ~preserved, lower acts quieter)
  // STEP 5 anti-mud rule 1: at most ONE foreground stochastic voice. In EXTRACT (Act III) ARBOR is the single
  // F; whenever a fan is active, DEMOTE GENDYN to affect -- amp x0.5, no solo gate -- so it sits UNDER the sheaf
  // (the processed-voice harshness) instead of competing as a second foreground. cArbor_ peaks in III (ARB[3]=1).
  if (cArbor_.load(std::memory_order_relaxed) > 0.5f) e.amp *= 0.5f;
  e.pan = 0.18f * (2.f * (float(mprng_) / 4294967296.f) - 1.f);   // center-ish
  push(e);
}
void AudioEngine::fireAnd() { int i = allocCapped(9, 3); if (i < 0) return; Voice& v = vox_[i]; v = Voice{}; v.on = true; v.layer = 9; v.life = 0.05f; v.age = 0.f; v.amp = 0.05f; v.pan = 0.1f * (2.f * frand() - 1.f); v.grng = rng_ ^ uint32_t(i * 668265263u); }   // tics lower
void AudioEngine::fireLoop(int L) {   // Eno loop L fires a note on its layer, relative to the current worker's register
  int deg = int(cStoryDeg_.load(std::memory_order_relaxed)) + LOOPDEG[L];
  int lay = LOOPLAYER[L];
  Ev e{}; e.kind = EV_NOTE; e.layer = lay; e.hz = degHz(deg) * (lay == 5 ? 0.5f : 1.f);
  e.amp = LOOPAMP[L]; e.pan = 0.7f * (2.f * float(L) / float(NLOOP - 1) - 1.f); schedule(e);
}

void AudioEngine::render(al::AudioIOData& io) {
  const int nf = int(io.framesPerBuffer()), nch = int(io.channelsOut());
  if (!ready()) { for (int c = 0; c < nch; ++c) for (int f = 0; f < nf; ++f) io.out(c, f) = 0.f; return; }

  Ev e; while (pop(e)) { if (e.kind == EV_RHYTHM || e.kind == EV_RHYTHM2 || e.kind == EV_TRACE_OFF || e.kind == EV_GLITCH) trigger(e); else schedule(e); }

  int cl = cClusterId_.load(std::memory_order_relaxed);
  if (cl != lastClusterId_) { lastClusterId_ = cl; Ev p{}; p.kind = EV_NOTE; p.layer = 5; p.hz = jiBassHz(); p.amp = 0.06f; p.pan = 0.f; schedule(p); }

  const float depth = clamp01(cDepth_.load(std::memory_order_relaxed));
  const float hes = cHesitation_.load(std::memory_order_relaxed), act = cActivity_.load(std::memory_order_relaxed);
  const float padRoot = cClusterRoot_.load(std::memory_order_relaxed) * 0.5f;
  const float mb = cMoodBright_.load(std::memory_order_relaxed), md = cMoodDens_.load(std::memory_order_relaxed), mw = cMoodWet_.load(std::memory_order_relaxed);

  float tensionTgt = clamp01(0.5f * hes + 0.5f * (1.f - depth));
  tension_ = dn(tension_ + (tensionTgt - tension_) * gco(0.3f, nf, sr_));
  moodLP_   += (clamp01((0.14f + 0.72f * depth) * (0.6f + 0.7f * mb)) - moodLP_) * gco(0.6f, nf, sr_);
  padBloom_ += ((0.30f + 0.70f * depth) - padBloom_) * gco(0.6f, nf, sr_);
  reverbWet_ += (clamp01(0.42f + 0.16f * (1.f - depth) + 0.10f * hes + 0.14f * mw) - reverbWet_) * gco(0.6f, nf, sr_);   // more global reverb
  subAmp_   += ((0.125f + 0.15f * (1.f - depth)) - subAmp_) * gco(7.f, nf, sr_);   // bass +~5 dB
  shepGain_ += ((0.011f + 0.007f * depth + 0.006f * tension_) - shepGain_) * gco(1.f, nf, sr_);   // tamed, -5 dB more
  emerge_ += (cEmerge_.load(std::memory_order_relaxed) - emerge_) * gco(0.4f, nf, sr_);   // 2c: glide emergence (~0.4 s, click-free)
  shepGain_ *= (1.f - 0.9f * emerge_);   // 2c INVERSE: as the image forms, the rising Shepard-NOISE bed recedes (faint air remains)
  shepRateTgt_ = (hes > 0.55f ? -1.f : 1.f) * (1.f / (80.f - 40.f * hes)); shepRate_ += (shepRateTgt_ - shepRate_) * gco(8.f, nf, sr_);   // near-static (~80 s/cycle)
  bool whisperOn = false; for (auto& v : vox_) if (v.on && v.layer == 4) { whisperOn = true; break; }
  duckGain_ += ((whisperOn ? 0.6f : 1.f) - duckGain_) * gco(0.25f, nf, sr_);
  lowDuck_  += ((whisperOn ? 0.8f : 1.f) - lowDuck_) * gco(0.25f, nf, sr_);
  // arrangement gates: drone pause / ensemble drop-outs / dubstep growl (glided, no clicks)
  for (int i = 0; i < NGATE; ++i)
    gate_[i] += (cGate_[i].load(std::memory_order_relaxed) - gate_[i]) * gco(cGateTau_[i].load(std::memory_order_relaxed), nf, sr_);
  growl_       += (cGrowl_.load(std::memory_order_relaxed)       - growl_)       * gco(0.4f, nf, sr_);
  cut_    += (cCut_.load(std::memory_order_relaxed)    - cut_)    * gco(0.05f, nf, sr_);   // IV structural cut (~50 ms)
  // LATER (GLITCH AXIS, Step 4): a glitch FREEZE op forces the SINGLE existing reverb freeze high via freezeHold_
  // (a grid-step countdown advanced in the euStep block) -- shared with the sim's cFreeze_, NO 2nd reverb-freeze,
  // NO race (audio never writes cFreeze_). freezeHold_>0 OR the sim latch -> freeze target 1.
  float freezeTgt = cFreeze_.load(std::memory_order_relaxed); if (freezeHold_ > 0) freezeTgt = 1.f;
  freeze_ += (freezeTgt - freeze_) * gco(0.10f, nf, sr_);
  // 1e anti-mud census (once/buffer). LATER (STEP 5 / rule 4): the layer-2 grains (cloud/arbor/glitch/choir/
  // shred/Risset) are all v.on so they fall into nOn automatically. The GENDYN soloist (layer 10) is a LOUD
  // FOREGROUND voice, not a quiet grain, so count each active layer-10 voice with EXTRA foreground weight
  // (foreWt extra) -> a GENDYN solo riding a full tutti still pulls busyGain back, keeping the 1/sqrt(n) headroom.
  int nOn = 0, nGendyn = 0; for (auto& v : vox_) if (v.on) { ++nOn; if (v.layer == 10) ++nGendyn; }
  float foreWt = float(nOn) + 4.f * float(nGendyn);                                     // layer-10 weighs ~5x in the census
  float busyGain = 1.f / (1.f + 0.012f * std::max(0.f, foreWt - 26.f));                 // gentle headroom as the tutti fills

  float decay = std::min(0.985f, 0.86f + 0.10f * (1.f - depth)), damp = 0.35f + 0.30f * (1.f - depth);
  if (std::fabs(decay - lastDecay_) > 1e-3f) { reverb_.decay(decay); lastDecay_ = decay; }
  if (std::fabs(damp - lastDamp_) > 1e-3f) { reverb_.damping(damp); lastDamp_ = damp; }

  const float isr = 1.f / float(sr_), master = 0.28f;
  const float aLo = 1.f - std::exp(-kTAU * 300.f / float(sr_)), aHi = 1.f - std::exp(-kTAU * 3000.f / float(sr_));
  const float slot = slotRate(depth, hes, act);
  // dubstep growl wobble rate VARIES a lot: re-pick a musical multiple of the beat every ~0.5-2 s,
  // and add a slow continuous sweep so the wub is never static.
  wobChange_ -= float(nf) * isr;
  if (wobChange_ <= 0.f) { static const float WM[8] = {0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f, 8.f};
    wobMult_ = WM[int(frand() * 8.f) % 8]; wobChange_ = 0.5f + 1.6f * frand();
    growlDutyTgt_ = (frand() < 0.6f) ? 1.f : 0.f; }     // ~40% of intervals: wobble OFF -> clean drone
  growlDuty_ += (growlDutyTgt_ - growlDuty_) * gco(0.25f, nf, sr_);
  const float wobHz = slot * wobMult_ * (1.f + 0.25f * std::sin(kTAU * panLFO_ * 1.7f));
  const float fifthB = 1.5f - 0.006f * tension_;
  // talking growl: tune the growl's formant resonators to the LAST whispered word's vowel
  float wf0 = cFmt0_.load(std::memory_order_relaxed), wf1 = cFmt1_.load(std::memory_order_relaxed), wf2 = cFmt2_.load(std::memory_order_relaxed);
  if (wf0 != lastFmt_[0] || wf1 != lastFmt_[1]) {
    lastFmt_[0] = wf0; lastFmt_[1] = wf1; lastFmt_[2] = wf2;
    const float gf[3] = {wf0, wf1, wf2}, gbw[3] = {110.f, 120.f, 150.f};
    for (int k = 0; k < 3; ++k) { float r = std::exp(-kPI * gbw[k] / float(sr_));
      growlFmtCoef_[k][0] = (1.f - r); growlFmtCoef_[k][1] = -2.f * r * std::cos(kTAU * gf[k] / float(sr_)); growlFmtCoef_[k][2] = r * r; } }

  // industrial-clang scheduler (sparse, "in places"; a touch more likely when tense)
  fxTimer_ -= float(nf) * isr; if (fxTimer_ <= 0.f) { fxTimer_ = 4.f + 10.f * frand() - 4.f * tension_; if (fxTimer_ < 2.f) fxTimer_ = 2.f; fireClang(); }   // industrial more present

  // Risset eternal-accelerando tick streams (precompute per buffer)
  risP_ += (float(nf) * isr) / 22.f; if (risP_ >= 1.f) risP_ -= 1.f;
  float risTempo[RIS], risGain[RIS];
  for (int i = 0; i < RIS; ++i) { float oct = risP_ + float(i) / RIS; oct -= std::floor(oct); risTempo[i] = 0.8f * std::pow(2.f, oct * 4.f); float c = oct * 4.f - 2.f; risGain[i] = std::exp(-0.5f * c * c / (1.3f * 1.3f)); }

  const float gAtk = gco(0.006f, 1, sr_), gRel = gco(0.012f, 1, sr_);   // LATER (GLITCH): master micro-gate one-pole coeffs (constant per buffer; hoisted out of the per-sample loop)
  const float subPulseAmt = cSubPulse_.load(std::memory_order_relaxed); // LOW-END: per-act sub-pedal pulse depth
  const float subLevel    = cSubLevel_.load(std::memory_order_relaxed); //          per-act sub-pedal presence
  const float subRatioTgt = cSubRatio_.load(std::memory_order_relaxed); //          per-phrase low ostinato note
  for (int f = 0; f < nf; ++f) {
    // micro-timing scheduler: fire pending events whose delay elapsed
    for (auto& p : pend_) if (p.used) { if (--p.left <= 0) { trigger(p.e); p.used = false; } }

    float bedL = 0.f, bedR = 0.f, leadL = 0.f, leadR = 0.f, lowMono = 0.f, revSend = 0.f;
    panLFO_ += 0.05f * isr; if (panLFO_ >= 1.f) panLFO_ -= 1.f; float drift = 0.25f * std::sin(kTAU * panLFO_);

    float euPrev = euPhase_;
    euPhase_ += slot * isr;
    if (euPhase_ >= 1.f) { euPhase_ -= 1.f; euStep_ = (euStep_ + 1) % euLen_;
      if (freezeHold_ > 0) --freezeHold_;                          // LATER (GLITCH): reverb-freeze countdown advances per grid step
      if ((euMask_ >> euStep_) & 1u) { fireKick(); fireGlitch(act, whisperOn); }   // LATER (GLITCH): grid-locked edit bus, fired DIRECTLY (bypasses schedule()/pend_)
      float spars = cSparsity_.load(std::memory_order_relaxed);   // Eno loops advance one step per beat
      for (int L = 0; L < NLOOP; ++L) { loopStep_[L] = (loopStep_[L] + 1) % LOOPLEN[L];
        if (loopStep_[L] == 0 && spars < float(NLOOP - L) / float(NLOOP)) fireLoop(L); } }
    if (euPrev < 0.5f && euPhase_ >= 0.5f) fireAnd();     // the off-beat "and"
    euPhase2_ += slot * 0.5f * isr; if (euPhase2_ >= 1.f) { euPhase2_ -= 1.f; euStep2_ = (euStep2_ + 1) % euLen2_; if ((euMask2_ >> euStep2_) & 1u) fireTimp(); }
    wobPhase_ += wobHz * isr; if (wobPhase_ >= 1.f) wobPhase_ -= 1.f;

    // Risset ticks (textural; fired directly so the accelerando stays coherent)
    for (int i = 0; i < RIS; ++i) { risPh_[i] += risTempo[i] * isr; if (risPh_[i] >= 1.f) { risPh_[i] -= 1.f; if (risGain[i] > 0.06f) { int gi = allocCapped(2, 28);
      if (gi >= 0) { Voice& g = vox_[gi]; g = Voice{}; g.on = true; g.layer = 2; g.hz = degHz(modeN_ * 2 + i); g.K = 2; g.pm[0] = 1; g.pm[1] = 2; g.pa[0] = 1; g.pa[1] = 0.3f; g.pnorm = 1.3f; g.amp = 0.022f * risGain[i] * gate_[G_GRAIN]; g.life = 0.012f; g.pan = 0.6f * (2.f * frand() - 1.f); g.grng = rng_ ^ uint32_t(gi * 2246822519u); } } } }

    // sub pedal (centred, dry). In a GROWL section it becomes a dubstep wobble: a resonant SVF
    // whose cutoff + amplitude wobble with wobPhase_, saturated for grit. droneGate pauses it.
    subRatioCur_ += (subRatioTgt - subRatioCur_) * 0.0006f;   // LOW-END: glide between the per-phrase ostinato notes
    subHz_ += (padRoot * 0.5f * subRatioCur_ - subHz_) * 0.0008f; subPhase_ += subHz_ * isr; if (subPhase_ >= 1.f) subPhase_ -= 1.f;
    float subS = std::sin(kTAU * subPhase_) + 0.3f * std::sin(kTAU * subPhase_ * 2.f);
    subLp_ = dn(subLp_ + 0.12f * (subS - subLp_));
    float subOut = subLp_;
    float gact = growl_ * growlDuty_;                    // 0 when the wobble is "off" -> clean drone
    if (gact > 0.01f) {
      float wob = 0.5f - 0.5f * std::cos(kTAU * wobPhase_);
      float ff = 2.f * std::sin(kPI * std::min(90.f + 520.f * wob, float(sr_) / 6.f) / float(sr_));
      growlLp_ = dn(growlLp_ + ff * growlBp_);
      float hp = subS - growlLp_ - 0.18f * growlBp_;
      growlBp_ = dn(growlBp_ + ff * hp);
      float g = std::tanh(2.4f * growlLp_) * (0.4f + 0.6f * wob);
      // formant-shape off the LAST whispered word -> a "talking" growl (the machine voicing the word)
      float fo = 0.f; const float fw[3] = {1.f, 0.8f, 0.5f};
      for (int k = 0; k < 3; ++k) { float y = growlFmtCoef_[k][0] * g - growlFmtCoef_[k][1] * growlFz_[k][0] - growlFmtCoef_[k][2] * growlFz_[k][1];
        growlFz_[k][1] = dn(growlFz_[k][0]); growlFz_[k][0] = dn(y); fo += fw[k] * y; }
      g = 0.55f * g + 1.6f * fo;
      growlOutLp_ = dn(growlOutLp_ + 0.035f * (g - growlOutLp_));   // ~250 Hz one-pole
      float growlHp = g - growlOutLp_;                              // HIGH-PASS: drop the lows off the bass
      subOut = subOut * (1.f - gact) + growlHp * gact * 0.047f;     // wobble down a further ~10 dB
    }
    // LOW-END INTEREST: beat-synced PULSE (full on the kick-grid onset, dipping to (1-depth) between) +
    // per-act PRESENCE -> the sub PEDAL pulses + breathes + moves in pitch, no longer a static sine on the pad.
    float subEnv = (1.f - subPulseAmt) + subPulseAmt * std::exp(-euPhase_ * 5.f);
    lowMono += subOut * subAmp_ * subLevel * subEnv * gate_[G_DRONE];
    // pure mono triangle SUB-BASS — deep (35 Hz and below), sidechain-GATED against the kick
    float ttgt = padRoot * 0.25f; if (ttgt > 35.f) ttgt = 35.f;          // 35 Hz and below
    triHz_ += (ttgt - triHz_) * 0.0004f; triPhase_ += triHz_ * isr; if (triPhase_ >= 1.f) triPhase_ -= 1.f;
    kickDuck_ += (1.f - kickDuck_) * 0.00025f;                            // recover (~90 ms) from the kick duck
    float subPulse = 0.42f + 0.58f * std::exp(-euPhase_ * 6.5f);   // fast ATTACK on each beat, then settle
    lowMono += (4.f * std::fabs(triPhase_ - 0.5f) - 1.f) * 0.45f * kickDuck_ * subPulse * gate_[G_BASS];

    // MOVING JI drone: per-partial slow tremolo + detune drift + vibrato
    for (int p = 0; p < PADN; ++p) {
      padDrift_[p] += (0.013f + 0.007f * p) * isr; if (padDrift_[p] >= 1.f) padDrift_[p] -= 1.f;
      padTrem_[p]  += (0.030f + 0.020f * p) * isr; if (padTrem_[p]  >= 1.f) padTrem_[p]  -= 1.f;
      float ratioMove = 1.f + 0.004f * std::sin(kTAU * padDrift_[p] + p * 1.7f);   // slow chorus drift
      padRatioCur_[p] += (cSpectrum_[p].load(std::memory_order_relaxed) - padRatioCur_[p]) * 0.00008f;   // slow spectral morph
      float tgt = padRoot * padRatioCur_[p] * ratioMove;
      if (tgt > 0.45f * float(sr_)) tgt = 0.45f * float(sr_);            // nyquist guard (inharmonic partials can fly high)
      padHz_[p] += (tgt - padHz_[p]) * 0.0006f;
      padVib_[p] += (0.10f + 0.06f * p) * isr; if (padVib_[p] >= 1.f) padVib_[p] -= 1.f;
      float vib = 1.f + 0.012f * std::sin(kTAU * padVib_[p] + p);
      padPhase_[p] += padHz_[p] * vib * isr; if (padPhase_[p] >= 1.f) padPhase_[p] -= std::floor(padPhase_[p]);
      float det = cBeat_.load(std::memory_order_relaxed) * padRatioCur_[p] * 0.5f;     // difference-tone detune
      padBeatPh_[p] += padHz_[p] * (1.f + det) * vib * isr; if (padBeatPh_[p] >= 1.f) padBeatPh_[p] -= std::floor(padBeatPh_[p]);
      float sg = 0.45f * std::sin(kTAU * padPhase_[p]) + 0.45f * std::sin(kTAU * padBeatPh_[p]) + 0.20f * std::sin(kTAU * padPhase_[p] * 2.f);
      padLp_[p] = dn(padLp_[p] + (0.12f + 0.40f * depth) * (sg - padLp_[p]));
      padHp_[p] = dn(padHp_[p] + 0.025f * (padLp_[p] - padHp_[p]));   // track lows (~180 Hz)
      float sigHp = padLp_[p] - padHp_[p];                            // HIGH-PASS the drone to clear room for the sub-bass
      float trem = 0.72f + 0.28f * std::sin(kTAU * padTrem_[p]);                    // slow amplitude swell
      float topFade = 1.f - cTopCut_.load(std::memory_order_relaxed) * 0.85f * float(p) / float(PADN);   // V thins the upper partials
      float padFloor = 0.55f + 0.45f * emerge_;   // 2c: the HARMONIC pad blooms IN as the image forms (never fully dies -> drone stays)
      float tgtAmp = ((p < 2) ? 0.026f : 0.026f * padBloom_) * trem * topFade * padFloor; padAmp_[p] += (tgtAmp - padAmp_[p]) * 0.002f;   // 0.026 (8 partials; spectrum audible)
      float a = padAmp_[p] * sigHp * gate_[G_DRONE];       // drone drops out independently
      float pan = -0.78f + 1.56f * float(p) / float(PADN - 1); pan += 0.12f * drift; float pp = 0.5f * (pan + 1.f);   // spread the 8 partials across the field
      bedL += a * std::cos(pp * 1.5707963f); bedR += a * std::sin(pp * 1.5707963f); revSend += a * 0.8f;
    }
    beatPhase_ += padRoot * fifthB * isr; if (beatPhase_ >= 1.f) beatPhase_ -= std::floor(beatPhase_);
    { float a = padAmp_[2] * 0.5f * std::sin(kTAU * beatPhase_); bedL += a * 0.3f; bedR -= a * 0.3f; revSend += a * 0.4f; }

    // Shepard tone as FILTERED WHITE NOISE: octave-spaced resonant band-passes on white noise,
    // Gaussian-windowed in log-frequency and sweeping forever -> an endless rising NOISE bed
    // (the moving filter IS the Shepard tone).
    shepPhase_ += shepRate_ * isr; if (shepPhase_ >= 1.f) shepPhase_ -= 1.f; if (shepPhase_ < 0.f) shepPhase_ += 1.f;
    float nzL = frand() * 2.f - 1.f, nzR = frand() * 2.f - 1.f;   // independent L/R noise -> true STEREO
    float shL = 0.f, shR = 0.f, shn = 0.f;
    for (int i = 0; i < SHEP; ++i) {
      float oct = shepPhase_ + float(i) / SHEP; oct -= std::floor(oct);
      float fi = 32.7f * std::pow(2.f, oct * 6.f), lf = std::log2(fi);
      float a = std::exp(-0.5f * (lf - std::log2(300.f)) * (lf - std::log2(300.f)) / (1.4f * 1.4f));
      float ff = 2.f * std::sin(kPI * std::min(fi, float(sr_) / 6.f) / float(sr_)); const float q = 0.6f;   // WIDE/soft bands -> no whistle
      shepPh_[i]  = dn(shepPh_[i]  + ff * shepBp_[i]);  float hpL = nzL - shepPh_[i]  - q * shepBp_[i];  shepBp_[i]  = dn(shepBp_[i]  + ff * hpL); shL += a * shepBp_[i];
      shepPhR_[i] = dn(shepPhR_[i] + ff * shepBpR_[i]); float hpR = nzR - shepPhR_[i] - q * shepBpR_[i]; shepBpR_[i] = dn(shepBpR_[i] + ff * hpR); shR += a * shepBpR_[i];
      shn += a;
    }
    float gn = shepGain_ * 3.4f / (shn > 0 ? shn : 1.f);
    float bedairL = shL * gn, bedairR = shR * gn;
    bedL += bedairL; bedR += bedairR;                            // full STEREO (no mono)
    revSend += (bedairL + bedairR) * 0.30f;                      // ~85% dry / 15% wet

    // grain cloud (Poisson)
    grainTimer_ -= isr;
    if (grainTimer_ <= 0.f) { float lam = 2.f + cCloudLam_.load(std::memory_order_relaxed); grainTimer_ += -std::log(std::max(1e-6f, frand())) / lam;
      int gi = allocCapped(2, 28); if (gi >= 0) { Voice& g = vox_[gi]; g = Voice{}; g.on = true; g.layer = 2;
        float spread = cCloudSpread_.load(std::memory_order_relaxed), quant = cQuant_.load(std::memory_order_relaxed);
        float span = 1.f + 4.f * spread, base = degHz(nodeDegree(int(rng_ & 2047), -1));
        g.hz = (quant > frand()) ? base * std::pow(2.f, float(int(frand() * span) - int(span * 0.5f)))
                                 : base * std::pow(2.f, (frand() * 2.f - 1.f) * span);   // continuous glissando = Xenakis cloud
        g.K = 2; g.pm[0] = 1; g.pm[1] = 2; g.pa[0] = 1; g.pa[1] = 0.25f + 0.4f * frand(); g.pnorm = 1.f + g.pa[1]; g.amp = (0.022f + 0.04f * act * (0.5f + frand())) * gate_[G_GRAIN]; g.life = 0.02f + 0.18f * frand(); g.pan = 0.9f * (2.f * frand() - 1.f); g.grng = rng_ ^ (uint32_t(gi) * 40503u); } }

    // ARBORESCENCES (LATER, Step 3): Metastaseis branching of the grain cloud. A 2nd Poisson clock fires a
    // BURST of N=4..8 layer-2 grains in one frame, all sharing a TRUNK pitch seeded from the worker register
    // (cStoryDeg_). Each grain fans +-octHalf around the trunk and GLIDES (tslot=-9 -> a constant-coef hz
    // ramp in tickVoice) either INTO the trunk (convergent / capture) or OUT to its endpoint (divergent /
    // fan-out), a coin choosing per burst so III breathes between intake and output -> reads as branching, not
    // random spray. Shares the layer-2 allocCapped(2,28) pool (no new voices beyond the 28-cap; oldest-steal).
    arborTimer_ -= isr; float arb = cArbor_.load(std::memory_order_relaxed);
    if (arb > 0.05f) {
      while (arborTimer_ <= 0.f) {                                              // re-arm ONLY on expiry (Poisson); multiple bursts can fire in one buffer
        arborTimer_ += -std::log(std::max(1e-6f, frand())) / (2.f + 10.f * arb);   // ~2 fans/s hairline -> ~12 peak
        int N = 4 + int(4.f * arb + 0.5f); if (N < 4) N = 4; else if (N > 8) N = 8;
        int jit = int(rng_ % 5u) - 2;                                           // trunk degree jitter (-2..+2)
        float trunk = degHz(int(cStoryDeg_.load(std::memory_order_relaxed)) + jit);
        float octHalf = 0.5f + 3.5f * cCloudSpread_.load(std::memory_order_relaxed);   // half-width in octaves (SP[3]=0.9 -> ~3.5 oct)
        bool convergent = frand() < 0.5f;                                       // coin: this fan converges (capture) or diverges (fan-out)
        for (int b = 0; b < N; ++b) {
          int gi = allocCapped(2, 28); if (gi < 0) break;
          Voice& g = vox_[gi]; g = Voice{};
          float off = octHalf * (float(b) / float(N - 1) * 2.f - 1.f);          // fan offset across [-octHalf,+octHalf]
          float endHz = trunk * std::pow(2.f, off);
          g.on = true; g.layer = 2; g.tslot = -9;                              // sentinel -9 -> the arbor glide branch
          g.K = 2; g.pm[0] = 1; g.pm[1] = 2; g.pa[0] = 1; g.pa[1] = 0.25f + 0.4f * frand(); g.pnorm = 1.f + g.pa[1];
          if (convergent) { g.hz = endHz; }                                     // convergent: start AT the endpoint, glide to trunk
          else            { g.hz = trunk; }                                     // divergent:  start at the trunk, glide OUT
          g.amp = (0.018f + 0.03f * arb) / std::sqrt(float(N)) * gate_[G_GRAIN];   // energy held against the longer life
          g.life = 0.25f + 0.55f * frand();                                     // audible sweep (vs cloud's 0.02..0.18)
          g.pan = 0.9f * (2.f * frand() - 1.f);
          g.grng = rng_ ^ (uint32_t(gi) * 2246822519u);
          g.body = convergent ? trunk : endHz;                                  // MUST set target AFTER g=Voice{} (default 0.6 else glides to garbage)
        }
      }
    }

    // CC0 SAMPLE CHOIR: long, slow-attack, reverb-washed grains drawn from the sample bank (US: the comfortable
    // background built atop their voice). Era biases the timbre family. tslot=-7 routes it to the washed tier.
    choirTimer_ -= isr; float clam = cChoirLam_.load(std::memory_order_relaxed);
    int choirN = 0; for (auto& v : vox_) if (v.on && v.layer == 2 && v.tslot == -7) ++choirN;   // STEP 5 / rule 3: choir census
    if (clam > 0.05f && choirTimer_ <= 0.f && choirN < 8) { choirTimer_ += -std::log(std::max(1e-6f, frand())) / clam;   // choir stays <=8 inside the shared layer-2 28-cap
      int gi = allocCapped(2, 28); if (gi >= 0) { Voice& g = vox_[gi]; g = Voice{};
        Role& R = (frand() < 0.6f) ? rTrace_ : rBass_;
        int sidx = pickSample(R, degHz(nodeDegree(int(rng_ & 2047), -1)));
        if (sidx >= 0) { const Sample& smp = samples_[sidx];
          g.on = true; g.layer = 2; g.tslot = -7;
          g.smp = smp.data.data(); g.smpLen = int(smp.data.size()); g.smpPos = 0.f;
          g.smpRate = (smp.srcSR / float(sr_)) * (smp.pitched ? std::pow(2.f, float((int(frand() * 3) - 1) * 12) / 12.f) : 1.f);
          g.amp = 0.05f * cChoir_.load(std::memory_order_relaxed); g.life = 1.5f + 2.5f * frand(); g.atk = 0.3f + 0.5f * frand();
          g.phase = frand(); g.pan = 0.9f * (2.f * frand() - 1.f);
        } } }

    // GRANULAR SHRED of the whisper: grains read the lead-capture ring (the machine chewing the human voice)
    granTimer_ -= isr; float gdv = cGranDens_.load(std::memory_order_relaxed);
    if (gdv > 0.5f && granTimer_ <= 0.f) { granTimer_ += -std::log(std::max(1e-6f, frand())) / gdv;
      int gi = allocCapped(2, 28); if (gi >= 0) { Voice& g = vox_[gi]; g = Voice{};
        float off = float(capPos_) - (cGranScat_.load(std::memory_order_relaxed) * (1.f - emerge_) + cGranSmear_.load(std::memory_order_relaxed)) * frand() * float(CAPN) * 0.9f;
        while (off < 0.f) off += float(CAPN);
        bool rev = frand() < cGranRev_.load(std::memory_order_relaxed);
        float semis = (2.f * frand() - 1.f) * cGranPitch_.load(std::memory_order_relaxed);
        g.on = true; g.layer = 2; g.smp = cap_.data(); g.smpLen = CAPN; g.smpPos = off;
        g.smpRate = std::pow(2.f, semis / 12.f) * (rev ? -1.f : 1.f);
        g.amp = 0.085f * gate_[G_GRAIN] * gate_[G_WHISPER];   // the voice-shred sits clearly in the mix
        g.life = cGranSize_.load(std::memory_order_relaxed); g.atk = g.life * 0.3f;
        g.pan = 0.9f * (2.f * frand() - 1.f); g.grng = rng_ ^ uint32_t(gi * 2654435761u); } }

    // pooled voices
    float ppSendL = 0.f, ppSendR = 0.f, sampSend = 0.f;
    for (auto& v : vox_) {
      if (!v.on) continue; float s = tickVoice(v, isr, depth, moodLP_);
      if (v.layer == 0) s *= gate_[G_MEL]; else if (v.layer == 1) s *= gate_[G_BELL]; else if (v.layer == 7) s *= gate_[G_TIMP];   // per-element dropouts
      if (v.layer == 4) {                                  // whisper: lead (gated), and into the delay
        float pp = 0.5f * (v.pan + drift + 1.f); pp = pp < 0 ? 0 : (pp > 1 ? 1 : pp);
        float cL = std::cos(pp * 1.5707963f), cR = std::sin(pp * 1.5707963f);
        leadL += gate_[G_WHISPER] * 1.0f * s * cL; leadR += gate_[G_WHISPER] * 1.0f * s * cR;            // voices a bit louder (0.85 -> 1.0, ~+1.4 dB)
        ppSendL += gate_[G_WHISPER] * 0.12f * s * cL; ppSendR += gate_[G_WHISPER] * 0.12f * s * cR; revSend += s * gate_[G_WHISPER] * 0.7f;   // blend more with the global verb
        sampSend += s * gate_[G_WHISPER] * 0.6f;                                                     // short MONO reverb on the whisper
      } else if (v.layer == 9) { leadL += 1.05f * s; leadR += 1.05f * s; ppSendL += 0.4f * s; ppSendR += 0.4f * s; }   // "and" tick: dry + into the ping-pong delay
      else if (v.layer == 6 || v.layer == 7) { lowMono += s; }
      else {
        float pan = v.pan + (v.layer == 2 || v.layer == 8 ? drift : 0.f); float pp = 0.5f * (pan + 1.f); pp = pp < 0 ? 0 : (pp > 1 ? 1 : pp);
        float cL = std::cos(pp * 1.5707963f), cR = std::sin(pp * 1.5707963f);
        bool up = (v.layer <= 2 || v.layer == 8);            // upper part: pluck/bell/grain/clang
        float dryg = up ? 0.72f : 1.f, rv = up ? 1.25f : 0.5f;   // pushed back + further into reverb
        if (v.smp == cap_.data()) { dryg = 1.2f; rv = 0.8f; }                    // the voice-shred grains stay present
        else if (v.tslot == -7) { dryg = 0.22f; rv = 2.8f; sampSend += s * 1.2f; }  // CHOIR: deep background wash
        else if (v.layer == 10) { dryg = 0.95f; rv = 0.6f; ppSendL += 0.2f * s * cL; ppSendR += 0.2f * s * cR; }  // LATER (GENDYN): dry-and-forward soloist; roughness stays legible (little reverb)
        else if (v.smp) { dryg = 0.38f; rv = 2.1f; }                             // other CC0 samples -> pushed back
        bedL += dryg * s * cL; bedR += dryg * s * cR; revSend += s * rv;
        if (v.layer == 2) { ppSendL += 0.55f * s * cL; ppSendR += 0.55f * s * cR; }   // grains -> delay
        if (v.smp) sampSend += s;                                                     // CC0 samples -> short reverb
      }
    }
    // ping-pong delay (grains + whispers bounce L<->R)
    { float dL = ppL_[ppPos_], dR = ppR_[ppPos_];
      ppL_[ppPos_] = dn(ppSendL + dR * 0.42f); ppR_[ppPos_] = dn(ppSendR + dL * 0.42f); ppPos_ = (ppPos_ + 1) % ppN_;
      bedL += dL * 0.32f; bedR += dR * 0.32f; }
    // short MONO reverb to blend the CC0 samples (pluck/bell/trace/organ)
    { float sw1 = 0.f, sw2 = 0.f; sampRev_(sampSend * 0.85f, sw1, sw2, 0.78f); bedL += sw1 * 0.85f; bedR += sw2 * 0.85f; }   // samples washed into the background

    cap_[capPos_] = leadL + leadR; capPos_ = (capPos_ + 1) & (CAPN - 1);   // capture the whisper lead for granulation
    capRms_ += (std::fabs(leadL + leadR) - capRms_) * 0.0008f;             // LATER (GLITCH): silence guard one-pole on cap_ writes
    revSend *= (1.f - 0.9f * freeze_);                                     // IV freeze: cut the reverb FEED; its tail rings on
    float w1 = 0.f, w2 = 0.f; reverb_(revSend * 0.5f, w1, w2, 0.6f);
    float L = busyGain * cut_ * (duckGain_ * bedL + leadL + lowDuck_ * lowMono + reverbWet_ * w1);   // cut_ = IV structural silence
    float R = busyGain * cut_ * (duckGain_ * bedR + leadR + lowDuck_ * lowMono + reverbWet_ * w2);
    masterLoL_ = dn(masterLoL_ + aLo * (L - masterLoL_)); L += 2.16f * masterLoL_;
    masterLoR_ = dn(masterLoR_ + aLo * (R - masterLoR_)); R += 2.16f * masterLoR_;
    masterHiL_ = dn(masterHiL_ + aHi * (L - masterHiL_)); L += 0.45f * (masterHiL_ - L);   // pink HF tilt
    masterHiR_ = dn(masterHiR_ + aHi * (R - masterHiR_)); R += 0.45f * (masterHiR_ - R);
    // 2c GLITCH: bitcrush + stutter the master inside an armed window (image resolving OUT of glitch).
    // Capture the clean master into the stutter ring every frame; when armed, blend in the looped+crushed window.
    // All fixed arrays, no alloc; applied AFTER the EQ tilt, BEFORE the tanh so the ceiling still bounds it.
    stutBufL_[stutWr_] = L; stutBufR_[stutWr_] = R; stutWr_ = (stutWr_ + 1) & (STUTN - 1);
    if (glitchT_ > 0) {
      float gx = stutBufL_[stutRead_], gy = stutBufR_[stutRead_];        // stutter: replay the looped tail
      stutRead_ = (stutRead_ + 1) & (STUTN - 1);
      if (--glitchT_ % stutPeriod_ == 0) stutRead_ = (stutWr_ - stutPeriod_ + STUTN) & (STUTN - 1);
      if (crushAmt_ > 0.001f) {                                          // bitcrush: decimate + quantize
        if (crushPhase_ == 0) { float q = 8.f + 56.f * (1.f - crushAmt_);
          crushHoldL_ = std::floor(gx * q + 0.5f) / q; crushHoldR_ = std::floor(gy * q + 0.5f) / q; }
        crushPhase_ = (crushPhase_ + 1) % crushStride_;
        gx = crushHoldL_ + (gx - crushHoldL_) * (1.f - crushAmt_);       // dry/wet by crush depth
        gy = crushHoldR_ + (gy - crushHoldR_) * (1.f - crushAmt_);
      }
      float gmix = float(glitchT_) / float(glitchLen_ > 0 ? glitchLen_ : 1);   // window fades out -> click-free release
      L = L + (gx - L) * gmix; R = R + (gy - R) * gmix;
    }
    // LATER (GLITCH AXIS, Step 4): the CONTINUOUS bitcrush (islot 1) + master micro-gate (islot 5), applied just
    // before the master tanh (which soft-clips any requantization spike). Partitioned from Phase-2's islot-0 window
    // above: the Step-4 crush is SKIPPED while that window is active (glitchT_>0) so the two crush domains never
    // fight over crushHoldL_/crushHoldR_. crushBits_ is the per-sample armed COUNTDOWN; crushPh_ sample-and-holds at
    // a crush-rate < sr_; the bit-depth is read live from cCrush_ + cCostBeat_ (cheap wage -> most-destroyed crush).
    if (glitchT_ <= 0 && crushBits_ > 0) {
      --crushBits_;
      float crush = clamp01(cCrush_.load(std::memory_order_relaxed));
      if (crush > 0.001f) {
        crushPh_ += (1.f - 0.92f * crush);                          // hold-rate: full crush -> ~one S&H per ~12 samples
        if (crushPh_ >= 1.f) { crushPh_ -= std::floor(crushPh_);
          float bits = 12.f - 8.f * crush - 2.f * cCostBeat_.load(std::memory_order_relaxed); if (bits < 2.f) bits = 2.f; if (bits > 12.f) bits = 12.f;
          float q = std::pow(2.f, bits);
          crushHoldL_ = std::floor(L * q + 0.5f) / q; crushHoldR_ = std::floor(R * q + 0.5f) / q; }
        L = crushHoldL_ + (L - crushHoldL_) * (1.f - crush);        // dry/wet by crush depth (click-free at edges)
        R = crushHoldR_ + (R - crushHoldR_) * (1.f - crush);
      }
    }
    microGate_ += (microGateTgt_ - microGate_) * (microGateTgt_ < microGate_ ? gAtk : gRel);   // attack faster than release (no pop)
    L *= microGate_; R *= microGate_;                               // islot 5: master micro-gate
    io.out(0, f) = std::tanh(L * master); if (nch > 1) io.out(1, f) = std::tanh(R * master);
    for (int c = 2; c < nch; ++c) io.out(c, f) = 0.f;
  }
  lastDepth_ = depth;
}

}  // namespace wosw
