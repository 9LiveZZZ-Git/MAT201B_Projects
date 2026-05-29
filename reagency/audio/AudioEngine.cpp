#include "audio/AudioEngine.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <dirent.h>

namespace wosw {

static constexpr float kPI  = 3.14159265358979f;
static constexpr float kTAU = 6.28318530717959f;
static const float kJI[8] = {1.f, 16.f / 15.f, 6.f / 5.f, 5.f / 4.f, 4.f / 3.f, 3.f / 2.f, 7.f / 4.f, 2.f};

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
  auto add = [](Role& r, int idx) { if (r.n < 24) r.idx[r.n++] = idx; };
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
      else if (has("psaltery") || has("bowed") || has("cello") || has("violin") || has("viola") || has("choir") || has("voice") || has("clarinet")) add(rTrace_, idx);
      else add(rPluck_, idx);
    }
    closedir(d);
    if (!samples_.empty()) { std::fprintf(stderr, "[wosw audio] %zu CC0 samples from %s/ (bass=%d pluck=%d bell=%d trace=%d timp=%d metal=%d)\n",
                                          samples_.size(), b.c_str(), rBass_.n, rPluck_.n, rBell_.n, rTrace_.n, rTimp_.n, rMetal_.n); return; }
  }
  std::fprintf(stderr, "[wosw audio] no assets/audio samples — fully procedural synthesis\n");
}
int AudioEngine::pickSample(Role& r, float hz) {
  if (r.n == 0) return -1;
  float best = 1e9f;
  for (int i = 0; i < r.n; ++i) { const Sample& s = samples_[r.idx[i]]; float dd = s.pitched ? std::fabs(std::log2(s.rootHz / hz)) : 0.f; if (dd < best) best = dd; }
  int cand[24], nc = 0;
  for (int i = 0; i < r.n; ++i) { const Sample& s = samples_[r.idx[i]]; float dd = s.pitched ? std::fabs(std::log2(s.rootHz / hz)) : 0.f; if (dd <= best + 0.34f && nc < 24) cand[nc++] = r.idx[i]; }
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
float AudioEngine::degHz(int degree) const { int idx = degree % modeN_, oct = degree / modeN_; if (idx < 0) { idx += modeN_; oct -= 1; } return root_ * std::pow(2.f, (modeBuf_[idx] + 12 * oct) / 12.f); }
int AudioEngine::nodeDegree(int node, int cluster) const { unsigned h = unsigned(node) * 2654435761u; int d = int(h % unsigned(modeN_ * 3)); if (cluster > 0 && (cluster % 2 == 1)) d += modeN_; return d; }
float AudioEngine::jiBassHz() const { return cClusterRoot_.load(std::memory_order_relaxed) * 0.5f; }

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

void AudioEngine::init(const std::string& assetDir, double sampleRate) {
  sr_ = sampleRate > 0 ? sampleRate : 44100.0;
  loadManifest(assetDir); loadSamples(assetDir); loadWords(assetDir);
  reverb_.bandwidth(0.9f); reverb_.damping(0.45f); reverb_.decay(0.9f);
  sampRev_.bandwidth(0.9f); sampRev_.damping(0.55f); sampRev_.decay(0.42f);   // short room for the samples
  cClusterRoot_.store(root_, std::memory_order_relaxed); subHz_ = root_ * 0.25f;
  for (int p = 0; p < PADN; ++p) padHz_[p] = root_ * kJI[p == 0 ? 0 : (p == 1 ? 2 : (p == 2 ? 5 : 7))];
  ppN_ = int(sr_ * 0.55); if (ppN_ < 64) ppN_ = 64;    // ping-pong delay buffers (allocated once)
  ppL_.assign(ppN_, 0.f); ppR_.assign(ppN_, 0.f); ppPos_ = 0;
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
void AudioEngine::whisper(const std::string& word, int node, float pan) {
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
  Ev e{}; e.kind = EV_WHISPER; e.layer = 4; e.amp = 0.48f; e.pan = pan; e.islot = syl; e.vow = vow; e.a = f0; e.b = f1; e.c = f2; push(e);
}
void AudioEngine::update(float dt, float hesitation, float depth, float progress, float focusPan) {
  if (!ready()) return; depth = clamp01(depth);
  simHes_ = hesitation; simDepth_ = depth;
  cHesitation_.store(hesitation, std::memory_order_relaxed); cDepth_.store(depth, std::memory_order_relaxed);
  cFocusPan_.store(focusPan < -1 ? -1 : (focusPan > 1 ? 1 : focusPan), std::memory_order_relaxed);
  activity_ *= std::pow(0.5f, dt / 0.8f); cActivity_.store(activity_, std::memory_order_relaxed); (void)progress;

  // mood scheduler on FIBONACCI-second intervals
  static const float FIB[6] = {8.f, 13.f, 21.f, 34.f, 55.f, 89.f};
  auto sr = [&]() { sprng_ ^= sprng_ << 13; sprng_ ^= sprng_ >> 17; sprng_ ^= sprng_ << 5; return float(sprng_) / 4294967296.f; };
  moodTimer_ -= dt;
  if (moodTimer_ <= 0.f) {
    moodTimer_ = FIB[int(sr() * 6.f) % 6]; mbTgt_ = sr(); mdTgt_ = sr(); mwTgt_ = sr();
    // arrangement SECTION (FULL most common): 0 full, 1 drone-pause, 2 growl, 3 sparse, 4 melody
    float r = sr(); section_ = (r < 0.42f) ? 0 : (r < 0.56f) ? 1 : (r < 0.70f) ? 2 : (r < 0.84f) ? 3 : 4;
    float dG = 1.f, wG = 1.f, gG = 1.f, grw = 0.f;
    if (section_ == 1) dG = 0.f;                                  // the drone pauses
    else if (section_ == 2) grw = 1.f;                            // dubstep growl on the low end
    else if (section_ == 3) { dG = 0.10f; wG = 0.20f; gG = 0.12f; }  // ensemble drops to near-silence
    else if (section_ == 4) { gG = 0.22f; melodyOn_ = true; melNote_ = 0; melTune_ = int(sr() * 3.f) % 3; melNoteTimer_ = 0.f; }
    if (section_ != 4) melodyOn_ = false;
    cDroneGate_.store(dG, std::memory_order_relaxed); cWhisperGate_.store(wG, std::memory_order_relaxed);
    cGrainGate_.store(gG, std::memory_order_relaxed); cGrowl_.store(grw, std::memory_order_relaxed);
  }
  float mc = std::min(1.f, dt * 0.25f); moodBright_ += (mbTgt_ - moodBright_) * mc; moodDens_ += (mdTgt_ - moodDens_) * mc; moodWet_ += (mwTgt_ - moodWet_) * mc;
  cMoodBright_.store(moodBright_, std::memory_order_relaxed); cMoodDens_.store(moodDens_, std::memory_order_relaxed); cMoodWet_.store(moodWet_, std::memory_order_relaxed);
  // recognizable PD melody: emit the next note ~once per beat; grains gated down so it emerges
  if (melodyOn_) {
    melNoteTimer_ -= dt;
    if (melNoteTimer_ <= 0.f) {
      const Tune& t = TUNES[melTune_ % 3];
      if (melNote_ >= t.n) { melodyOn_ = false; cGrainGate_.store(1.f, std::memory_order_relaxed); }
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
  int i; if (e.layer == 0 || e.layer == 1) i = allocCapped(e.layer, 4); else i = allocVoice(); if (i < 0) return;
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
              if (s >= 0) setSample(s); else { v.K = 5; float mr[5] = {1.f, 2.76f, 5.40f, 8.93f, 13.3f}; v.pnorm = 0.f; for (int k = 0; k < 5; ++k) { v.pm[k] = mr[k]; v.pa[k] = 1.f / std::pow(k + 1.f, 0.8f); v.pnorm += v.pa[k]; } } break; }
    case 4: { v.syl = e.islot < 1 ? 1 : (e.islot > 5 ? 5 : e.islot); v.life = 1.5f + 0.55f * v.syl; v.body = 0.62f; v.atk = 0.09f; v.curSyl = -1;
              for (int i = 0; i < 5; ++i) v.vowels[i] = (e.vow >> (3 * i)) & 7; break; }   // formants set per-syllable in tickVoice
    default: v.life = 0.15f; v.atk = 0.01f; setPartials(2, 0.f); break;
  }
}

float AudioEngine::tickVoice(Voice& v, float isr, float depth, float bright) {
  v.age += isr; if (v.age >= v.life) { v.on = false; return 0.f; } const float u = v.age / v.life;
  if (v.layer == 6) { float fk = 36.f + 80.f * std::exp(-v.age / 0.032f); v.phase += fk * isr; if (v.phase >= 1.f) v.phase -= 1.f;
    float click = (v.age < 0.004f) ? (1.f - v.age / 0.004f) * (vrand(v.grng) * 2.f - 1.f) * 0.6f : 0.f; return (std::sin(kTAU * v.phase) * std::exp(-v.age / 0.10f) + click) * v.amp; }
  if (v.layer == 9) { float n = vrand(v.grng) * 2.f - 1.f; v.lp = dn(v.lp + 0.5f * (n - v.lp)); return (n - v.lp) * std::exp(-v.age / 0.0035f) * v.amp; }   // off-beat "and" click (high-passed)
  if (v.smp) { if (v.smpPos >= v.smpLen - 2) { v.on = false; return 0.f; } int i0 = int(v.smpPos); float fr = v.smpPos - i0; float s = v.smp[i0]*(1.f-fr) + v.smp[i0+1]*fr; v.smpPos += v.smpRate;
    float env; if (v.layer == 3 || v.layer == 5) env = (u < (v.atk/v.life)) ? (u/std::max(1e-4f, v.atk/v.life)) : (u < 0.6f ? 1.f : std::max(0.f, 1.f-(u-0.6f)/0.4f));
    else { float a = v.atk/v.life; env = (u < a) ? (u/std::max(1e-4f,a)) : (u > 0.85f ? (1.f-(u-0.85f)/0.15f) : 1.f); } return s * env * v.amp; }
  if (v.layer == 4) {
    // step the formant resonators through the word's vowels (speech-like synthesis)
    int sk = (u < v.body) ? int((u / v.body) * float(v.syl)) : (v.syl - 1); if (sk < 0) sk = 0; if (sk >= v.syl) sk = v.syl - 1;
    if (sk != v.curSyl) { v.curSyl = sk; formantForIdx(v.vowels[sk], v.fmt[0], v.fmt[1], v.fmt[2]);
      const float bw[3] = {90.f, 110.f, 140.f}; for (int k = 0; k < 3; ++k) { float r = std::exp(-kPI*bw[k]/float(sr_)); v.fcoef[k][0]=(1.f-r); v.fcoef[k][1]=-2.f*r*std::cos(kTAU*v.fmt[k]/float(sr_)); v.fcoef[k][2]=r*r; } }
    float env, cons = 0.f;
    if (u < v.body) { float t = u/v.body, ph = t*float(v.syl), frac = ph-std::floor(ph); env = std::pow(0.5f-0.5f*std::cos(kTAU*frac), 0.6f)*std::sin(kPI*t);
      // consonant ONSET: a brief high-passed noise burst (fricative) at each syllable start -> reads as speech
      float cnz = vrand(v.grng)*2.f-1.f; v.lp = dn(v.lp + 0.55f*(cnz - v.lp)); float chp = cnz - v.lp;
      if (frac < 0.16f) cons = chp * (1.f - frac/0.16f) * 0.6f * std::sin(kPI*t); }
    else { float t = (u-v.body)/(1.f-v.body), gp = t*(6.f+10.f*t), gf = gp-std::floor(gp), duty = 0.6f*(1.f-t); float grain = (gf<duty)?(0.5f-0.5f*std::cos(kTAU*gf/std::max(1e-3f,duty))):0.f; env = grain*(1.f-t); }
    float nz = vrand(v.grng)*2.f-1.f, s = 0.f; const float w[3] = {1.f, 0.7f, 0.45f};
    for (int k = 0; k < 3; ++k) { float y = v.fcoef[k][0]*nz - v.fcoef[k][1]*v.fz_[k][0] - v.fcoef[k][2]*v.fz_[k][1]; v.fz_[k][1]=dn(v.fz_[k][0]); v.fz_[k][0]=dn(y); s += w[k]*y; } return (s * env + cons) * v.amp * 1.25f; }
  v.phase += v.hz * isr; if (v.phase >= 1.f) v.phase -= std::floor(v.phase);
  float s = 0.f; const float nyq = 0.45f * float(sr_);
  for (int k = 0; k < v.K; ++k) { float pf = v.hz * v.pm[k]; if (pf >= nyq) break; s += v.pa[k] * std::sin(kTAU * v.phase * v.pm[k]); } s /= v.pnorm;
  float env; if (v.layer == 2) { float c = u*2.f-1.f; env = std::exp(-c*c*6.f); }
  else if (v.layer == 3) env = (u < 0.12f) ? (u/0.12f) : (u < 0.55f ? 1.f : std::max(0.f, 1.f-(u-0.55f)/0.45f));
  else { float a = v.atk/v.life; env = (u < a) ? (u/std::max(1e-4f,a)) : std::exp(-3.5f*(u-a)/(1.f-a)); }
  float out = s * env * v.amp; float cut = (v.layer == 1 || v.layer == 8) ? (0.55f + 0.40f*bright) : (0.22f + 0.45f*bright);
  v.lp = dn(v.lp + cut * (out - v.lp)); return v.lp;
}

void AudioEngine::fireKick() { Ev e{}; e.kind = EV_NOTE; e.layer = 6; e.hz = jiBassHz(); e.amp = (euStep_ == 0) ? 0.55f : 0.38f; e.pan = 0.f; schedule(e); }
void AudioEngine::fireTimp() { timpDeg_ = (timpDeg_ + 1 + int(frand() * 2.f)) % (modeN_ * 2); Ev e{}; e.kind = EV_NOTE; e.layer = 7; e.hz = degHz(timpDeg_) * 0.5f; e.amp = 0.24f; e.pan = 0.18f * (2.f * frand() - 1.f); schedule(e); }
void AudioEngine::fireClang() { Ev e{}; e.kind = EV_NOTE; e.layer = 8; e.hz = 60.f + 200.f * frand(); e.amp = 0.14f + 0.10f * frand(); e.pan = 0.8f * (2.f * frand() - 1.f); schedule(e); }
void AudioEngine::fireAnd() { int i = allocCapped(9, 3); if (i < 0) return; Voice& v = vox_[i]; v = Voice{}; v.on = true; v.layer = 9; v.life = 0.05f; v.age = 0.f; v.amp = 0.11f; v.pan = 0.1f * (2.f * frand() - 1.f); v.grng = rng_ ^ uint32_t(i * 668265263u); }

void AudioEngine::render(al::AudioIOData& io) {
  const int nf = int(io.framesPerBuffer()), nch = int(io.channelsOut());
  if (!ready()) { for (int c = 0; c < nch; ++c) for (int f = 0; f < nf; ++f) io.out(c, f) = 0.f; return; }

  Ev e; while (pop(e)) { if (e.kind == EV_RHYTHM || e.kind == EV_RHYTHM2 || e.kind == EV_TRACE_OFF) trigger(e); else schedule(e); }

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
  shepRateTgt_ = (hes > 0.55f ? -1.f : 1.f) * (1.f / (80.f - 40.f * hes)); shepRate_ += (shepRateTgt_ - shepRate_) * gco(8.f, nf, sr_);   // near-static (~80 s/cycle)
  bool whisperOn = false; for (auto& v : vox_) if (v.on && v.layer == 4) { whisperOn = true; break; }
  duckGain_ += ((whisperOn ? 0.6f : 1.f) - duckGain_) * gco(0.25f, nf, sr_);
  lowDuck_  += ((whisperOn ? 0.8f : 1.f) - lowDuck_) * gco(0.25f, nf, sr_);
  // arrangement gates: drone pause / ensemble drop-outs / dubstep growl (glided, no clicks)
  droneGate_   += (cDroneGate_.load(std::memory_order_relaxed)   - droneGate_)   * gco(0.5f, nf, sr_);
  whisperGate_ += (cWhisperGate_.load(std::memory_order_relaxed) - whisperGate_) * gco(0.4f, nf, sr_);
  grainGate_   += (cGrainGate_.load(std::memory_order_relaxed)   - grainGate_)   * gco(0.5f, nf, sr_);
  growl_       += (cGrowl_.load(std::memory_order_relaxed)       - growl_)       * gco(0.4f, nf, sr_);

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
  const float padRatio[PADN] = {1.f, kJI[2], 1.5f, 2.f};

  // industrial-clang scheduler (sparse, "in places"; a touch more likely when tense)
  fxTimer_ -= float(nf) * isr; if (fxTimer_ <= 0.f) { fxTimer_ = 7.f + 18.f * frand() - 6.f * tension_; fireClang(); }

  // Risset eternal-accelerando tick streams (precompute per buffer)
  risP_ += (float(nf) * isr) / 22.f; if (risP_ >= 1.f) risP_ -= 1.f;
  float risTempo[RIS], risGain[RIS];
  for (int i = 0; i < RIS; ++i) { float oct = risP_ + float(i) / RIS; oct -= std::floor(oct); risTempo[i] = 0.8f * std::pow(2.f, oct * 4.f); float c = oct * 4.f - 2.f; risGain[i] = std::exp(-0.5f * c * c / (1.3f * 1.3f)); }

  for (int f = 0; f < nf; ++f) {
    // micro-timing scheduler: fire pending events whose delay elapsed
    for (auto& p : pend_) if (p.used) { if (--p.left <= 0) { trigger(p.e); p.used = false; } }

    float bedL = 0.f, bedR = 0.f, leadL = 0.f, leadR = 0.f, lowMono = 0.f, revSend = 0.f;
    panLFO_ += 0.05f * isr; if (panLFO_ >= 1.f) panLFO_ -= 1.f; float drift = 0.25f * std::sin(kTAU * panLFO_);

    float euPrev = euPhase_;
    euPhase_ += slot * isr; if (euPhase_ >= 1.f) { euPhase_ -= 1.f; euStep_ = (euStep_ + 1) % euLen_; if ((euMask_ >> euStep_) & 1u) fireKick(); }
    if (euPrev < 0.5f && euPhase_ >= 0.5f) fireAnd();     // the off-beat "and"
    euPhase2_ += slot * 0.5f * isr; if (euPhase2_ >= 1.f) { euPhase2_ -= 1.f; euStep2_ = (euStep2_ + 1) % euLen2_; if ((euMask2_ >> euStep2_) & 1u) fireTimp(); }
    wobPhase_ += wobHz * isr; if (wobPhase_ >= 1.f) wobPhase_ -= 1.f;

    // Risset ticks (textural; fired directly so the accelerando stays coherent)
    for (int i = 0; i < RIS; ++i) { risPh_[i] += risTempo[i] * isr; if (risPh_[i] >= 1.f) { risPh_[i] -= 1.f; if (risGain[i] > 0.06f) { int gi = allocCapped(2, 28);
      if (gi >= 0) { Voice& g = vox_[gi]; g = Voice{}; g.on = true; g.layer = 2; g.hz = degHz(modeN_ * 2 + i); g.K = 2; g.pm[0] = 1; g.pm[1] = 2; g.pa[0] = 1; g.pa[1] = 0.3f; g.pnorm = 1.3f; g.amp = 0.022f * risGain[i] * grainGate_; g.life = 0.012f; g.pan = 0.6f * (2.f * frand() - 1.f); g.grng = rng_ ^ uint32_t(gi * 2246822519u); } } } }

    // sub pedal (centred, dry). In a GROWL section it becomes a dubstep wobble: a resonant SVF
    // whose cutoff + amplitude wobble with wobPhase_, saturated for grit. droneGate pauses it.
    subHz_ += (padRoot * 0.5f - subHz_) * 0.0004f; subPhase_ += subHz_ * isr; if (subPhase_ >= 1.f) subPhase_ -= 1.f;
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
    lowMono += subOut * subAmp_ * droneGate_;
    // pure mono triangle SUB-BASS — deep (35 Hz and below), sidechain-GATED against the kick
    float ttgt = padRoot * 0.25f; if (ttgt > 35.f) ttgt = 35.f;          // 35 Hz and below
    triHz_ += (ttgt - triHz_) * 0.0004f; triPhase_ += triHz_ * isr; if (triPhase_ >= 1.f) triPhase_ -= 1.f;
    kickDuck_ += (1.f - kickDuck_) * 0.00025f;                            // recover (~90 ms) from the kick duck
    float subPulse = 0.8f + 0.2f * (0.5f + 0.5f * std::cos(kTAU * euPhase_));   // subtle slow pulse on the beat
    lowMono += (4.f * std::fabs(triPhase_ - 0.5f) - 1.f) * 0.45f * kickDuck_ * subPulse;

    // MOVING JI drone: per-partial slow tremolo + detune drift + vibrato
    for (int p = 0; p < PADN; ++p) {
      padDrift_[p] += (0.013f + 0.007f * p) * isr; if (padDrift_[p] >= 1.f) padDrift_[p] -= 1.f;
      padTrem_[p]  += (0.030f + 0.020f * p) * isr; if (padTrem_[p]  >= 1.f) padTrem_[p]  -= 1.f;
      float ratioMove = 1.f + 0.004f * std::sin(kTAU * padDrift_[p] + p * 1.7f);   // slow chorus drift
      float tgt = padRoot * padRatio[p] * ratioMove; padHz_[p] += (tgt - padHz_[p]) * 0.0006f;
      padVib_[p] += (0.10f + 0.06f * p) * isr; if (padVib_[p] >= 1.f) padVib_[p] -= 1.f;
      float vib = 1.f + 0.012f * std::sin(kTAU * padVib_[p] + p);
      padPhase_[p] += padHz_[p] * vib * isr; if (padPhase_[p] >= 1.f) padPhase_[p] -= std::floor(padPhase_[p]);
      float sg = 0.6f * std::sin(kTAU * padPhase_[p]) + 0.25f * std::sin(kTAU * padPhase_[p] * 2.f);
      padLp_[p] = dn(padLp_[p] + (0.12f + 0.40f * depth) * (sg - padLp_[p]));
      padHp_[p] = dn(padHp_[p] + 0.025f * (padLp_[p] - padHp_[p]));   // track lows (~180 Hz)
      float sigHp = padLp_[p] - padHp_[p];                            // HIGH-PASS the drone to clear room for the sub-bass
      float trem = 0.72f + 0.28f * std::sin(kTAU * padTrem_[p]);                    // slow amplitude swell
      float tgtAmp = ((p < 2) ? 0.032f : 0.032f * padBloom_) * trem; padAmp_[p] += (tgtAmp - padAmp_[p]) * 0.002f;   // drone ~-10 dB
      float a = padAmp_[p] * sigHp * droneGate_;           // drone pauses / drops out by section
      float pan = (p == 0) ? 0.f : (p == 1 ? -0.55f : (p == 2 ? 0.55f : 0.78f)); pan += 0.12f * drift; float pp = 0.5f * (pan + 1.f);
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
    if (grainTimer_ <= 0.f) { float lam = 5.f + 30.f * clamp01(0.3f * act + 0.3f * (1.f - depth) + 0.4f * hes + 0.4f * md); grainTimer_ += -std::log(std::max(1e-6f, frand())) / lam;
      int gi = allocCapped(2, 28); if (gi >= 0) { Voice& g = vox_[gi]; g = Voice{}; g.on = true; g.layer = 2; int oc = int(frand() * 4.f) - 1; g.hz = degHz(nodeDegree(int(rng_ & 2047), -1)) * std::pow(2.f, float(oc));
        g.K = 2; g.pm[0] = 1; g.pm[1] = 2; g.pa[0] = 1; g.pa[1] = 0.25f + 0.4f * frand(); g.pnorm = 1.f + g.pa[1]; g.amp = (0.022f + 0.04f * act * (0.5f + frand())) * grainGate_; g.life = 0.02f + 0.18f * frand(); g.pan = 0.9f * (2.f * frand() - 1.f); g.grng = rng_ ^ (uint32_t(gi) * 40503u); } }

    // pooled voices
    float ppSendL = 0.f, ppSendR = 0.f, sampSend = 0.f;
    for (auto& v : vox_) {
      if (!v.on) continue; float s = tickVoice(v, isr, depth, moodLP_);
      if (v.layer == 4) {                                  // whisper: lead (gated), and into the delay
        float pp = 0.5f * (v.pan + drift + 1.f); pp = pp < 0 ? 0 : (pp > 1 ? 1 : pp);
        float cL = std::cos(pp * 1.5707963f), cR = std::sin(pp * 1.5707963f);
        leadL += whisperGate_ * 1.25f * s * cL; leadR += whisperGate_ * 1.25f * s * cR;        // drier + a touch louder
        ppSendL += whisperGate_ * 0.12f * s * cL; ppSendR += whisperGate_ * 0.12f * s * cR; revSend += s * whisperGate_ * 0.35f;   // much less wet
      } else if (v.layer == 9) { leadL += 1.05f * s; leadR += 1.05f * s; }   // "and" tick: dry, centred, un-ducked
      else if (v.layer == 6 || v.layer == 7) { lowMono += s; }
      else {
        float pan = v.pan + (v.layer == 2 || v.layer == 8 ? drift : 0.f); float pp = 0.5f * (pan + 1.f); pp = pp < 0 ? 0 : (pp > 1 ? 1 : pp);
        float cL = std::cos(pp * 1.5707963f), cR = std::sin(pp * 1.5707963f);
        bool up = (v.layer <= 2 || v.layer == 8);            // upper part: pluck/bell/grain/clang
        float dryg = up ? 0.72f : 1.f, rv = up ? 1.25f : 0.5f;   // pushed back + further into reverb
        if (v.smp) { dryg = 1.35f; rv = 0.7f; }              // bring the CC0 SAMPLES out front
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
    { float sw1 = 0.f, sw2 = 0.f; sampRev_(sampSend * 0.5f, sw1, sw2, 0.5f); float sw = 0.5f * (sw1 + sw2); bedL += sw * 0.3f; bedR += sw * 0.3f; }

    float w1 = 0.f, w2 = 0.f; reverb_(revSend * 0.5f, w1, w2, 0.6f);
    float L = duckGain_ * bedL + leadL + lowDuck_ * lowMono + reverbWet_ * w1;
    float R = duckGain_ * bedR + leadR + lowDuck_ * lowMono + reverbWet_ * w2;
    masterLoL_ = dn(masterLoL_ + aLo * (L - masterLoL_)); L += 2.16f * masterLoL_;
    masterLoR_ = dn(masterLoR_ + aLo * (R - masterLoR_)); R += 2.16f * masterLoR_;
    masterHiL_ = dn(masterHiL_ + aHi * (L - masterHiL_)); L += 0.45f * (masterHiL_ - L);   // pink HF tilt
    masterHiR_ = dn(masterHiR_ + aHi * (R - masterHiR_)); R += 0.45f * (masterHiR_ - R);
    io.out(0, f) = std::tanh(L * master); if (nch > 1) io.out(1, f) = std::tanh(R * master);
    for (int c = 2; c < nch; ++c) io.out(c, f) = 0.f;
  }
  lastDepth_ = depth;
}

}  // namespace wosw
