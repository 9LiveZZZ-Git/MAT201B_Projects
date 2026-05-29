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

// Just-intonation ratios (Kuchera-Morin): unison, m2, m3 6/5, M3 5/4, P4, P5 3/2, h7 7/4, octave.
static const float kJI[8] = {1.f, 16.f / 15.f, 6.f / 5.f, 5.f / 4.f, 4.f / 3.f, 3.f / 2.f, 7.f / 4.f, 2.f};

static inline float dn(float x) { return (std::fabs(x) < 1e-15f) ? 0.f : x; }  // denormal flush
static inline float clamp01(float x) { return x < 0.f ? 0.f : (x > 1.f ? 1.f : x); }

// ===================== tiny 16-bit PCM WAV loader (matches our converted assets) =====================
static bool loadWav16(const std::string& path, std::vector<float>& out, float& srcSR) {
  FILE* f = std::fopen(path.c_str(), "rb");
  if (!f) return false;
  char riff[4]; if (std::fread(riff, 1, 4, f) != 4 || std::memcmp(riff, "RIFF", 4)) { std::fclose(f); return false; }
  std::fseek(f, 4, SEEK_CUR);                       // file size
  char wave[4]; if (std::fread(wave, 1, 4, f) != 4 || std::memcmp(wave, "WAVE", 4)) { std::fclose(f); return false; }
  int ch = 1, bits = 16; uint32_t rate = 32000;
  char id[4]; uint32_t sz = 0;
  bool gotFmt = false, gotData = false;
  while (std::fread(id, 1, 4, f) == 4 && std::fread(&sz, 4, 1, f) == 1) {
    if (!std::memcmp(id, "fmt ", 4)) {
      uint16_t fmt = 1, c = 1, b = 16; uint32_t r = 32000;
      std::fread(&fmt, 2, 1, f); std::fread(&c, 2, 1, f); std::fread(&r, 4, 1, f);
      std::fseek(f, 6, SEEK_CUR);                    // byterate + blockalign
      std::fread(&b, 2, 1, f);
      ch = c; bits = b; rate = r; gotFmt = true;
      std::fseek(f, int(sz) - 16, SEEK_CUR);
    } else if (!std::memcmp(id, "data", 4)) {
      if (!gotFmt || bits != 16) { std::fclose(f); return false; }
      int n = int(sz) / 2;
      std::vector<int16_t> raw(n);
      if (int(std::fread(raw.data(), 2, n, f)) != n) n = 0;
      int frames = (ch > 0) ? n / ch : 0;
      out.resize(frames);
      for (int i = 0; i < frames; ++i) {              // downmix to mono if needed
        int acc = 0; for (int c = 0; c < ch; ++c) acc += raw[i * ch + c];
        out[i] = float(acc) / (ch * 32768.f);
      }
      srcSR = float(rate); gotData = true;
      break;
    } else {
      std::fseek(f, int(sz) + (int(sz) & 1), SEEK_CUR);
    }
  }
  std::fclose(f);
  return gotData && !out.empty();
}

// note name in a filename -> Hz (C4=261.63). Sanitized names lost '#', so we read letter+octave.
static bool parseNoteHz(const std::string& name, float& hz) {
  static const int semi[7] = {9, 11, 0, 2, 4, 5, 7};   // A,B,C,D,E,F,G
  for (size_t i = 0; i + 1 < name.size(); ++i) {
    char c = char(std::toupper((unsigned char)name[i]));
    if (c < 'A' || c > 'G') continue;
    size_t j = i + 1; int sharp = 0;
    if (name[j] == '#' || name[j] == 's') { sharp = 1; ++j; }
    if (j >= name.size() || !std::isdigit((unsigned char)name[j])) continue;
    // avoid matching letters inside words: require the prev char to be a separator/none
    if (i > 0 && std::isalpha((unsigned char)name[i - 1])) continue;
    int oct = name[j] - '0';
    int midi = (oct + 1) * 12 + semi[c - 'A'] + sharp;
    hz = 440.f * std::pow(2.f, (midi - 69) / 12.f);
    return true;
  }
  return false;
}

void AudioEngine::loadSamples(const std::string& assetDir) {
  const std::string bases[] = { assetDir + "/audio", "assets/audio", "../../assets/audio",
                                "reagency/assets/audio", "MAT201B_Projects/reagency/assets/audio" };
  for (const auto& b : bases) {
    DIR* d = opendir(b.c_str());
    if (!d) continue;
    struct dirent* e;
    while ((e = readdir(d)) != nullptr) {
      std::string fn = e->d_name;
      if (fn.size() < 5) continue;
      std::string lc = fn; for (auto& ch : lc) ch = char(std::tolower((unsigned char)ch));
      if (lc.substr(lc.size() - 4) != ".wav") continue;
      Sample s;
      if (!loadWav16(b + "/" + fn, s.data, s.srcSR)) continue;
      s.pitched = parseNoteHz(fn, s.rootHz);
      if (lc.find("agogo") != std::string::npos || lc.find("bell") != std::string::npos) s.pitched = false;
      int idx = int(samples_.size());
      samples_.push_back(std::move(s));
      if ((lc.find("organ") != std::string::npos || lc.find("pedal") != std::string::npos) && sBass_ < 0) sBass_ = idx;
      else if ((lc.find("kalimba") != std::string::npos || lc.find("mbira") != std::string::npos) && sPluck_ < 0) sPluck_ = idx;
      else if ((lc.find("agogo") != std::string::npos || lc.find("glock") != std::string::npos || lc.find("bell") != std::string::npos) && sBell_ < 0) sBell_ = idx;
      else if ((lc.find("psaltery") != std::string::npos || lc.find("bowed") != std::string::npos || lc.find("viol") != std::string::npos) && sTrace_ < 0) sTrace_ = idx;
    }
    closedir(d);
    if (!samples_.empty()) {
      std::fprintf(stderr, "[wosw audio] %zu CC0 samples from %s/  (bass=%d pluck=%d bell=%d trace=%d)\n",
                   samples_.size(), b.c_str(), sBass_, sPluck_, sBell_, sTrace_);
      return;
    }
  }
  std::fprintf(stderr, "[wosw audio] no assets/audio samples — fully procedural synthesis\n");
}

// ===================== manifest scale =====================
void AudioEngine::loadManifest(const std::string& assetDir) {
  const std::string bases[] = { assetDir, "assets", "../../assets",
                                "reagency/assets", "MAT201B_Projects/reagency/assets" };
  for (const auto& b : bases) {
    FILE* f = std::fopen((b + "/manifest.json").c_str(), "rb");
    if (!f) continue;
    std::string s; char buf[4096]; size_t n;
    while ((n = std::fread(buf, 1, sizeof(buf), f)) > 0) s.append(buf, n);
    std::fclose(f);
    auto pos = s.find("\"root_hz\"");
    if (pos != std::string::npos) {
      double v = std::atof(s.c_str() + s.find(':', pos) + 1);
      if (v > 20.0 && v < 4000.0) root_ = float(v);
    }
    pos = s.find("\"mode\"");
    if (pos != std::string::npos) {
      size_t lb = s.find('[', pos), rb = s.find(']', lb);
      if (lb != std::string::npos && rb != std::string::npos) {
        int m[MAXMODE], cnt = 0; const char* p = s.c_str() + lb + 1; const char* end = s.c_str() + rb;
        while (p < end && cnt < MAXMODE) {
          while (p < end && (*p == ' ' || *p == ',')) ++p;
          if (p >= end) break;
          m[cnt++] = std::atoi(p);
          while (p < end && *p != ',') ++p;
        }
        if (cnt >= 3) { modeN_ = cnt; for (int i = 0; i < cnt; ++i) modeBuf_[i] = m[i]; }
      }
    }
    std::fprintf(stderr, "[wosw audio] scale: root=%.1f Hz, %d degrees from %s/manifest.json\n",
                 root_, modeN_, b.c_str());
    return;
  }
  std::fprintf(stderr, "[wosw audio] manifest.json not found — default A minor (root 220)\n");
}

float AudioEngine::degHz(int degree) const {
  int idx = degree % modeN_, oct = degree / modeN_;
  if (idx < 0) { idx += modeN_; oct -= 1; }
  return root_ * std::pow(2.f, (modeBuf_[idx] + 12 * oct) / 12.f);
}
int AudioEngine::nodeDegree(int node, int cluster) const {
  unsigned h = unsigned(node) * 2654435761u;
  int d = int(h % unsigned(modeN_ * 3));
  if (cluster > 0 && (cluster % 2 == 1)) d += modeN_;
  return d;
}
float AudioEngine::jiBassHz() const {
  return cClusterRoot_.load(std::memory_order_relaxed) * 0.5f;   // an octave below the cluster centre
}

// ===================== lock-free ring =====================
bool AudioEngine::push(const Ev& e) {
  uint32_t h = rHead_.load(std::memory_order_relaxed), nx = (h + 1) % RING;
  if (nx == rTail_.load(std::memory_order_acquire)) return false;
  ring_[h] = e; rHead_.store(nx, std::memory_order_release); return true;
}
bool AudioEngine::pop(Ev& e) {
  uint32_t t = rTail_.load(std::memory_order_relaxed);
  if (t == rHead_.load(std::memory_order_acquire)) return false;
  e = ring_[t]; rTail_.store((t + 1) % RING, std::memory_order_release); return true;
}

void AudioEngine::init(const std::string& assetDir, double sampleRate) {
  sr_ = sampleRate > 0 ? sampleRate : 44100.0;
  loadManifest(assetDir);
  loadSamples(assetDir);
  reverb_.bandwidth(0.9f); reverb_.damping(0.4f); reverb_.decay(0.86f);
  cClusterRoot_.store(root_, std::memory_order_relaxed);
  subHz_ = root_ * 0.25f;
  for (int p = 0; p < PADN; ++p) padHz_[p] = root_ * kJI[p == 0 ? 0 : (p == 1 ? 2 : (p == 2 ? 5 : 7))];
  ready_.store(true, std::memory_order_release);     // publishes the init writes above
}

// ===================== helpers =====================
static void formantFor(char vowel, float& f0, float& f1, float& f2) {
  switch (std::tolower((unsigned char)vowel)) {
    case 'a': f0 = 800; f1 = 1150; f2 = 2900; break;
    case 'e': f0 = 500; f1 = 1800; f2 = 2500; break;
    case 'i': f0 = 320; f1 = 2300; f2 = 3000; break;
    case 'o': f0 = 450; f1 =  850; f2 = 2800; break;
    case 'u': f0 = 325; f1 =  700; f2 = 2600; break;
    default:  f0 = 500; f1 = 1500; f2 = 2500; break;
  }
}
// Euclidean rhythm (Bresenham): k onsets spread over n slots -> bitmask.
static uint32_t euclidMask(int k, int n) {
  if (n <= 0) return 1u; if (k < 1) k = 1; if (k > n) k = n;
  uint32_t mask = 0; int prev = -1;
  for (int i = 0; i < n; ++i) { int cur = (i * k) / n; if (cur != prev) { mask |= 1u << i; prev = cur; } }
  return mask;
}
// Euclidean slot rate (slots/sec): faster out in the galaxy + when active, slower sunk/hesitant.
static float slotRate(float depth, float hes, float act) {
  float r = (2.0f + 1.7f * depth) * (1.f - 0.28f * hes) * (1.f + 0.22f * act);
  return r < 0.6f ? 0.6f : (r > 6.f ? 6.f : r);
}

// ===================== PRIMARY sim-thread API =====================
void AudioEngine::onArrival(int node, int type, float density, int cluster) {
  if (!ready()) return;
  curNode_ = node; clusterCur_ = cluster; simDensity_ = density < 0 ? 0.5f : density;
  ++visits_; cVisits_.store(visits_, std::memory_order_relaxed);
  cClusterId_.store(cluster, std::memory_order_relaxed);
  // cluster tonal centre (mid register) — pad/bass glide to it
  int croot = (cluster >= 0) ? (cluster % modeN_) : 0;
  cClusterRoot_.store(degHz(croot + modeN_), std::memory_order_relaxed);
  activity_ = std::min(1.f, activity_ + 0.5f);
  // the arrival "ping": the node's own tuned pitch (image=pluck, word=bell)
  Ev e{}; e.kind = EV_NOTE; e.layer = (type == 1) ? 1 : 0;
  e.hz = degHz(nodeDegree(node, cluster)); e.amp = 0.26f + 0.26f * simDensity_;
  e.pan = cFocusPan_.load(std::memory_order_relaxed);
  push(e);
  melDeg_ = nodeDegree(node, cluster);             // seed the Markov walk at the arrival degree
}

void AudioEngine::igniteArp(const std::vector<std::pair<int, float>>& nbrs, int cluster) {
  if (!ready()) return;
  // STOCHASTIC MARKOV PHRASE (replaces the ascending arp). Deterministic per node so a revisit
  // re-rings the same gesture; edge weights bias amplitude + a gravity pull; hesitation widens
  // the step distribution (a torn machine leaps); a voice-leading guard forbids two big leaps.
  mprng_ = unsigned(curNode_) * 2654435761u ^ unsigned(visits_) * 40503u;
  mel_.clear(); melIdx_ = 0; melTimer_ = 0.f;
  melPan_ = cFocusPan_.load(std::memory_order_relaxed);
  const float hes = simHes_, depth = simDepth_;
  const int octShift = (depth > 0.6f) ? modeN_ : (depth < 0.35f ? -modeN_ : 0);
  const int L = 3 + int(4.f * simDensity_);
  lastLeap_ = false;
  // gravity target: strongest neighbour's degree
  int gravDeg = melDeg_; float bestW = -1.f;
  for (const auto& nb : nbrs) if (nb.second > bestW) { bestW = nb.second; gravDeg = nodeDegree(nb.first, cluster); }
  static const int dStep[13] = {-7, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 7};
  auto frnd = [&]() { mprng_ ^= mprng_ << 13; mprng_ ^= mprng_ >> 17; mprng_ ^= mprng_ << 5; return float(mprng_) / 4294967296.f; };
  for (int step = 0; step < L; ++step) {
    float w[13], sum = 0.f;
    for (int i = 0; i < 13; ++i) {
      int d = dStep[i];
      float ww = std::exp(-std::fabs(float(d)) / 3.f);     // small steps dominate
      ww += hes * 0.6f * (std::abs(d) >= 4 ? 1.f : 0.f);   // torn -> leaps
      if ((melDeg_ + d) % modeN_ == gravDeg % modeN_) ww += 0.8f;  // pull to chord/home tone
      if (lastLeap_ && std::abs(d) > 2) ww = 0.f;          // voice-leading: no two big leaps
      w[i] = ww; sum += ww;
    }
    float r = frnd() * sum, acc = 0.f; int pick = 6;
    for (int i = 0; i < 13; ++i) { acc += w[i]; if (r <= acc) { pick = i; break; } }
    int d = dStep[pick];
    lastLeap_ = std::abs(d) > 2;
    melDeg_ += d;
    melDeg_ = std::max(-modeN_, std::min(modeN_ * 3, melDeg_));   // ambitus clamp
    float edgeW = nbrs.empty() ? 0.5f : nbrs[step % nbrs.size()].second;
    int layer = (step == 0 || edgeW > 0.78f) ? 1 : 0;            // strong edges sparkle as bells
    int hz_deg = melDeg_ + octShift;
    mel_.push_back({degHz(hz_deg), layer});
  }
}

void AudioEngine::traceOn(int slot, int node, float pan) {
  if (!ready()) return;
  Ev e{}; e.kind = EV_TRACE_ON; e.layer = 3;
  e.hz = degHz(nodeDegree(node, 0)) * 0.5f; e.amp = 0.22f; e.pan = pan; e.islot = slot;
  push(e);
}
void AudioEngine::traceOff(int slot) { if (ready()) { Ev e{}; e.kind = EV_TRACE_OFF; e.islot = slot; push(e); } }

void AudioEngine::whisper(const std::string& word, int node, float pan) {
  if (!ready()) return;
  int syl = 0, firstV = -1;
  for (char c : word) { char l = char(std::tolower((unsigned char)c));
    if (l == 'a' || l == 'e' || l == 'i' || l == 'o' || l == 'u') { if (firstV < 0) firstV = l; ++syl; } }
  float f0, f1, f2;
  if (word.empty() || syl == 0) {
    static const char vs[5] = {'a', 'e', 'i', 'o', 'u'};
    formantFor(vs[(node >= 0 ? node : 0) % 5], f0, f1, f2);
    syl = (node >= 0 ? node : 0) % 3 + 1;
  } else formantFor(char(firstV), f0, f1, f2);
  syl = syl < 1 ? 1 : (syl > 5 ? 5 : syl);
  Ev e{}; e.kind = EV_WHISPER; e.layer = 4; e.amp = 0.4f; e.pan = pan; e.islot = syl;
  e.a = f0; e.b = f1; e.c = f2; push(e);
}

void AudioEngine::update(float dt, float hesitation, float depth, float progress, float focusPan) {
  if (!ready()) return;
  simHes_ = hesitation; simDepth_ = depth;
  cHesitation_.store(hesitation, std::memory_order_relaxed);
  cDepth_.store(depth, std::memory_order_relaxed);
  cFocusPan_.store(focusPan < -1 ? -1 : (focusPan > 1 ? 1 : focusPan), std::memory_order_relaxed);
  activity_ *= std::pow(0.5f, dt / 0.8f);
  cActivity_.store(activity_, std::memory_order_relaxed);
  (void)progress;

  // recompute the Euclidean pattern when (k,n) change; push to the audio thread (no audio alloc)
  int k = 2 + std::min(6, visits_ / 20);
  int n = (depth > 0.7f) ? ((clusterCur_ & 1) ? 12 : 8)
        : (depth > 0.4f) ? ((clusterCur_ & 1) ? 9 : 7)
                         : ((clusterCur_ & 1) ? 16 : 5);
  if (k > n - 1) k = n - 1; if (k < 1) k = 1;
  if (k != euK_ || n != euN_) {
    euK_ = k; euN_ = n;
    Ev e{}; e.kind = EV_RHYTHM; e.islot = int(euclidMask(k, n)); e.layer = n; push(e);
  }

  // pace the stochastic melody onto ~eighth-note slots of the current tempo
  if (melIdx_ < mel_.size()) {
    float period = 1.f / (2.f * slotRate(depth, hesitation, activity_));
    melTimer_ += dt;
    while (melIdx_ < mel_.size() && melTimer_ >= period) {
      melTimer_ -= period;
      Ev e{}; e.kind = EV_NOTE; e.layer = mel_[melIdx_].second; e.hz = mel_[melIdx_].first;
      e.amp = 0.11f + 0.16f * clamp01(0.5f + 0.5f * (float((melIdx_ * 37) % 100) / 100.f - 0.5f));
      e.pan = melPan_ + 0.12f * (float(melIdx_ % 5) - 2.f);
      push(e); ++melIdx_;
    }
  }
}

// ===================== audio-thread =====================
float AudioEngine::frand() { rng_ ^= rng_ << 13; rng_ ^= rng_ >> 17; rng_ ^= rng_ << 5; return float(rng_) / 4294967296.f; }
static inline float vrand(uint32_t& s) { s ^= s << 13; s ^= s >> 17; s ^= s << 5; return float(s) / 4294967296.f; }

int AudioEngine::allocVoice() { for (int i = 0; i < NVOX; ++i) if (!vox_[i].on) return i; return -1; }

void AudioEngine::trigger(const Ev& e) {
  if (e.kind == EV_RHYTHM) { euMask_ = uint32_t(e.islot); euLen_ = e.layer < 1 ? 1 : e.layer; return; }
  if (e.kind == EV_TRACE_OFF) {
    for (auto& v : vox_) if (v.on && v.layer == 3 && v.tslot == e.islot) if (v.life > v.age + 1.0f) v.life = v.age + 1.0f;
    return;
  }
  int i = allocVoice(); if (i < 0) return;
  Voice& v = vox_[i]; v = Voice{};
  v.on = true; v.layer = e.layer; v.hz = e.hz; v.amp = e.amp; v.pan = e.pan; v.age = 0.f;
  v.grng = 0x2545F491u ^ (uint32_t(i) * 2654435761u);

  // choose a CC0 sample for this layer if one is loaded; else additive synthesis
  auto setSample = [&](int sidx) {
    const Sample& s = samples_[sidx];
    v.smp = s.data.data(); v.smpLen = int(s.data.size()); v.smpPos = 0.f;
    float ratio = s.pitched ? (v.hz / s.rootHz) : 1.f;
    v.smpRate = ratio * (s.srcSR / float(sr_));
  };
  auto setPartials = [&](int K, float inh) {
    v.K = K > 6 ? 6 : K; v.pnorm = 0.f;
    for (int k = 1; k <= v.K; ++k) { v.pm[k - 1] = k * (1.f + inh * k * k); v.pa[k - 1] = 1.f / std::pow(float(k), 1.3f); v.pnorm += v.pa[k - 1]; }
    if (v.pnorm <= 0.f) v.pnorm = 1.f;
  };

  switch (e.layer) {
    case 0:  v.life = 2.4f; v.atk = 0.010f; if (sPluck_ >= 0) setSample(sPluck_); else setPartials(5, 0.0008f); break;  // pluck/image
    case 1:  v.life = 1.6f; v.atk = 0.004f; if (sBell_  >= 0) setSample(sBell_);  else setPartials(6, 0.004f);  break;  // bell/word
    case 3:  v.life = 7.5f; v.atk = 0.9f;  v.tslot = e.islot; if (sTrace_ >= 0) setSample(sTrace_); else setPartials(4, 0.0006f); break;  // trace
    case 5:  v.life = 0.55f; v.atk = 0.006f; if (sBass_ >= 0) setSample(sBass_); else setPartials(4, 0.0003f); break;  // bass onset
    case 4: {                                 // whisper
      v.syl = e.islot < 1 ? 1 : e.islot; v.life = 1.4f + 0.5f * v.syl; v.body = 0.62f; v.atk = 0.05f;
      v.fmt[0] = e.a; v.fmt[1] = e.b; v.fmt[2] = e.c;
      const float bw[3] = {90.f, 110.f, 140.f};
      for (int k = 0; k < 3; ++k) {
        float r = std::exp(-kPI * bw[k] / float(sr_));
        v.fcoef[k][0] = (1.f - r); v.fcoef[k][1] = -2.f * r * std::cos(kTAU * v.fmt[k] / float(sr_)); v.fcoef[k][2] = r * r;
      }
      break;
    }
    default: v.life = 0.15f; v.atk = 0.01f; setPartials(2, 0.f); break;   // grain
  }
}

float AudioEngine::tickVoice(Voice& v, float isr, float depth, float bright) {
  v.age += isr;
  if (v.age >= v.life) { v.on = false; return 0.f; }
  const float u = v.age / v.life;

  // ---- sampled voices ----
  if (v.smp) {
    if (v.smpPos >= v.smpLen - 2) { v.on = false; return 0.f; }
    int i0 = int(v.smpPos); float fr = v.smpPos - i0;
    float s = v.smp[i0] * (1.f - fr) + v.smp[i0 + 1] * fr;
    v.smpPos += v.smpRate;
    float env;
    if (v.layer == 3) {                                  // trace: attack/sustain/release
      env = (u < 0.12f) ? (u / 0.12f) : (u < 0.55f ? 1.f : std::max(0.f, 1.f - (u - 0.55f) / 0.45f));
    } else if (v.layer == 5) {                           // bass onset: fast attack, exp decay
      float a = v.atk / v.life; env = (u < a) ? (u / std::max(1e-4f, a)) : std::exp(-3.0f * (u - a) / (1.f - a));
    } else {                                             // pluck/bell: let the sample decay, fade edges
      float a = v.atk / v.life; env = (u < a) ? (u / std::max(1e-4f, a)) : (u > 0.85f ? (1.f - (u - 0.85f) / 0.15f) : 1.f);
    }
    return s * env * v.amp;
  }

  // ---- whisper: breath -> formants -> syllabic env -> granular fade ----
  if (v.layer == 4) {
    float env;
    if (u < v.body) {
      float t = u / v.body, ph = t * float(v.syl), frac = ph - std::floor(ph);
      env = std::pow(0.5f - 0.5f * std::cos(kTAU * frac), 0.6f) * std::sin(kPI * t);
    } else {
      float t = (u - v.body) / (1.f - v.body), gp = t * (6.f + 10.f * t), gf = gp - std::floor(gp), duty = 0.6f * (1.f - t);
      float grain = (gf < duty) ? (0.5f - 0.5f * std::cos(kTAU * gf / std::max(1e-3f, duty))) : 0.f;
      env = grain * (1.f - t);
    }
    float nz = vrand(v.grng) * 2.f - 1.f, s = 0.f; const float w[3] = {1.f, 0.7f, 0.45f};
    for (int k = 0; k < 3; ++k) {
      float y = v.fcoef[k][0] * nz - v.fcoef[k][1] * v.fz_[k][0] - v.fcoef[k][2] * v.fz_[k][1];
      v.fz_[k][1] = dn(v.fz_[k][0]); v.fz_[k][0] = dn(y); s += w[k] * y;
    }
    return s * env * v.amp * 1.4f;
  }

  // ---- additive (pluck/bell/grain/trace fallback) ----
  v.phase += v.hz * isr; if (v.phase >= 1.f) v.phase -= std::floor(v.phase);
  float s = 0.f; const float nyq = 0.45f * float(sr_);
  for (int k = 0; k < v.K; ++k) {
    float pf = v.hz * v.pm[k]; if (pf >= nyq) break;
    s += v.pa[k] * std::sin(kTAU * v.phase * v.pm[k]);
  }
  s /= v.pnorm;
  float env;
  if (v.layer == 2) { float c = u * 2.f - 1.f; env = std::exp(-c * c * 6.f); }
  else if (v.layer == 3) env = (u < 0.12f) ? (u / 0.12f) : (u < 0.55f ? 1.f : std::max(0.f, 1.f - (u - 0.55f) / 0.45f));
  else { float a = v.atk / v.life; env = (u < a) ? (u / std::max(1e-4f, a)) : std::exp(-3.5f * (u - a) / (1.f - a)); }
  float out = s * env * v.amp;
  float cut = (v.layer == 1) ? (0.45f + 0.45f * bright) : (0.22f + 0.45f * bright);
  v.lp = dn(v.lp + cut * (out - v.lp));
  return v.lp;
}

void AudioEngine::fireBass() {
  Ev e{}; e.kind = EV_NOTE; e.layer = 5;
  e.hz = jiBassHz(); e.amp = 0.34f + 0.30f * (1.f - cDepth_.load(std::memory_order_relaxed));   // swells when sunk
  e.pan = 0.f;
  trigger(e);                                       // audio-side direct (already on the audio thread)
}

void AudioEngine::render(al::AudioIOData& io) {
  const int nf = int(io.framesPerBuffer()), nch = int(io.channelsOut());
  if (!ready()) { for (int c = 0; c < nch; ++c) for (int f = 0; f < nf; ++f) io.out(c, f) = 0.f; return; }

  Ev e; while (pop(e)) trigger(e);

  const float depth = cDepth_.load(std::memory_order_relaxed);
  const float hes   = cHesitation_.load(std::memory_order_relaxed);
  const float act   = cActivity_.load(std::memory_order_relaxed);
  const float padRoot = cClusterRoot_.load(std::memory_order_relaxed) * 0.5f;

  // derived per-buffer (mood/tension), all glided
  tension_  += (clamp01(0.5f * hes + 0.5f * (1.f - depth)) - tension_) * 0.05f;
  float brightTgt = 0.12f + 0.78f * depth;                 // galaxy bright, vessel dark
  moodLP_   += (brightTgt - moodLP_) * 0.02f;
  padBloom_ += ((0.30f + 0.70f * depth) - padBloom_) * 0.02f;
  reverbWet_ += ((0.22f + 0.16f * (1.f - depth) + 0.10f * hes) - reverbWet_) * 0.02f;
  subAmp_   += ((0.05f + 0.16f * (1.f - depth)) - subAmp_) * 0.0004f;
  shepGain_ += ((0.045f + 0.05f * depth + 0.05f * tension_) - shepGain_) * 0.01f;
  shepRateTgt_ = (hes > 0.55f ? -1.f : 1.f) * (1.f / (28.f - 14.f * hes));
  shepRate_ += (shepRateTgt_ - shepRate_) * 0.0008f;

  float decay = 0.82f + 0.12f * (1.f - depth), damp = 0.30f + 0.30f * (1.f - depth);
  if (std::fabs(decay - lastDecay_) > 1e-3f) { reverb_.decay(decay); lastDecay_ = decay; }
  if (std::fabs(damp - lastDamp_)  > 1e-3f) { reverb_.damping(damp); lastDamp_ = damp; }

  const float isr = 1.f / float(sr_), master = 0.5f;
  const float slot = slotRate(depth, hes, act);
  // tension bends the pad's just fifth toward a beating wolf interval
  const float fifth = 1.5f - 0.019f * tension_;
  const float padRatio[PADN] = {1.f, kJI[2], fifth, 2.f};   // root, m3, P5(detuned), octave

  for (int f = 0; f < nf; ++f) {
    float L = 0.f, R = 0.f, mono = 0.f, bassMono = 0.f;

    // ---- Euclidean clock -> bass onsets ----
    euPhase_ += slot * isr;
    if (euPhase_ >= 1.f) { euPhase_ -= 1.f; euStep_ = (euStep_ + 1) % euLen_; if ((euMask_ >> euStep_) & 1u) fireBass(); }

    // ---- sub-bass pedal (procedural floor) ----
    subHz_ += (padRoot * 0.5f - subHz_) * 0.0004f;
    subPhase_ += subHz_ * isr; if (subPhase_ >= 1.f) subPhase_ -= 1.f;
    float subS = std::sin(kTAU * subPhase_) + 0.3f * std::sin(kTAU * subPhase_ * 2.f);
    subLp_ = dn(subLp_ + (0.10f + 0.10f * depth) * (subS - subLp_));
    bassMono += subLp_ * subAmp_;

    // ---- JI pad bed ----
    for (int p = 0; p < PADN; ++p) {
      float tgt = padRoot * padRatio[p];
      padHz_[p] += (tgt - padHz_[p]) * 0.0006f;
      padVib_[p] += (0.15f + 0.05f * p) * isr; if (padVib_[p] >= 1.f) padVib_[p] -= 1.f;
      float vib = 1.f + 0.006f * std::sin(kTAU * padVib_[p] + p);
      padPhase_[p] += padHz_[p] * vib * isr; if (padPhase_[p] >= 1.f) padPhase_[p] -= std::floor(padPhase_[p]);
      float s = 0.6f * std::sin(kTAU * padPhase_[p]) + 0.25f * std::sin(kTAU * padPhase_[p] * 2.f);
      padLp_[p] = dn(padLp_[p] + (0.10f + 0.40f * depth) * (s - padLp_[p]));
      float tgtAmp = (p < 2) ? 0.05f : 0.05f * padBloom_;   // upper partials bloom with depth
      padAmp_[p] += (tgtAmp - padAmp_[p]) * 0.001f;
      float a = padAmp_[p] * padLp_[p];
      float pan = (p == 0) ? 0.f : (p == 1 ? -0.3f : (p == 2 ? 0.3f : 0.5f));
      float pp = 0.5f * (pan + 1.f);
      L += a * std::cos(pp * 1.5707963f); R += a * std::sin(pp * 1.5707963f); mono += a;
    }

    // ---- Shepard ladder (the endless rise) ----
    shepPhase_ += shepRate_ * isr; if (shepPhase_ >= 1.f) shepPhase_ -= 1.f; if (shepPhase_ < 0.f) shepPhase_ += 1.f;
    float sh = 0.f, shn = 0.f;
    for (int i = 0; i < SHEP; ++i) {
      float oct = shepPhase_ + float(i) / SHEP; oct -= std::floor(oct);
      float fi = 32.7f * std::pow(2.f, oct * 6.f);
      float lf = std::log2(fi), a = std::exp(-0.5f * (lf - std::log2(300.f)) * (lf - std::log2(300.f)) / (1.4f * 1.4f));
      shepPh_[i] += fi * isr; if (shepPh_[i] >= 1.f) shepPh_[i] -= std::floor(shepPh_[i]);
      sh += a * std::sin(kTAU * shepPh_[i]); shn += a;
    }
    sh = (shn > 0 ? sh / shn : 0.f) * shepGain_;
    L += sh; R += sh; mono += sh;

    // ---- grain cloud (Poisson, Xenakis mass) ----
    grainTimer_ -= isr;
    if (grainTimer_ <= 0.f) {
      float lam = 6.f + 44.f * clamp01(0.4f * act + 0.4f * (1.f - depth) + 0.5f * hes);
      grainTimer_ += -std::log(std::max(1e-6f, frand())) / lam;
      int gi = allocVoice();
      if (gi >= 0) {
        Voice& g = vox_[gi]; g = Voice{}; g.on = true; g.layer = 2;
        g.hz = degHz(nodeDegree(int(rng_ & 1023), -1)) * (depth > 0.55f ? 2.f : 1.f);
        g.K = 2; g.pm[0] = 1; g.pm[1] = 2; g.pa[0] = 1; g.pa[1] = 0.4f; g.pnorm = 1.4f;
        g.amp = 0.04f + 0.05f * act; g.life = 0.02f + 0.10f * frand();
        float spread = 0.3f + 0.7f * clamp01(hes); g.pan = spread * (2.f * frand() - 1.f);
        g.grng = rng_ ^ (uint32_t(gi) * 40503u);
      }
    }

    // ---- pooled voices ----
    for (auto& v : vox_) {
      if (!v.on) continue;
      float s = tickVoice(v, isr, depth, moodLP_);
      if (v.layer == 5) { bassMono += s; continue; }     // bass -> center, drier send
      float pp = 0.5f * (v.pan + 1.f);
      L += s * std::cos(pp * 1.5707963f); R += s * std::sin(pp * 1.5707963f); mono += s;
    }

    // bass to centre
    L += bassMono; R += bassMono;

    // ---- reverb (bass sent drier) + soft limiter ----
    float w1 = 0.f, w2 = 0.f;
    reverb_(mono * 0.5f + bassMono * 0.15f, w1, w2, 0.6f);
    float outL = std::tanh((L + reverbWet_ * w1) * master);
    float outR = std::tanh((R + reverbWet_ * w2) * master);
    io.out(0, f) = outL;
    if (nch > 1) io.out(1, f) = outR;
    for (int c = 2; c < nch; ++c) io.out(c, f) = 0.f;
  }
  lastDepth_ = depth;
}

}  // namespace wosw
