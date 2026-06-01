#include "viz/DetectionHUD.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <utility>

namespace wosw {

using namespace al;

// Same additive-line contract as WebRenderer: loc0 position, loc1 vec4 color, uIntensity scales the
// additive alpha. Reusing this exact path keeps the HUD dome-warp-safe (allolib feeds the omni/dome
// model-view + projection matrices) and avoids a second shader style. No raw GL.
static const char* kVert = R"GLSL(
#version 330
uniform mat4 al_ModelViewMatrix;
uniform mat4 al_ProjectionMatrix;
layout (location = 0) in vec3 vertexPosition;
layout (location = 1) in vec4 vertexColor;
out vec4 vcol;
void main() {
  vcol = vertexColor;
  gl_Position = al_ProjectionMatrix * al_ModelViewMatrix * vec4(vertexPosition, 1.0);
}
)GLSL";

// The chromatic split is produced by the CPU builder (it shifts the whole mesh along camR/camU and
// draws once per channel); uMask isolates one colour per pass so the three additive passes fringe
// apart. uScan adds faint horizontal CRT scanlines via gl_FragCoord parity. Still allolib's own
// al::ShaderProgram — no raw GL, no FBO/post pass.
static const char* kFrag = R"GLSL(
#version 330
in vec4 vcol;
out vec4 fragColor;
uniform float uIntensity;
uniform vec3  uMask;     // per-channel pass mask; (1,1,1) = no split (default)
uniform float uScan;     // CRT scanline darken amount 0..1
void main() {
  float line = 1.0 - uScan * 0.5 * (0.5 - 0.5 * cos(gl_FragCoord.y * 3.14159265));
  fragColor = vec4(vcol.rgb * uMask, vcol.a * uIntensity * line);
}
)GLSL";

// the CV-overlay tint: a cool cyan-green so it reads as "machine vision", distinct from the warm
// human traces. Confidence bar leans warmer.
static const Vec3f kHud(0.45f, 1.00f, 0.85f);
static const Vec3f kHot(1.00f, 0.95f, 0.55f);

bool DetectionHUD::init() {
  shader_ok_ = shader_.compile(kVert, kFrag);
  if (!shader_ok_) std::fprintf(stderr, "[wosw] ERROR: DetectionHUD shader failed to compile\n");
  return shader_ok_;
}

void DetectionHUD::seg(const Vec3f& a, const Vec3f& b, float r, float gg, float bb,
                       float aa0, float aa1) {
  mesh_.vertex(a.x, a.y, a.z); mesh_.color(r, gg, bb, aa0);
  mesh_.vertex(b.x, b.y, b.z); mesh_.color(r, gg, bb, aa1);
}

// Shift every vertex of m by d in world space (cheap per-pass offset for the chromatic split).
// Operates on the al::VAOMesh CPU vertex buffer (std::vector<Vec3f>); caller calls m.update() after.
// Order-independent and stateless across frames (caller always restores with -d), so dome-deterministic.
void DetectionHUD::pushOffset_(al::VAOMesh& m, const Vec3f& d) {
  auto& v = m.vertices();
  for (auto& p : v) { p.x += d.x; p.y += d.y; p.z += d.z; }
}

void DetectionHUD::drawReticle(Graphics& g, const Vec3f& center, const Vec3f& camR,
                               const Vec3f& camU, int node, double simTime,
                               float age, float conf01, float bright) {
  if (!shader_ok_ || bright <= 0.f) return;
  const uint32_t nid = uint32_t(node < 0 ? 0 : node);
  const float t = float(simTime);

  // --- LOCK-ON acquire: the box springs from oversized to snug over ~0.9s of the target's life ---
  // pure function of synced age, so every node sees the same acquisition.
  float acq = age / 0.9f; acq = acq < 0.f ? 0.f : (acq > 1.f ? 1.f : acq);
  acq = acq * acq * (3.f - 2.f * acq);                 // smoothstep ease-in
  // a tiny settle wobble that decays, deterministic from age (reads as the tracker "snapping")
  const float wob = (1.f - acq) * 0.18f * std::sin(age * 22.f + hash01(nid, 7u) * 6.28f);
  const float ph  = hash01(nid, 3u) * 6.2832f;          // per-target phase so boxes are out of sync

  mesh_.reset();
  mesh_.primitive(Mesh::LINES);

  const Vec3f R = camR, U = camU;
  // box half-size in world units: starts 1.9x, springs to 1.0x (+wobble). boxHalf_ is the caller's
  // per-target world half-size, set immediately before this call.
  const float bh = boxHalf_ * (1.9f - 0.9f * acq + wob);
  const Vec3f rx = R * bh, uy = U * bh;
  const Vec3f TL = center - rx + uy, TR = center + rx + uy;
  const Vec3f BL = center - rx - uy, BR = center + rx - uy;
  const float aLock = 0.35f + 0.65f * acq;              // brackets brighten as they lock

  // --- four CORNER BRACKETS (L shapes) — the bounding box, drawn as 8 short segments ---
  const float L = bracketLen_ * bh * 2.f;               // arm length
  // TL
  seg(TL, TL + R * L, kHud.x, kHud.y, kHud.z, aLock, aLock * 0.5f);
  seg(TL, TL - U * L, kHud.x, kHud.y, kHud.z, aLock, aLock * 0.5f);
  // TR
  seg(TR, TR - R * L, kHud.x, kHud.y, kHud.z, aLock, aLock * 0.5f);
  seg(TR, TR - U * L, kHud.x, kHud.y, kHud.z, aLock, aLock * 0.5f);
  // BL
  seg(BL, BL + R * L, kHud.x, kHud.y, kHud.z, aLock, aLock * 0.5f);
  seg(BL, BL + U * L, kHud.x, kHud.y, kHud.z, aLock, aLock * 0.5f);
  // BR
  seg(BR, BR - R * L, kHud.x, kHud.y, kHud.z, aLock, aLock * 0.5f);
  seg(BR, BR + U * L, kHud.x, kHud.y, kHud.z, aLock, aLock * 0.5f);

  // --- CROSSHAIR reticle at the box center (acquire fades it in) ---
  const float cl = 0.22f * bh;
  seg(center - R * cl, center + R * cl, kHud.x, kHud.y, kHud.z, 0.0f, acq * 0.8f);
  seg(center - U * cl, center + U * cl, kHud.x, kHud.y, kHud.z, acq * 0.8f, 0.0f);

  // --- vertical SCAN LINE sweeping top->bottom at a fixed rate (only once locked) ---
  // pure function of simTime: y = 1 - frac(t*0.6 + ph). identical on every node.
  if (acq > 0.6f) {
    float sy = 1.f - (std::fmod(t * 0.6f + ph, 1.f));   // 0..1 top->bottom
    const Vec3f scl = center + U * (bh * (2.f * sy - 1.f));
    seg(scl - rx, scl + rx, kHud.x, kHud.y, kHud.z, 0.0f, 0.6f * acq);
    seg(scl + rx, scl - rx, kHud.x, kHud.y, kHud.z, 0.6f * acq, 0.0f); // bidirectional fade
  }

  // --- tracked FEATURE DOTS: short crosses at hashed positions, re-rolled on a 10Hz time grid ---
  // deterministic jitter: position = hash(node*step + dot, floor(t*10)). No rand, no clock.
  const int ND = int(dotCount_);
  const uint32_t bin = uint32_t(std::floor(t * 10.0));   // 10 Hz quantized grid (synced t)
  const float ds = 0.10f * bh;
  for (int d = 0; d < ND; ++d) {
    const uint32_t k = nid * 131u + uint32_t(d);
    const float fx = hash01(k, bin) * 2.f - 1.f;          // -1..1 inside the box
    const float fy = hash01(k, bin + 9173u) * 2.f - 1.f;
    const Vec3f dp = center + R * (fx * bh * 0.82f) + U * (fy * bh * 0.82f);
    const float da = (0.4f + 0.6f * hash01(k, bin + 55u)) * acq;
    seg(dp - R * ds, dp + R * ds, kHud.x, kHud.y, kHud.z, da, da);
    seg(dp - U * ds, dp + U * ds, kHud.x, kHud.y, kHud.z, da, da);
  }

  // --- flickering CONFIDENCE BAR along the bottom edge ---
  // base from node conf01; flicker = hash dropout on a 12Hz grid (reads as a live classifier).
  const uint32_t cbin = uint32_t(std::floor(t * 12.0));
  float conf = conf01 * (0.85f + 0.15f * hash01(nid, cbin));       // 12Hz micro-flicker
  conf = 0.55f + 0.44f * conf;                                     // keep it high (looks confident)
  conf = conf < 0.f ? 0.f : (conf > 1.f ? 1.f : conf);
  const Vec3f barL = BL + U * (0.06f * bh);
  const Vec3f barFull = barL + R * (2.f * bh);
  const Vec3f barEnd = barL + (barFull - barL) * conf;
  seg(barL, barEnd, kHot.x, kHot.y, kHot.z, 0.85f * acq, 0.85f * acq);
  // a faint full-width track behind it
  seg(barL, barFull, kHud.x, kHud.y, kHud.z, 0.12f * acq, 0.12f * acq);

  mesh_.update();
  g.depthTesting(false); g.blending(true); g.blendAdd();
  g.shader(shader_);
  g.shader().uniform("uIntensity", bright);

  // --- CRT / CHROMATIC-ABERRATION COMPOSITE (in-layer, no FBO) -------------------------------
  // Draw the same geometry THREE times, once per colour channel, each shifted a few world-px along
  // the billboard basis (camR/camU). Split magnitude/direction + scanline + tear/dropout are a
  // deterministic hash of (node, floor(t*9)), so it shimmers/tears identically on every dome node.
  const uint32_t gbin = uint32_t(std::floor(t * 9.0));            // 9 Hz glitch grid (synced t)
  const float ga = hash01(nid, gbin) * 6.2832f;                  // split direction
  const float gm = 0.010f * bh * (0.6f + 0.8f * hash01(nid, gbin + 41u)); // split magnitude
  const bool  tear = hash01(nid, gbin + 991u) > 0.86f;          // ~1-in-7 quanta: a wider tear
  const float scan = 0.55f + (tear ? 0.35f : 0.0f);             // CRT scanline strength
  const Vec3f off  = (camR * std::cos(ga) + camU * std::sin(ga)) * (gm * (tear ? 3.2f : 1.0f));
  const bool  drop = tear && (hash01(nid, gbin + 7u) > 0.6f);   // green dropout on some tears
  const Vec3f passOff[3]  = { off, Vec3f(0,0,0), -off };        // R +off, G centred, B -off
  const Vec3f passMask[3] = { Vec3f(1,0,0), Vec3f(0,1,0), Vec3f(0,0,1) };
  g.shader().uniform("uScan", scan);
  g.lineWidth(2.0f);
  for (int p = 0; p < 3; ++p) {
    if (drop && p == 1) continue;                               // CRT green-channel dropout glitch
    g.shader().uniform("uMask", passMask[p]);
    pushOffset_(mesh_, passOff[p]); mesh_.update();             // shift every vertex for this channel
    g.draw(mesh_);
    pushOffset_(mesh_, -passOff[p]);                            // restore for the next pass
  }
  // reset mask/scan so later HUD draws are unaffected unless they set their own.
  g.shader().uniform("uMask", Vec3f(1,1,1));
  g.shader().uniform("uScan", 0.0f);
  g.lineWidth(1.f);
  g.blendTrans(); g.depthTesting(true);   // RESTORE
}

void DetectionHUD::drawSortStrip(Graphics& g, const Vec3f& camPos, const Vec3f& camFwd,
                                 const Vec3f& camR, const Vec3f& camU, const float* keys,
                                 const float* hue, int n, double simTime, float bright) {
  if (!shader_ok_ || bright <= 0.f || n <= 0) return;
  const float t = float(simTime);
  const int N = n > 64 ? 64 : n;

  // REWORK: the old strip sorted by cluster, but the 3-cluster galaxy gave near-uniform keys -> a
  // flat green line. Now each marker gets a DETERMINISTIC PER-MARKER key that drifts on its own
  // synced phase, so the keys are always well-spread and the order perpetually bubble-swaps. Drawn
  // as glowing VERTICAL BARS whose height = the marker's live key, in varied hues -> a running sort
  // that reads even with few clusters. Pure function of (keys[], hue[], simTime).
  int idx[64]; float sk[64], bar[64];
  for (int i = 0; i < N; ++i) {
    idx[i] = i;
    const float osc = 0.5f + 0.5f * std::sin(t * 0.5f + float(i) * 2.399963f);  // per-marker drift
    sk[i]  = keys[i] * 0.35f + 0.65f * osc;                    // visibly moving sort key
    bar[i] = 0.15f + 0.85f * osc;                             // bar height 0.15..1.0
  }
  std::stable_sort(idx, idx + N, [&](int a, int b) { return sk[a] < sk[b]; });

  // head-locked row of vertical bars low-centre, in front of the camera.
  const float DIST = 4.5f, Y = -1.30f, HALFW = 1.7f, BARW = 0.030f, BARH = 0.42f;
  const Vec3f base = camPos + camFwd * DIST + camU * Y;
  // glitch hash on the same 9 Hz grid, keyed to a fixed pseudo-node so the strip tears with the reticles.
  const uint32_t gbin = uint32_t(std::floor(t * 9.0));
  const float ga = hash01(7777u, gbin) * 6.2832f;
  const float gm = 0.008f * (0.6f + 0.8f * hash01(7777u, gbin + 41u));
  const bool  tear = hash01(7777u, gbin + 991u) > 0.86f;
  const float scan = 0.55f + (tear ? 0.35f : 0.0f);
  const Vec3f off  = (camR * std::cos(ga) + camU * std::sin(ga)) * (gm * (tear ? 3.2f : 1.0f));

  mesh_.reset();
  mesh_.primitive(Mesh::LINES);
  for (int slot = 0; slot < N; ++slot) {
    const int m = idx[slot];                                  // which marker sits at this slot now
    const float x = (N > 1) ? (float(slot) / float(N - 1) * 2.f - 1.f) : 0.f;
    const Vec3f c = base + camR * (x * HALFW);
    const float h = hue[m];
    const float r = 0.5f + 0.5f * std::sin(6.2832f * h);
    const float gg = 0.5f + 0.5f * std::sin(6.2832f * h + 2.094f);
    const float bb = 0.5f + 0.5f * std::sin(6.2832f * h + 4.188f);
    const float bv = bar[m] * BARH;                           // bar height = its live key
    const Vec3f rx = camR * BARW;
    seg(c, c + camU * bv, r, gg, bb, 0.85f, 0.10f);                       // bright base -> faint tip
    seg(c - rx, c - rx + camU * bv, r, gg, bb, 0.25f, 0.04f);            // faint left flank (bloom)
    seg(c + rx, c + rx + camU * bv, r, gg, bb, 0.25f, 0.04f);            // faint right flank (bloom)
    seg(c - rx, c + rx, r, gg, bb, 0.5f, 0.5f);                          // base cap nub
  }
  seg(base - camR * HALFW, base + camR * HALFW, kHud.x, kHud.y, kHud.z, 0.18f, 0.18f);  // baseline rail

  g.depthTesting(false); g.blending(true); g.blendAdd();
  g.shader(shader_);
  g.shader().uniform("uIntensity", bright);
  g.shader().uniform("uScan", scan);
  const Vec3f passOff[3]  = { off, Vec3f(0,0,0), -off };
  const Vec3f passMask[3] = { Vec3f(1,0,0), Vec3f(0,1,0), Vec3f(0,0,1) };
  g.lineWidth(2.0f);
  for (int p = 0; p < 3; ++p) {
    g.shader().uniform("uMask", passMask[p]);
    pushOffset_(mesh_, passOff[p]); mesh_.update();
    g.draw(mesh_);
    pushOffset_(mesh_, -passOff[p]);
  }
  g.shader().uniform("uMask", Vec3f(1,1,1));
  g.shader().uniform("uScan", 0.0f);
  g.lineWidth(1.f);
  g.blendTrans(); g.depthTesting(true);   // RESTORE
}

}  // namespace wosw
