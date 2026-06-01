#include "viz/DreamLayer.hpp"
#include "viz/EmergencePlayer.hpp"
#include "al/graphics/al_Image.hpp"
#include "al/graphics/al_OpenGL.hpp"
#include <cmath>
#include <cstdio>
#include <algorithm>
#include <vector>

namespace wosw {

using namespace al;

// The dreams CYCLE through ALL of them over time, independent of the camera: a sliding window of
// ON_SCREEN dreams is active at once — each appears (forming from noise), holds, then hands off to the
// next — so every one of the N dreams gets shown, ~ON_SCREEN at a time, over a full CYCLE = N*STEP_T s.
static constexpr int   ON_SCREEN = 8;     // dreams visible at once (artist: 8 on screen)
static constexpr float STEP_T    = 3.0f;  // seconds between one dream appearing and the next (full cycle = N*STEP_T)
static constexpr float WHALF = 0.60f;     // world half-size of a (square) dream billboard (~3/4 the original 0.80)
// Activation order is STRIDED (coprime with the dream count) so the ON_SCREEN active dreams are always
// from DIFFERENT source artworks — never several variants (melt/outpaint/shadow) of the same image at once.
static constexpr int   STRIDE = 7;        // gcd(7,192)=1 -> a bijection; 7 indices apart = different source

static inline float smooth01(float x) {
  x = x < 0.f ? 0.f : (x > 1.f ? 1.f : x);
  return x * x * (3.f - 2.f * x);
}
// deterministic [0,1) phase offset per dream node, so they re-dream out of unison (no RNG/wallclock)
static inline float nodePhase(int node) {
  uint32_t h = uint32_t(node) * 2654435761u; h ^= h >> 15; h *= 2246822519u; h ^= h >> 13;
  return float(h & 0xffffffu) / float(0x1000000u);
}

static const char* kVert = R"GLSL(
#version 330
uniform mat4 al_ModelViewMatrix;
uniform mat4 al_ProjectionMatrix;
layout (location = 0) in vec3 vertexPosition;
layout (location = 2) in vec2 vertexTexCoord;
out vec2 vuv;
void main() {
  vuv = vertexTexCoord;
  gl_Position = al_ProjectionMatrix * al_ModelViewMatrix * vec4(vertexPosition, 1.0);
}
)GLSL";

// the dream image, softened to a disc by a radial vignette (so it reads as a glowing confabulation,
// not a hard photo-square), with a faint op tint (melt warm / outpaint neutral / shadow cool).
// DREAM DIGITAL-TV / VHS GLITCH — done ENTIRELY inside this billboard fragment shader by sampling
// THIS dream's own atlas cell at offset UVs (sampleCell clamps to the cell so no neighbour-tile
// bleed). NO FBO/fullscreen pass, so it composes with the omni/dome warp. All 'randomness' is
// h11/h21 integer-style hashes of (floor(uTime*RATE), uSeed, row) + sin(uTime,...): zero GL noise,
// zero wall-clock -> bit-identical on every dome node. Bursts are gated by floor(uTime*RATE) so the
// tear is intermittent, not constant mush. Existing vignette + uOp tint preserved.
static const char* kFrag = R"GLSL(
#version 330
uniform sampler2D tex;
uniform float uAlpha;
uniform vec4  uUV;      // atlas cell (u0,v0,u1,v1)
uniform float uOp;      // 0 melt, 1 outpaint, 2 shadow
uniform float uTime;    // synced simTime -> deterministic glitch clock
uniform float uSeed;    // per-dream index -> per-dream glitch phase
in vec2 vuv;
out vec4 fragColor;

float h11(float n){ return fract(sin(n) * 43758.5453123); }
float h21(vec2 p){ return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453123); }

// sample THIS dream's atlas cell at local uv (0..1), clamped to the cell so offset UVs never bleed
// into a neighbouring tile.
vec3 sampleCell(vec2 uv){
  uv = clamp(uv, 0.0, 1.0);
  vec2 t = mix(uUV.xy, uUV.zw, uv);
  return texture(tex, t).rgb;
}

void main() {
  // ---- glitch clock: a burst fires only on quanta whose hash crosses a threshold -------------
  float RATE = 7.0;
  float tick = floor(uTime * RATE);
  float gate = h21(vec2(tick, uSeed));                 // per-(quantum,dream) gate
  float burst = smoothstep(0.82, 0.98, gate);          // 0 most frames; >0 during a burst
  float ph    = fract(uTime * RATE);                   // 0..1 position inside the quantum
  float env   = burst * sin(ph * 3.14159265);          // shape the burst over its window

  // ---- per-row line-tear / block displacement (a few horizontal bands jump sideways) ---------
  float row   = floor(vuv.y * 18.0);
  float jr    = h21(vec2(row, tick + uSeed));
  float tearAmt = step(0.86, jr) * env * 0.06 * (jr - 0.5) * 2.0;  // only some rows, only in burst

  // ---- RGB channel split: horizontal offset, magnitude rides the burst envelope -------------
  float split = (0.004 + 0.020 * env) * (0.5 + 0.5 * sin(uTime * 2.0 + uSeed));
  vec2  uv    = vuv + vec2(tearAmt, 0.0);
  float rC = sampleCell(uv + vec2( split, 0.0)).r;
  float gC = sampleCell(uv                     ).g;
  float bC = sampleCell(uv + vec2(-split, 0.0)).b;
  vec3  c  = vec3(rC, gC, bC);

  // ---- chroma snow: a wash of hashed noise that only shows during a burst --------------------
  float snow = h21(floor(uv * 220.0) + vec2(tick, uSeed));
  c = mix(c, vec3(snow), env * 0.18);

  // ---- scanlines (in-shader, per-fragment; NOT a post pass) + dropout/brightness flicker -----
  float scan = 1.0 - 0.25 * env * (0.5 - 0.5 * cos(vuv.y * 220.0));
  float drop = 1.0 - step(0.96, h21(vec2(tick + 13.0, uSeed))) * env * 0.7;  // occasional dim-out
  c *= scan * drop;

  // ---- preserved vignette + op tint ----------------------------------------------------------
  vec2 d = vuv - 0.5;
  float vig = smoothstep(0.5, 0.34, length(d));        // soft circular falloff
  vec3 tint = (uOp < 0.5) ? vec3(1.04, 0.99, 0.92)     // melt: warm double-exposure
            : (uOp < 1.5) ? vec3(1.0, 1.0, 1.0)        // outpaint: neutral
                          : vec3(0.92, 0.95, 1.04);    // shadow: cool absence
  float a = uAlpha * vig;
  if (a <= 0.004) discard;
  fragColor = vec4(c * tint, a);
}
)GLSL";

bool DreamLayer::loadAtlas(const std::string& path) {
  Image im;
  if (!im.load(path)) return false;
  auto& a = im.array();
  if (a.empty()) return false;
  atlasW_ = int(im.width()); atlasH_ = int(im.height());
  GLint maxTex = 0; glGetIntegerv(GL_MAX_TEXTURE_SIZE, &maxTex);   // allolib doesn't check; a too-big atlas goes silently black
  if (maxTex > 0 && (atlasW_ > maxTex || atlasH_ > maxTex))
    std::fprintf(stderr, "[wosw] WARNING dream_atlas %dx%d EXCEEDS GL_MAX_TEXTURE_SIZE %d on this node — will be blank (lower COLS in bake_dreams.py)\n",
                 atlasW_, atlasH_, int(maxTex));
  tex_.create2D(atlasW_, atlasH_, Texture::RGBA8, Texture::RGBA, Texture::UBYTE);
  tex_.submit(&a[0], Texture::RGBA, Texture::UBYTE);
  tex_.filter(Texture::LINEAR);
  tex_.wrap(Texture::CLAMP_TO_EDGE, Texture::CLAMP_TO_EDGE, Texture::CLAMP_TO_EDGE);
  return true;
}

void DreamLayer::loadPos(const std::string& assetDir) {
  const std::string bases[] = { assetDir + "/../v2", "../../v2", "reagency/v2",
                                "MAT201B_Projects/reagency/v2", assetDir };
  for (const auto& b : bases) {
    FILE* f = std::fopen((b + "/dream_pos.txt").c_str(), "r");
    if (!f) continue;
    char line[512], did[256];
    int node, col, row, op; float x, y, z;
    while (std::fgets(line, sizeof(line), f)) {
      if (line[0] == '#') continue;
      if (std::sscanf(line, "%255s %d %f %f %f %d %d %d", did, &node, &x, &y, &z, &col, &row, &op) == 8)
        dreams_.push_back({Vec3f(x, y, z), node, col, row, op});
    }
    std::fclose(f);
    break;
  }
}

bool DreamLayer::init(const std::string& assetDir) {
  const std::string candidates[] = {
    assetDir + "/../v2/dream_atlas.png", "../../v2/dream_atlas.png",
    "reagency/v2/dream_atlas.png", "MAT201B_Projects/reagency/v2/dream_atlas.png",
    assetDir + "/dream_atlas.png",
  };
  tex_ok_ = false;
  for (const auto& c : candidates)
    if (loadAtlas(c)) { tex_ok_ = true;
      std::fprintf(stderr, "[wosw] dream atlas loaded from %s\n", c.c_str()); break; }
  if (tex_ok_) { cols_ = atlasW_ / cell_; if (cols_ < 1) cols_ = 1; loadPos(assetDir); }
  else std::fprintf(stderr, "[wosw] dream_atlas.png not found — dreams off (bake v2/bake_dreams.py)\n");
  shader_ok_ = shader_.compile(kVert, kFrag);
  std::fprintf(stderr, "[wosw] dreams in the galaxy: %zu\n", dreams_.size());
  return ready();
}

void DreamLayer::draw(Graphics& g, const Vec3f& camPos, const Vec3f& camRight,
                      const Vec3f& camUp, float globalAlpha, float time) {
  if (!ready() || globalAlpha <= 0.004f || dreams_.empty()) return;
  (void)camPos;                                   // selection is time-cycled, NOT camera-distance driven
  const int   N     = int(dreams_.size());
  const float WIN   = float(ON_SCREEN) * STEP_T;  // how long each dream stays on (ON_SCREEN active at once)
  const float CYCLE = float(N) * STEP_T;          // time to cycle through ALL N dreams

  // Per-dream render params, shared by the FORMING pass and the RESOLVED pass so they register. The
  // window slides through all N over CYCLE in a STRIDED order (slot k -> dream (k*STRIDE)%N), so the
  // ON_SCREEN active dreams are distinct source artworks. Purely a function of synced simTime + the
  // stable dream index, so it is identical on every dome node (no camera, no state).
  struct R { Vec3f c, right, up, rx, uy; float formPhase, emergeA, resolvedA; int idx; };
  std::vector<R> rs; rs.reserve(ON_SCREEN + 1);
  for (int k = 0; k < N; ++k) {
    float age = std::fmod(time - float(k) * STEP_T, CYCLE);     // seconds since slot k turned on
    if (age < 0.f) age += CYCLE;
    if (age >= WIN) continue;                                   // outside its on-window
    const int di = (k * STRIDE) % N;                            // strided -> different source each slot
    const Dream& d = dreams_[di];
    const float u  = age / WIN;                                 // 0 = just appeared .. 1 = about to leave
    const float ph = nodePhase(d.node) * 6.2831853f;
    // appear -> hold -> leave envelope (+ a gentle shimmer so it's never dead-still)
    float env = smooth01(u / 0.15f) * smooth01((1.f - u) / 0.20f) * globalAlpha;
    env *= 0.85f + 0.15f * std::sin(time * 1.3f + ph);
    if (env <= 0.01f) continue;
    // FORM from noise -> image over the first ~60% of the window, then hold the resolved image
    const float formPhase = smooth01(u / 0.60f);
    const float resolvedW = smooth01((formPhase - 0.55f) / 0.45f);
    const float breathe = WHALF * (1.f + 0.08f * std::sin(time * 0.7f + ph));
    const float wob = 0.10f * std::sin(time * 0.35f + ph);
    R r; r.idx = di; r.formPhase = formPhase;
    r.right = camRight * std::cos(wob) + camUp * std::sin(wob);
    r.up    = camUp * std::cos(wob) - camRight * std::sin(wob);
    r.rx = r.right * breathe; r.uy = r.up * breathe;
    r.c  = d.pos + camUp * (0.04f * std::sin(time * 0.5f + ph));   // at its galaxy node position
    r.emergeA = env * (1.f - resolvedW); r.resolvedA = env * resolvedW;
    rs.push_back(r);
  }
  if (rs.empty()) return;

  // PASS 1 — the FORMING latents (EmergencePlayer owns its own atlas/shader; self-contained per call).
  if (emerge_)
    for (const auto& r : rs) {
      const Dream& d = dreams_[r.idx];
      if (r.emergeA > 0.01f && emerge_->has(d.node))
        emerge_->draw(g, r.c, r.right, r.up, d.node, r.formPhase, r.emergeA);
    }

  // PASS 2 — the RESOLVED dreams (our atlas + VHS-glitch shader, bound once).
  g.depthTesting(false); g.blending(true); g.blendTrans();
  g.shader(shader_);
  tex_.bind(0);
  g.shader().uniform("tex", 0);
  for (const auto& r : rs) {
    if (r.resolvedA <= 0.01f) continue;
    const Dream& d = dreams_[r.idx];
    const Vec3f BL = r.c - r.rx - r.uy, BR = r.c + r.rx - r.uy, TL = r.c - r.rx + r.uy, TR = r.c + r.rx + r.uy;
    mesh_.reset(); mesh_.primitive(Mesh::TRIANGLE_STRIP);
    mesh_.vertex(BL.x, BL.y, BL.z); mesh_.texCoord(0.f, 1.f);
    mesh_.vertex(BR.x, BR.y, BR.z); mesh_.texCoord(1.f, 1.f);
    mesh_.vertex(TL.x, TL.y, TL.z); mesh_.texCoord(0.f, 0.f);
    mesh_.vertex(TR.x, TR.y, TR.z); mesh_.texCoord(1.f, 0.f);
    mesh_.update();
    const float u0 = float(d.col * cell_) / atlasW_, u1 = float((d.col + 1) * cell_) / atlasW_;
    const float v0 = 1.f - float((d.row + 1) * cell_) / atlasH_;
    const float v1 = 1.f - float(d.row * cell_) / atlasH_;
    g.shader().uniform("uUV", Vec4f(u0, v0, u1, v1));
    g.shader().uniform("uAlpha", r.resolvedA);
    g.shader().uniform("uOp", float(d.op));
    g.shader().uniform("uTime", time);          // synced simTime -> dome-deterministic glitch clock
    g.shader().uniform("uSeed", float(d.node));  // per-dream-node glitch phase (stable across shortlisting)
    g.draw(mesh_);
  }
  tex_.unbind(0);
  g.depthTesting(true);
}

}  // namespace wosw
