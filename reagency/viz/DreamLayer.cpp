#include "viz/DreamLayer.hpp"
#include "al/graphics/al_Image.hpp"
#include <cmath>
#include <cstdio>

namespace wosw {

using namespace al;

// proximity reveal: a dream blooms in as you approach its node, fades by FAR.
static constexpr float NEAR = 1.6f, FAR = 6.5f;
static constexpr float WHALF = 0.80f;   // world half-size of a (square) dream billboard (SIZE SWAP: was 0.42 -> dreams now dominate)

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
        dreams_.push_back({Vec3f(x, y, z), col, row, op});
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
  if (!ready() || globalAlpha <= 0.004f) return;
  g.depthTesting(false); g.blending(true); g.blendTrans();
  g.shader(shader_);
  tex_.bind(0);
  g.shader().uniform("tex", 0);

  for (size_t di = 0; di < dreams_.size(); ++di) {
    const Dream& d = dreams_[di];
    const float ph = float(di) * 0.7f;                       // per-dream phase so they don't pulse in unison
    const float dist = (d.pos - camPos).mag();
    float a = (FAR - dist) / (FAR - NEAR);
    a = a < 0.f ? 0.f : (a > 1.f ? 1.f : a);
    a = a * a * (3.f - 2.f * a) * globalAlpha;
    a *= 0.80f + 0.20f * std::sin(time * 1.7f + ph);         // brightness shimmer — never fully still
    if (a <= 0.01f) continue;

    // alive: breathe (scale), bob (drift), and a gentle in-plane tilt wobble
    const float breathe = WHALF * (1.f + 0.10f * std::sin(time * 0.9f + ph));
    const float wob = 0.12f * std::sin(time * 0.4f + ph);
    const Vec3f R = camRight * std::cos(wob) + camUp * std::sin(wob);
    const Vec3f U = camUp * std::cos(wob) - camRight * std::sin(wob);
    const Vec3f rx = R * breathe, uy = U * breathe;
    const Vec3f c  = d.pos + camUp * (0.05f * std::sin(time * 0.6f + ph));
    const Vec3f BL = c - rx - uy, BR = c + rx - uy, TL = c - rx + uy, TR = c + rx + uy;
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
    g.shader().uniform("uAlpha", a);
    g.shader().uniform("uOp", float(d.op));
    g.shader().uniform("uTime", time);          // synced simTime -> dome-deterministic glitch clock
    g.shader().uniform("uSeed", float(di));      // per-dream index -> per-dream glitch phase
    g.draw(mesh_);
  }
  tex_.unbind(0);
  g.depthTesting(true);
}

}  // namespace wosw
