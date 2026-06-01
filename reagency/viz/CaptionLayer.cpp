#include "viz/CaptionLayer.hpp"
#include "al/graphics/al_Image.hpp"
#include <cmath>
#include <cstdio>

namespace wosw {

using namespace al;

// --- schedule tunables ---
static constexpr float TYPE_CPS = 24.f;   // characters per second (human typing pace)
static constexpr float GAP_T    = 1.6f;   // pause between captions
static constexpr float HOLD_T   = 4.2f;   // hold the fully-typed line (readable)
static constexpr float FADE_T   = 1.2f;   // fade out

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

// Typewriter: show ink left of the reveal edge; a soft cursor bar AT the edge (the human hand
// writing the staged "I"). Atlas is warm-white text on transparent (alpha = glyph coverage).
static const char* kFrag = R"GLSL(
#version 330
uniform sampler2D tex;
uniform float uAlpha;     // caption fade
uniform vec4  uUV;        // atlas cell (u0,v0,u1,v1)
uniform float uRevealX;   // [0,1] reveal edge in cell-u (left-to-right)
uniform float uCursor;    // 0..1 cursor intensity (0 when line already finished)
uniform float uGlitch;    // v2 T10: secondary chromatic wobble on the INK only (affect; never the cursor)
in vec2 vuv;
out vec4 fragColor;
void main() {
  vec2 t = mix(uUV.xy, uUV.zw, vuv);
  vec4 c = texture(tex, t);
  float ink = (vuv.x <= uRevealX) ? c.a : 0.0;                 // only the typed-so-far ink
  float bar = (1.0 - smoothstep(0.0, 0.010, abs(vuv.x - uRevealX)))   // thin vertical bar
            * step(abs(vuv.y - 0.5), 0.40);                            // ~80% of cell height
  float cur = uCursor * bar;
  vec3 rgb = c.rgb;
  // glitch = affect only, scoped to the typed ink: a faint red-push / blue-pull unease. It does NOT
  // touch the cursor (the human-authorship mark stays pure) nor the reveal (WHAT is typed is fixed).
  if (uGlitch > 0.001) { rgb.r += uGlitch * 0.25 * ink; rgb.b -= uGlitch * 0.15 * ink; }
  rgb += vec3(cur);
  float a = max(ink, cur) * uAlpha;
  if (a <= 0.002) discard;
  fragColor = vec4(rgb, a);
}
)GLSL";

bool CaptionLayer::loadAtlas(const std::string& path) {
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

void CaptionLayer::loadIndex(const std::string& assetDir) {
  const std::string bases[] = { assetDir + "/../v2", "../../v2", "reagency/v2",
                                "MAT201B_Projects/reagency/v2", assetDir };
  for (const auto& b : bases) {
    FILE* f = std::fopen((b + "/captions_index.txt").c_str(), "r");
    if (!f) continue;
    char line[512];
    int cid, person, act, row, chars; float ink;
    while (std::fgets(line, sizeof(line), f)) {
      if (line[0] == '#') continue;
      if (std::sscanf(line, "%d %d %d %d %d %f", &cid, &person, &act, &row, &chars, &ink) == 6)
        lines_.push_back({cid, person, act, row, chars, ink});
    }
    std::fclose(f);
    break;
  }
  // group consecutive line-cells of the same captionId into captions
  for (int i = 0; i < int(lines_.size()); ) {
    const int cid = lines_[i].caption;
    Caption cap{lines_[i].person, lines_[i].act, 0, {}};
    int j = i;
    for (; j < int(lines_.size()) && lines_[j].caption == cid; ++j) {
      cap.lineIdx.push_back(j);
      cap.totalChars += lines_[j].chars;
    }
    captions_.push_back(cap);
    i = j;
  }
}

bool CaptionLayer::init(const std::string& assetDir) {
  const std::string candidates[] = {
    assetDir + "/../v2/captions_atlas.png", "../../v2/captions_atlas.png",
    "reagency/v2/captions_atlas.png", "MAT201B_Projects/reagency/v2/captions_atlas.png",
    assetDir + "/captions_atlas.png",
  };
  tex_ok_ = false;
  for (const auto& c : candidates)
    if (loadAtlas(c)) { tex_ok_ = true;
      std::fprintf(stderr, "[wosw] caption atlas loaded from %s\n", c.c_str()); break; }
  if (tex_ok_) loadIndex(assetDir);
  else std::fprintf(stderr, "[wosw] captions_atlas.png not found — the voice is silent (bake v2/bake_captions.py)\n");
  shader_ok_ = shader_.compile(kVert, kFrag);
  std::fprintf(stderr, "[wosw] caption voice: %zu captions / %zu line-cells\n",
               captions_.size(), lines_.size());
  return ready();
}

int CaptionLayer::pickNext(int act) {
  const int n = int(captions_.size());
  if (n == 0) return -1;
  for (int k = 0; k < n; ++k) {                     // pass 1: a caption written for this act
    int idx = (actCursor_ + k) % n;
    if (captions_[idx].person == 1) continue;       // YOU is reserved for the Act-IV turn (forceYou)
    if (captions_[idx].act == act) { actCursor_ = (idx + 1) % n; return idx; }
  }
  for (int k = 0; k < n; ++k) {                     // pass 2: the SOLO line (act 0) as the fallback
    int idx = (actCursor_ + k) % n;
    if (captions_[idx].act == 0) { actCursor_ = (idx + 1) % n; return idx; }
  }
  return -1;
}

bool CaptionLayer::forceYou() {
  if (!ready() || !meShown_) return false;          // confess-first: never YOU before a ME this cycle
  const int n = int(captions_.size());
  for (int k = 0; k < n; ++k) {
    int idx = (actCursor_ + k) % n;
    if (captions_[idx].person == 1) {               // the staged YOU line
      cur_ = idx; typed_ = 0.f; alpha_ = 1.f; st_ = TYPING; phase_ = 0.f;
      actCursor_ = (idx + 1) % n;
      return true;
    }
  }
  return false;
}

void CaptionLayer::update(int act, float dt) {
  if (!ready() || captions_.empty()) return;
  curAct_ = act;
  if (act == 1 && lastAct_ > 1) meShown_ = false;   // reset confess-first guard at the loop seam
  lastAct_ = act;
  blinkClock_ += dt;
  switch (st_) {
    case GAP:
      phase_ += dt;
      if (phase_ >= GAP_T) {
        const int n = pickNext(curAct_);
        if (n >= 0) { cur_ = n; typed_ = 0.f; alpha_ = 1.f; st_ = TYPING;
                      if (captions_[n].person == 0) meShown_ = true; }   // ME/SOLO satisfies confess-first
        phase_ = 0.f;
      }
      break;
    case TYPING: {
      // jittered pace -> reads as a human hand, not a metronome
      typed_ += dt * TYPE_CPS * (0.80f + 0.40f * std::fabs(std::sin(typed_ * 1.7f)));
      const float total = float(captions_[cur_].totalChars);
      if (typed_ >= total) { typed_ = total; st_ = HOLDING; phase_ = 0.f; }
      break;
    }
    case HOLDING:
      phase_ += dt;
      if (phase_ >= HOLD_T) { st_ = FADING; phase_ = 0.f; }
      break;
    case FADING:
      phase_ += dt;
      alpha_ = 1.f - phase_ / FADE_T;
      if (phase_ >= FADE_T) { alpha_ = 0.f; cur_ = -1; st_ = GAP; phase_ = 0.f; }
      break;
  }
}

void CaptionLayer::draw(Graphics& g, const Vec3f& camPos, const Vec3f& camFwd,
                        const Vec3f& camRight, const Vec3f& camUp, float glitch) {
  if (!ready() || alpha_ <= 0.002f || cur_ < 0) return;
  const Caption& cap = captions_[cur_];
  const int K = int(cap.lineIdx.size());
  if (K == 0) return;

  const float DIST = 5.0f, VOFF = -1.45f, WHALF = 2.0f, LINEH = 0.42f;
  // person register, shown NON-HUE: a small vertical nudge (ME low .. US high) + the stack offset.
  static const float personY[4] = { -0.10f, 0.02f, 0.10f, 0.16f };
  const float py = personY[(cap.person >= 0 && cap.person < 4) ? cap.person : 0];
  const Vec3f center = camPos + camFwd * DIST + camUp * (VOFF + py);
  const float blink = (std::fmod(blinkClock_, 0.7f) < 0.45f) ? 1.f : 0.25f;

  g.depthTesting(false); g.blending(true); g.blendTrans();
  g.shader(shader_);
  tex_.bind(0);
  g.shader().uniform("tex", 0);
  g.shader().uniform("uGlitch", glitch);   // T10: secondary chromatic wobble on the ink (affect only)

  int cum = 0;
  for (int i = 0; i < K; ++i) {
    const Line& ln = lines_[cap.lineIdx[i]];
    const int start = cum, end = cum + ln.chars;
    cum = end;
    if (typed_ <= float(start)) continue;                       // not typed yet
    const bool typing = typed_ < float(end);
    const float frac = typing ? (typed_ - float(start)) / float(ln.chars) : 1.f;
    const float revX = ln.ink * frac;

    const float yo = (float(K - 1) * 0.5f - float(i)) * LINEH;   // top line first
    const Vec3f c  = center + camUp * yo;
    const Vec3f rx = camRight * WHALF, uy = camUp * (LINEH * 0.5f);
    const Vec3f BL = c - rx - uy, BR = c + rx - uy, TL = c - rx + uy, TR = c + rx + uy;
    mesh_.reset(); mesh_.primitive(Mesh::TRIANGLE_STRIP);
    mesh_.vertex(BL.x, BL.y, BL.z); mesh_.texCoord(0.f, 1.f);
    mesh_.vertex(BR.x, BR.y, BR.z); mesh_.texCoord(1.f, 1.f);
    mesh_.vertex(TL.x, TL.y, TL.z); mesh_.texCoord(0.f, 0.f);
    mesh_.vertex(TR.x, TR.y, TR.z); mesh_.texCoord(1.f, 0.f);
    mesh_.update();
    const float v0 = 1.f - float((ln.row + 1) * ch_) / atlasH_;
    const float v1 = 1.f - float(ln.row * ch_) / atlasH_;
    g.shader().uniform("uUV", Vec4f(0.f, v0, 1.f, v1));
    g.shader().uniform("uAlpha", alpha_);
    g.shader().uniform("uRevealX", revX);
    g.shader().uniform("uCursor", typing ? blink : 0.f);
    g.draw(mesh_);
  }
  tex_.unbind(0);
  g.depthTesting(true);
}

}  // namespace wosw
