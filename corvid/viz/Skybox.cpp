#include "viz/Skybox.hpp"
#include "al/graphics/al_Image.hpp"
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <vector>

namespace corvid {

using namespace al;

// GLSL embedded as string literals (no per-node file I/O — robust for the dome).
// Uses allolib's standard custom-shader interface: al_ModelViewMatrix,
// al_ProjectionMatrix, vertexPosition (location 0).
static const char* kVert = R"GLSL(
#version 330
uniform mat4 al_ModelViewMatrix;
uniform mat4 al_ProjectionMatrix;
layout (location = 0) in vec3 vertexPosition;
out vec3 vdir;
void main() {
  // sphere is centered on the eye and not rotated, so the local vertex position
  // IS the world-space view direction.
  vdir = normalize(vertexPosition);
  gl_Position = al_ProjectionMatrix * al_ModelViewMatrix * vec4(vertexPosition, 1.0);
}
)GLSL";

static const char* kFrag = R"GLSL(
#version 330
in vec3 vdir;
out vec4 fragColor;

uniform float uTime;
uniform float uEntropy;     // 0..1  : turbulence / incoherence ("hallucination")
uniform float uConfidence;  // 0..1  : crispness + brightness
uniform float uValue;       // squashed critic value -> overall luminance
uniform vec3  uActionColor; // hue nudge from the action distribution
uniform int   uPaletteN;
uniform sampler2D uPalette; // crow-derived palette, N x 1

// --- hash / value noise on the sphere of directions ---
float hash(vec3 p) {
  p = fract(p * 0.3183099 + 0.1);
  p *= 17.0;
  return fract(p.x * p.y * p.z * (p.x + p.y + p.z));
}
float vnoise(vec3 x) {
  vec3 i = floor(x), f = fract(x);
  f = f * f * (3.0 - 2.0 * f);
  float n000 = hash(i + vec3(0,0,0)), n100 = hash(i + vec3(1,0,0));
  float n010 = hash(i + vec3(0,1,0)), n110 = hash(i + vec3(1,1,0));
  float n001 = hash(i + vec3(0,0,1)), n101 = hash(i + vec3(1,0,1));
  float n011 = hash(i + vec3(0,1,1)), n111 = hash(i + vec3(1,1,1));
  return mix(mix(mix(n000,n100,f.x), mix(n010,n110,f.x), f.y),
             mix(mix(n001,n101,f.x), mix(n011,n111,f.x), f.y), f.z);
}
float fbm(vec3 p) {
  float a = 0.5, s = 0.0;
  for (int i = 0; i < 5; ++i) { s += a * vnoise(p); p *= 2.02; a *= 0.5; }
  return s;
}

vec3 palette(float t) {
  if (uPaletteN <= 0) return vec3(0.10, 0.11, 0.16);
  float u = clamp(t, 0.0, 1.0);
  return texture(uPalette, vec2(u, 0.5)).rgb;
}

void main() {
  vec3 d = normalize(vdir);

  // domain warp: stronger + faster with higher entropy ("churning hallucination")
  float warp = 0.35 + 1.4 * uEntropy;
  float spd  = 0.05 + 0.25 * uEntropy;
  vec3 q = d * (1.6 + uConfidence) + vec3(0.0, 0.0, uTime * spd);
  vec3 w = vec3(fbm(q + 11.5), fbm(q + 47.2), fbm(q + 83.1));
  float f = fbm(d * 2.2 + warp * w + uTime * spd * 0.5);

  // crow palette lookup, biased by value
  float v = 1.0 / (1.0 + exp(-uValue));            // squash to 0..1
  vec3 col = palette(f * 0.85 + 0.15 * v);

  // action-distribution hue nudge + entropy desaturation
  col = mix(col, col * (0.5 + uActionColor), 0.35);
  float sat = mix(0.55, 1.0, uConfidence);
  float lum = dot(col, vec3(0.299, 0.587, 0.114));
  col = mix(vec3(lum), col, sat);

  // luminance: brighter when confident / high-value; dim, smoky when uncertain
  float bright = (0.18 + 0.5 * v) * (0.6 + 0.4 * uConfidence);
  // faint banded "scan" to read as generative/multi-view
  bright *= 0.85 + 0.15 * sin(f * 12.0 + uTime * 0.7);
  col *= bright;

  fragColor = vec4(col, 1.0);
}
)GLSL";

// Average color of one image (RGBA8). Returns false on load failure.
static bool avgColor(const std::string& path, float& r, float& g, float& b) {
    Image im;
    if (!im.load(path)) return false;
    auto& a = im.array();
    if (a.empty()) return false;
    // stride-sample to keep it fast on big images
    size_t px = a.size() / 4;
    size_t stride = std::max<size_t>(1, px / 4096);
    double sr = 0, sg = 0, sb = 0; size_t n = 0;
    for (size_t i = 0; i < px; i += stride) {
        sr += a[i * 4 + 0]; sg += a[i * 4 + 1]; sb += a[i * 4 + 2]; ++n;
    }
    if (!n) return false;
    r = float(sr / n) / 255.f; g = float(sg / n) / 255.f; b = float(sb / n) / 255.f;
    return true;
}

bool Skybox::init(const std::string& crow_dir, float radius) {
    radius_ = radius;

    // inward-facing sphere: build, then flip winding by negating positions' scale
    // at draw time we disable culling, so winding is not critical.
    addSphere(sphere_, radius_, 64, 48);
    sphere_.update();

    shader_ok_ = shader_.compile(kVert, kFrag);

    // Build crow palette texture (N x 1 RGB8).
    std::vector<unsigned char> pix;
    namespace fs = std::filesystem;
    std::error_code ec;
    if (!crow_dir.empty() && fs::exists(crow_dir, ec)) {
        std::vector<std::string> files;
        for (auto& e : fs::directory_iterator(crow_dir, ec)) {
            if (!e.is_regular_file()) continue;
            auto p = e.path().string();
            auto ext = e.path().extension().string();
            std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
            if (ext == ".jpg" || ext == ".jpeg" || ext == ".png") files.push_back(p);
        }
        std::sort(files.begin(), files.end());
        // cap palette size; spread samples across the set
        const int MAXP = 64;
        int want = int(std::min<size_t>(files.size(), MAXP));
        for (int i = 0; i < want; ++i) {
            const std::string& f = files[size_t(i) * files.size() / std::max(1, want)];
            float r, g, b;
            if (avgColor(f, r, g, b)) {
                pix.push_back((unsigned char)(r * 255));
                pix.push_back((unsigned char)(g * 255));
                pix.push_back((unsigned char)(b * 255));
            }
        }
    }
    if (pix.empty()) {  // neutral corvid palette fallback (blue-greys)
        const unsigned char neutral[] = {18,20,28, 30,34,46, 52,58,74, 80,88,104, 120,128,150};
        pix.assign(neutral, neutral + sizeof(neutral));
    }
    palette_n_ = int(pix.size() / 3);
    palette_.create2D(palette_n_, 1, Texture::RGB8, Texture::RGB, Texture::UBYTE);
    palette_.submit(pix.data(), Texture::RGB, Texture::UBYTE);
    palette_.filter(Texture::LINEAR);
    palette_.wrap(Texture::CLAMP_TO_EDGE, Texture::CLAMP_TO_EDGE, Texture::CLAMP_TO_EDGE);

    return shader_ok_;
}

void Skybox::draw(Graphics& g, const Vec3d& eyePos, const ThoughtVector& tv, float t) {
    if (!shader_ok_) return;

    // action distribution -> a soft RGB nudge (first 3 vs last 3 action channels)
    float warmth = tv.action[4] + tv.action[3];     // food + predator -> warm
    float cool   = tv.action[0] + tv.action[2];     // align + cohere  -> cool
    Vec3f ac(0.6f + 0.6f * warmth, 0.6f, 0.6f + 0.6f * cool);

    g.depthTesting(false);          // background; never occlude agents
    g.culling(false);               // draw inside faces of the sphere (FILL is default)

    g.shader(shader_);
    g.shader().uniform("uTime", t);
    g.shader().uniform("uEntropy", tv.entropy);
    g.shader().uniform("uConfidence", tv.confidence);
    g.shader().uniform("uValue", tv.value);
    g.shader().uniform("uActionColor", ac);
    g.shader().uniform("uPaletteN", palette_n_);
    palette_.bind(0);
    g.shader().uniform("uPalette", 0);

    g.pushMatrix();
    g.translate(float(eyePos.x), float(eyePos.y), float(eyePos.z));
    g.draw(sphere_);
    g.popMatrix();

    palette_.unbind(0);
    g.depthTesting(true);
    g.culling(false);
}

} // namespace corvid
