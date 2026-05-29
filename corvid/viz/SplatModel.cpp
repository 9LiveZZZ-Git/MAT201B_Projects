#include "viz/SplatModel.hpp"
#include "al/graphics/al_Image.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>

namespace corvid {

using namespace al;

// ---------------------------------------------------------------------------
// student.bin format (little-endian, produced by the offline distill tool):
//   char    magic[4]   = "CRVS"
//   int32   version    = 1
//   int32   d_in       (must equal 3 + COND)
//   int32   d_hidden
//   int32   d_out      (must equal 7 : dpos3 + dcol3 + dsigma1)
//   float32 W1[d_hidden*d_in], b1[d_hidden]
//   float32 W2[d_out*d_hidden], b2[d_out]
// A 2-layer tanh MLP applied per point: out = W2*tanh(W1*in + b1) + b2.
// (Few-step diffusion is folded into the offline distillation; at runtime this
// is a single conditioned forward per frame — the "1-step student".)
// ---------------------------------------------------------------------------

static const char* kVert = R"GLSL(
#version 330
uniform mat4 al_ModelViewMatrix;
uniform mat4 al_ProjectionMatrix;
uniform float uPointSize;
layout (location = 0) in vec3 vertexPosition;
layout (location = 1) in vec4 vertexColor;   // rgb + alpha(weight)
layout (location = 2) in vec2 vertexTexCoord; // x = sigma (0..1), y = unused
out vec4 vcol;
out float vsigma;
void main() {
  vcol = vertexColor;
  vsigma = max(vertexTexCoord.x, 0.05);
  gl_Position = al_ProjectionMatrix * al_ModelViewMatrix * vec4(vertexPosition, 1.0);
  gl_PointSize = uPointSize;
}
)GLSL";

static const char* kFrag = R"GLSL(
#version 330
in vec4 vcol;
in float vsigma;
out vec4 fragColor;
void main() {
  vec2 c = gl_PointCoord * 2.0 - 1.0;     // [-1,1]
  float r2 = dot(c, c);
  if (r2 > 1.0) discard;
  // radial Gaussian; vsigma shrinks the effective splat within the sprite
  float g = exp(-r2 / (2.0 * vsigma * vsigma));
  fragColor = vec4(vcol.rgb, vcol.a * g);   // additive blend in draw()
}
)GLSL";

// value noise (matches the skybox feel, in 3D) -------------------------------
static float hash3(float x, float y, float z) {
    float h = std::sin(x * 12.9898f + y * 78.233f + z * 37.719f) * 43758.5453f;
    return h - std::floor(h);
}
static float vnoise(const Vec3f& p) {
    Vec3f i(std::floor(p.x), std::floor(p.y), std::floor(p.z));
    Vec3f f(p.x - i.x, p.y - i.y, p.z - i.z);
    Vec3f u(f.x*f.x*(3-2*f.x), f.y*f.y*(3-2*f.y), f.z*f.z*(3-2*f.z));
    auto L = [](float a, float b, float t){ return a + (b-a)*t; };
    float n000=hash3(i.x,i.y,i.z),       n100=hash3(i.x+1,i.y,i.z);
    float n010=hash3(i.x,i.y+1,i.z),     n110=hash3(i.x+1,i.y+1,i.z);
    float n001=hash3(i.x,i.y,i.z+1),     n101=hash3(i.x+1,i.y,i.z+1);
    float n011=hash3(i.x,i.y+1,i.z+1),   n111=hash3(i.x+1,i.y+1,i.z+1);
    return L(L(L(n000,n100,u.x), L(n010,n110,u.x), u.y),
             L(L(n001,n101,u.x), L(n011,n111,u.x), u.y), u.z);
}

bool SplatModel::init(const std::string& crow_path,
                      const std::string& student_path, int n_points) {
    n_ = std::max(1, n_points);
    splats_.clear();
    splats_.reserve(n_);

    // Seed the field from a crow image: sample bright/edge pixels, map (u,v) to a
    // billboard plane, push depth from luma + noise -> a crow-shaped volume.
    Image im;
    bool img_ok = !crow_path.empty() && im.load(crow_path);
    std::mt19937 rng(0xC0FFEEu);
    std::uniform_real_distribution<float> uni(0.f, 1.f);

    if (img_ok && !im.array().empty()) {
        int W = int(im.width()), H = int(im.height());
        auto& a = im.array();
        int placed = 0, guard = 0;
        while (placed < n_ && guard < n_ * 50) {
            ++guard;
            int x = int(uni(rng) * (W - 1));
            int y = int(uni(rng) * (H - 1));
            size_t idx = (size_t(y) * W + x) * 4;
            float r = a[idx] / 255.f, gg = a[idx+1] / 255.f, b = a[idx+2] / 255.f;
            float luma = 0.299f*r + 0.587f*gg + 0.114f*b;
            // crows are dark: favor dark-but-not-background pixels
            float pick = (luma < 0.55f) ? (0.6f - luma) : 0.05f;
            if (uni(rng) > pick) continue;
            Splat s;
            float u = (float(x)/W - 0.5f) * 2.0f;
            float v = -(float(y)/H - 0.5f) * 2.0f;
            float depth = (luma - 0.3f) * 1.2f + (vnoise(Vec3f(u*3,v*3,0)) - 0.5f);
            s.base  = Vec3f(u, v, depth);
            s.pos   = s.base;
            s.color = Vec3f(r, gg, b) * 1.3f + Vec3f(0.05f);
            s.sigma = 0.35f + 0.5f * (1.f - luma);
            s.alpha = 0.5f + 0.5f * pick;
            splats_.push_back(s);
            ++placed;
        }
    }
    if (splats_.empty()) {  // no image: a procedural corvid-ish blob
        for (int i = 0; i < n_; ++i) {
            Vec3f p(uni(rng)-0.5f, uni(rng)-0.5f, uni(rng)-0.5f);
            p *= 2.f;
            Splat s; s.base = p; s.pos = p;
            s.color = Vec3f(0.15f + 0.2f*uni(rng), 0.18f + 0.2f*uni(rng), 0.25f + 0.3f*uni(rng));
            s.sigma = 0.4f; s.alpha = 0.5f;
            splats_.push_back(s);
        }
    }
    n_ = int(splats_.size());

    have_student_ = loadStudent(student_path);

    mesh_.primitive(Mesh::POINTS);
    rebuildMesh();
    shader_ok_ = shader_.compile(kVert, kFrag);
    return shader_ok_;
}

bool SplatModel::loadStudent(const std::string& path) {
    if (path.empty()) return false;
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return false;
    char magic[4] = {};
    int32_t ver = 0, din = 0, dh = 0, dout = 0;
    bool ok = std::fread(magic, 1, 4, f) == 4
           && std::fread(&ver, 4, 1, f) == 1
           && std::fread(&din, 4, 1, f) == 1
           && std::fread(&dh, 4, 1, f) == 1
           && std::fread(&dout, 4, 1, f) == 1;
    if (!ok || magic[0]!='C'||magic[1]!='R'||magic[2]!='V'||magic[3]!='S'
        || din != 3 + COND || dout != 7 || dh <= 0 || dh > 4096) {
        std::fclose(f); return false;
    }
    sd_in_ = din; sd_hid_ = dh; sd_out_ = dout;
    sd_W1.resize(size_t(dh)*din); sd_b1.resize(dh);
    sd_W2.resize(size_t(dout)*dh); sd_b2.resize(dout);
    ok = std::fread(sd_W1.data(),4,sd_W1.size(),f)==sd_W1.size()
      && std::fread(sd_b1.data(),4,sd_b1.size(),f)==sd_b1.size()
      && std::fread(sd_W2.data(),4,sd_W2.size(),f)==sd_W2.size()
      && std::fread(sd_b2.data(),4,sd_b2.size(),f)==sd_b2.size();
    std::fclose(f);
    if (!ok) { sd_W1.clear(); sd_b1.clear(); sd_W2.clear(); sd_b2.clear(); return false; }
    return true;
}

void SplatModel::studentForward(const float* in, float* out) const {
    // h = tanh(W1*in + b1)
    std::vector<float> h(sd_hid_);
    for (int j = 0; j < sd_hid_; ++j) {
        float acc = sd_b1[j];
        const float* row = &sd_W1[size_t(j) * sd_in_];
        for (int k = 0; k < sd_in_; ++k) acc += row[k] * in[k];
        h[j] = std::tanh(acc);
    }
    for (int o = 0; o < sd_out_; ++o) {
        float acc = sd_b2[o];
        const float* row = &sd_W2[size_t(o) * sd_hid_];
        for (int j = 0; j < sd_hid_; ++j) acc += row[j] * h[j];
        out[o] = acc;
    }
}

void SplatModel::update(const ThoughtVector& tv, float t, float dt) {
    (void)dt;
    // compact conditioning vector from the thought
    float cond[COND] = { tv.action[0], tv.action[1], tv.action[2], tv.action[3],
                         tv.action[4], tv.entropy, tv.confidence,
                         1.f / (1.f + std::exp(-tv.value)) };

    // entropy -> disperse / re-noise ("hallucinating"); confidence -> coalesce.
    float disperse = 0.15f + 1.1f * tv.entropy;
    float cohere   = 0.25f + 0.7f * tv.confidence;

    for (auto& s : splats_) {
        Vec3f target = s.base;
        Vec3f delta(0,0,0);
        if (have_student_) {
            float in[3 + COND];
            in[0]=s.base.x; in[1]=s.base.y; in[2]=s.base.z;
            for (int c = 0; c < COND; ++c) in[3+c] = cond[c];
            float out[7];
            studentForward(in, out);
            delta = Vec3f(out[0], out[1], out[2]);
            s.color = (s.color * 0.96f) + Vec3f(out[3], out[4], out[5]) * 0.04f;
            s.sigma = std::max(0.08f, s.sigma * 0.98f + (0.5f + 0.5f*out[6]) * 0.02f);
        } else {
            // procedural churn: domain-warped noise scaled by entropy
            Vec3f q = s.base * 1.7f + Vec3f(0, 0, t * (0.1f + 0.3f * tv.entropy));
            delta = Vec3f(vnoise(q + Vec3f(11,0,0)) - 0.5f,
                          vnoise(q + Vec3f(0,17,0)) - 0.5f,
                          vnoise(q + Vec3f(0,0,23)) - 0.5f) * disperse;
        }
        // blend toward base (cohere) plus current hallucination delta
        s.pos = s.pos + ((target + delta) - s.pos) * std::min(1.f, cohere * 0.15f + 0.05f);
        s.alpha = 0.35f + 0.5f * tv.confidence;
    }
    rebuildMesh();
}

void SplatModel::rebuildMesh() {
    mesh_.reset();
    mesh_.primitive(Mesh::POINTS);
    for (auto& s : splats_) {
        mesh_.vertex(s.pos.x, s.pos.y, s.pos.z);
        mesh_.color(s.color.x, s.color.y, s.color.z, s.alpha);
        mesh_.texCoord(std::min(1.f, s.sigma), 0.f);
    }
    mesh_.update();
}

void SplatModel::draw(Graphics& g, const Vec3f& center, float worldScale) {
    if (!shader_ok_) return;
    g.depthTesting(true);
    g.blending(true);
    g.blendAdd();                  // additive Gaussian splats

    g.shader(shader_);
    g.shader().uniform("uPointSize", 14.f * worldScale);

    g.pushMatrix();
    g.translate(center.x, center.y, center.z);
    g.scale(worldScale);
    g.pointSize(14.f * worldScale);  // fallback for drivers ignoring gl_PointSize
    g.draw(mesh_);
    g.popMatrix();

    g.blendTrans();
}

} // namespace corvid
