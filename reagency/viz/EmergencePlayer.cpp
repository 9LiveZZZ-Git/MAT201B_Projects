#include "viz/EmergencePlayer.hpp"
#include "al/graphics/al_Image.hpp"
#include "al/graphics/al_OpenGL.hpp"
#include <cstdio>

namespace wosw {

using namespace al;

static constexpr float WHALF = 0.42f;   // tracks the ~3/4 dream size (0.60), keeps the forming latent ~0.7x the resolved

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

// the forming latent, softened to a disc; a faint cool cast early in formation that warms as it
// resolves (uPhase) — so the "thinking" reads as cooling out of noise into the image.
static const char* kFrag = R"GLSL(
#version 330
uniform sampler2D tex;
uniform float uAlpha;
uniform float uPhase;   // 0 forming .. 1 formed
uniform vec4  uUV;
in vec2 vuv;
out vec4 fragColor;
void main() {
  vec2 t = mix(uUV.xy, uUV.zw, vuv);
  vec3 c = texture(tex, t).rgb;
  vec2 d = vuv - 0.5;
  float vig = smoothstep(0.5, 0.32, length(d));
  vec3 cool = vec3(0.86, 0.92, 1.06);
  vec3 tint = mix(cool, vec3(1.0), clamp(uPhase, 0.0, 1.0));   // cool while forming -> neutral when formed
  float a = uAlpha * vig;
  if (a <= 0.004) discard;
  fragColor = vec4(c * tint, a);
}
)GLSL";

bool EmergencePlayer::loadAtlas(const std::string& path) {
  Image im;
  if (!im.load(path)) return false;
  auto& a = im.array();
  if (a.empty()) return false;
  atlasW_ = int(im.width()); atlasH_ = int(im.height());
  GLint maxTex = 0; glGetIntegerv(GL_MAX_TEXTURE_SIZE, &maxTex);   // allolib doesn't check; a too-big atlas goes silently black
  if (maxTex > 0 && (atlasW_ > maxTex || atlasH_ > maxTex))
    std::fprintf(stderr, "[wosw] WARNING emergence_atlas %dx%d EXCEEDS GL_MAX_TEXTURE_SIZE %d on this node — will be blank (raise COLS in bake_emergence.py)\n",
                 atlasW_, atlasH_, int(maxTex));
  tex_.create2D(atlasW_, atlasH_, Texture::RGBA8, Texture::RGBA, Texture::UBYTE);
  tex_.submit(&a[0], Texture::RGBA, Texture::UBYTE);
  tex_.filter(Texture::LINEAR);
  tex_.wrap(Texture::CLAMP_TO_EDGE, Texture::CLAMP_TO_EDGE, Texture::CLAMP_TO_EDGE);
  return true;
}

void EmergencePlayer::loadIndex(const std::string& assetDir) {
  const std::string bases[] = { assetDir + "/../v2", "../../v2", "reagency/v2",
                                "MAT201B_Projects/reagency/v2", assetDir };
  for (const auto& b : bases) {
    FILE* f = std::fopen((b + "/emergence_index.txt").c_str(), "r");
    if (!f) continue;
    char line[256];
    int node, start, num;
    while (std::fgets(line, sizeof(line), f)) {
      if (line[0] == '#') continue;
      if (std::sscanf(line, "%d %d %d", &node, &start, &num) == 3)
        seq_[node] = {start, num};
    }
    std::fclose(f);
    break;
  }
}

bool EmergencePlayer::init(const std::string& assetDir) {
  const std::string candidates[] = {
    assetDir + "/../v2/emergence_atlas.png", "../../v2/emergence_atlas.png",
    "reagency/v2/emergence_atlas.png", "MAT201B_Projects/reagency/v2/emergence_atlas.png",
    assetDir + "/emergence_atlas.png",
  };
  tex_ok_ = false;
  for (const auto& c : candidates)
    if (loadAtlas(c)) { tex_ok_ = true;
      std::fprintf(stderr, "[wosw] emergence atlas loaded from %s\n", c.c_str()); break; }
  if (tex_ok_) { cols_ = atlasW_ / cell_; if (cols_ < 1) cols_ = 1; loadIndex(assetDir); }
  else std::fprintf(stderr, "[wosw] emergence_atlas.png not found — 'watch it think' off (bake v2/bake_emergence.py)\n");
  shader_ok_ = shader_.compile(kVert, kFrag);
  std::fprintf(stderr, "[wosw] emergence sequences: %zu\n", seq_.size());
  return ready();
}

void EmergencePlayer::draw(Graphics& g, const Vec3f& pos, const Vec3f& camRight,
                           const Vec3f& camUp, int node, float phase, float alpha) {
  if (!ready() || alpha <= 0.004f) return;
  auto it = seq_.find(node);
  if (it == seq_.end() || it->second.num <= 0) return;
  const int num = it->second.num;
  int frame = int(phase * float(num));
  if (frame < 0) frame = 0;
  if (frame >= num) frame = num - 1;
  const int cellIdx = it->second.start + frame;
  const int col = cellIdx % cols_, row = cellIdx / cols_;

  g.depthTesting(false); g.blending(true); g.blendTrans();
  g.shader(shader_);
  tex_.bind(0);
  g.shader().uniform("tex", 0);
  const Vec3f rx = camRight * WHALF, uy = camUp * WHALF;
  const Vec3f BL = pos - rx - uy, BR = pos + rx - uy, TL = pos - rx + uy, TR = pos + rx + uy;
  mesh_.reset(); mesh_.primitive(Mesh::TRIANGLE_STRIP);
  mesh_.vertex(BL.x, BL.y, BL.z); mesh_.texCoord(0.f, 1.f);
  mesh_.vertex(BR.x, BR.y, BR.z); mesh_.texCoord(1.f, 1.f);
  mesh_.vertex(TL.x, TL.y, TL.z); mesh_.texCoord(0.f, 0.f);
  mesh_.vertex(TR.x, TR.y, TR.z); mesh_.texCoord(1.f, 0.f);
  mesh_.update();
  const float u0 = float(col * cell_) / atlasW_, u1 = float((col + 1) * cell_) / atlasW_;
  const float v0 = 1.f - float((row + 1) * cell_) / atlasH_;
  const float v1 = 1.f - float(row * cell_) / atlasH_;
  g.shader().uniform("uUV", Vec4f(u0, v0, u1, v1));
  g.shader().uniform("uAlpha", alpha);
  g.shader().uniform("uPhase", phase);
  g.draw(mesh_);
  tex_.unbind(0);
  g.depthTesting(true);
}

}  // namespace wosw
