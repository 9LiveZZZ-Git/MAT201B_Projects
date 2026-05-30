#include "viz/LabelLayer.hpp"
#include "al/graphics/al_Image.hpp"
#include <cstdio>

namespace wosw {

using namespace al;

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

// label atlas is white text on transparent; alpha = glyph coverage
static const char* kFrag = R"GLSL(
#version 330
uniform sampler2D tex;
uniform float uAlpha;
uniform vec4  uUV;
in vec2 vuv;
out vec4 fragColor;
void main() {
  vec2 t = mix(uUV.xy, uUV.zw, vuv);
  vec4 c = texture(tex, t);
  fragColor = vec4(c.rgb, c.a * uAlpha);
}
)GLSL";

bool LabelLayer::loadAtlas(const std::string& path) {
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

void LabelLayer::loadText(const std::string& assetDir) {
  const std::string bases[] = { assetDir, "assets", "../../assets",
                                "reagency/assets", "MAT201B_Projects/reagency/assets" };
  for (const auto& b : bases) {
    FILE* f = std::fopen((b + "/labels.txt").c_str(), "r");
    if (!f) continue;
    char line[1024]; float x, y, z; int slot, off = 0;
    while (std::fgets(line, sizeof(line), f))
      if (std::sscanf(line, "%f %f %f %d %n", &x, &y, &z, &slot, &off) >= 4) {
        clusters_.push_back({Vec3f(x, y, z), slot});
        std::string w(line + off);                       // the rest of the line is the word
        while (!w.empty() && (w.back() == '\n' || w.back() == '\r' || w.back() == ' ')) w.pop_back();
        if (!w.empty()) slotWord_[slot] = w;
      }
    std::fclose(f);
    if (FILE* c = std::fopen((b + "/classify.txt").c_str(), "r")) {
      int node, sl;
      while (std::fscanf(c, "%d %d", &node, &sl) == 2) nodeSlot_[node] = sl;
      std::fclose(c);
    }
    // slot -> word for EVERY slot (image labels too, not just clusters), if present
    if (FILE* w = std::fopen((b + "/label_words.txt").c_str(), "r")) {
      char line2[1024]; int sl, o2 = 0;
      while (std::fgets(line2, sizeof(line2), w))
        if (std::sscanf(line2, "%d %n", &sl, &o2) >= 1) {
          std::string s(line2 + o2);
          while (!s.empty() && (s.back() == '\n' || s.back() == '\r' || s.back() == ' ')) s.pop_back();
          if (!s.empty()) slotWord_[sl] = s;
        }
      std::fclose(w);
    }
    break;
  }
}

bool LabelLayer::init(const std::string& assetDir) {
  const std::string candidates[] = {
    assetDir + "/labels_atlas.png", "assets/labels_atlas.png", "../../assets/labels_atlas.png",
    "reagency/assets/labels_atlas.png", "MAT201B_Projects/reagency/assets/labels_atlas.png",
  };
  tex_ok_ = false;
  for (const auto& c : candidates)
    if (loadAtlas(c)) { tex_ok_ = true;
      std::fprintf(stderr, "[wosw] label atlas loaded from %s\n", c.c_str()); break; }
  if (tex_ok_) loadText(assetDir);
  else std::fprintf(stderr, "[wosw] labels_atlas.png not found — classification labels off until regen\n");
  shader_ok_ = shader_.compile(kVert, kFrag);
  return ready();
}

// one label cell, billboarded at `c` (cells are ~4:1, so width = 4*height)
static void drawCell(Graphics& g, ShaderProgram& sh, Texture& tex, VAOMesh& mesh,
                     const Vec3f& c, const Vec3f& r, const Vec3f& u, int slot, float alpha,
                     float h, int cols, int cw, int ch, int aw, int ah) {
  if (slot < 0) return;
  const Vec3f rx = r * (h * 2.0f), uy = u * (h * 0.5f);
  const Vec3f BL = c - rx - uy, BR = c + rx - uy, TL = c - rx + uy, TR = c + rx + uy;
  mesh.reset();
  mesh.primitive(Mesh::TRIANGLE_STRIP);
  mesh.vertex(BL.x, BL.y, BL.z); mesh.texCoord(0.f, 1.f);   // v flipped (text upright)
  mesh.vertex(BR.x, BR.y, BR.z); mesh.texCoord(1.f, 1.f);
  mesh.vertex(TL.x, TL.y, TL.z); mesh.texCoord(0.f, 0.f);
  mesh.vertex(TR.x, TR.y, TR.z); mesh.texCoord(1.f, 0.f);
  mesh.update();
  const int col = slot % cols, row = slot / cols;
  const float u0 = float(col * cw) / aw, u1 = float((col + 1) * cw) / aw;
  const float v0 = 1.f - float((row + 1) * ch) / ah, v1 = 1.f - float(row * ch) / ah;
  g.shader(sh);
  tex.bind(0);
  g.shader().uniform("tex", 0);
  g.shader().uniform("uAlpha", alpha);
  g.shader().uniform("uUV", Vec4f(u0, v0, u1, v1));
  g.draw(mesh);
  tex.unbind(0);
}

void LabelLayer::drawClusters(Graphics& g, const Vec3f& camR, const Vec3f& camU, float alpha) {
  if (!ready() || alpha <= 0.f) return;
  g.depthTesting(false); g.blending(true); g.blendTrans();
  for (const auto& c : clusters_)
    drawCell(g, shader_, tex_, mesh_, c.pos, camR, camU, c.slot, alpha, 0.13f,   // smaller still
             cols_, cw_, ch_, atlasW_, atlasH_);
  g.depthTesting(true);
}

void LabelLayer::drawSlot(Graphics& g, const Vec3f& pos, const Vec3f& camR,
                          const Vec3f& camU, int slot, float alpha, float h) {
  if (!ready() || slot < 0 || alpha <= 0.f) return;
  g.depthTesting(false); g.blending(true); g.blendTrans();
  drawCell(g, shader_, tex_, mesh_, pos, camR, camU, slot, alpha, h,
           cols_, cw_, ch_, atlasW_, atlasH_);
  g.depthTesting(true);
}

}  // namespace wosw
