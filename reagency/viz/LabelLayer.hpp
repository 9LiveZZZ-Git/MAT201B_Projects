#pragma once
// LabelLayer — the machine's CLASSIFICATION made visible (à la the Open Images visualizer):
// the concept WORD the machine assigns to each image/cluster, shown as text. Two roles:
//   - drawClusters(): floating word labels at cluster centroids (a standing map of the
//     machine's classifications across the galaxy),
//   - drawSlot(): one label billboarded at a point — used on a surfacing photo so you see the
//     word the machine calls it.
// Pre-rendered offline by stage_b into labels_atlas.png (any script, via PIL) + labels.txt
// (cluster centroids + slots) + classify.txt (image node -> slot). Inert until those exist.
#include "al/graphics/al_Graphics.hpp"
#include "al/graphics/al_Shader.hpp"
#include "al/graphics/al_Texture.hpp"
#include "al/graphics/al_VAOMesh.hpp"
#include "al/math/al_Vec.hpp"
#include <iterator>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>

namespace wosw {

struct LabelLayer {
  bool init(const std::string& assetDir);
  bool ready() const { return shader_ok_ && tex_ok_; }

  void drawClusters(al::Graphics& g, const al::Vec3f& camR, const al::Vec3f& camU, float alpha);
  void drawSlot(al::Graphics& g, const al::Vec3f& pos, const al::Vec3f& camR,
                const al::Vec3f& camU, int slot, float alpha, float h = 0.5f);
  int  slotForNode(int node) const {
    auto it = nodeSlot_.find(node); return it == nodeSlot_.end() ? -1 : it->second;
  }
  // total number of distinct class WORDS available (slots that carry text) — lets the detector pick a
  // deterministically-RANDOM class to stamp, not just the node's own. 0 if none loaded.
  int  slotCount() const { return int(slotWord_.size()); }
  // map a 0..slotCount()-1 ordinal to an actual slot id that has a word (slots may be sparse). The
  // walk is over std::map so it is key-ordered and identical on every dome node.
  int  slotByOrdinal(int i) const {
    if (slotWord_.empty()) return -1;
    const int n = int(slotWord_.size()); i = ((i % n) + n) % n;
    auto it = slotWord_.begin(); std::advance(it, i); return it->first;
  }
  // The classification WORD for a node ("" if unknown) — used by the audio whisper.
  std::string wordForNode(int node) const {
    auto it = nodeSlot_.find(node); if (it == nodeSlot_.end()) return "";
    auto w = slotWord_.find(it->second); return w == slotWord_.end() ? "" : w->second;
  }

 private:
  al::Texture       tex_;
  al::ShaderProgram shader_;
  al::VAOMesh       mesh_;
  bool tex_ok_ = false, shader_ok_ = false;
  int  cols_ = 8, cw_ = 256, ch_ = 64, atlasW_ = 1, atlasH_ = 1;

  struct CL { al::Vec3f pos; int slot; };
  std::vector<CL> clusters_;
  std::unordered_map<int, int> nodeSlot_;
  std::map<int, std::string> slotWord_;   // slot -> word text; std::map so slotByOrdinal's iteration
                                          // order is key-sorted + identical on every dome render node

  bool loadAtlas(const std::string& path);
  void loadText(const std::string& assetDir);
};

}  // namespace wosw
