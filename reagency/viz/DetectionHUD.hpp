#pragma once
// DetectionHUD — feature B: a procedural computer-vision read-out that ALWAYS looks like a blob/
// object detector is running. It annotates the currently-surfaced hero photos and near-camera dream
// billboards with animated LOCK-ON reticles (corner-bracket bounding box + vertical scan line +
// jittering tracked feature dots + a flickering confidence bar), and draws a head-locked SORT STRIP
// of markers that continuously re-order (a visible running sort by cluster/density).
//
// HARD DOME RULE: every coordinate/alpha here is a PURE DETERMINISTIC FUNCTION of synced state —
// the synced double simTime, the synced node index, and the identically-loaded node key
// (cluster/density). No wall clock, no rand(): per-target jitter is hash(nodeIndex, floor(t*rate)).
// So the PRIMARY and every renderer compute the identical HUD. allolib-only: ONE additive-line
// al::ShaderProgram (the same loc0-pos / loc1-vec4-color contract as WebRenderer) feeds one VAOMesh
// rebuilt per frame; the class WORD is drawn by the caller via LabelLayer (no new text path).
#include "al/graphics/al_Graphics.hpp"
#include "al/graphics/al_Shader.hpp"
#include "al/graphics/al_VAOMesh.hpp"
#include "al/math/al_Vec.hpp"
#include <cstdint>
#include <string>

namespace wosw {

struct DetectionHUD {
  bool init();                    // compiles the line shader (no assets needed)
  bool ready() const { return shader_ok_; }

  // ---- world-space lock-on reticle, billboarded at `center` (sits ON the surfaced object) ----
  // node      : the node index being "detected" (drives per-target deterministic phase/jitter)
  // camR/camU : synced camera right/up (billboard plane)
  // age       : seconds the target has been alive (drives the lock-on acquire spring); for a target
  //             with no age clock, pass a synced phase (0 = just appeared, large = settled)
  // conf01    : base confidence (0..1) e.g. from the node density; the bar flickers around it
  // bright    : overall intensity (fades with the act/depth crossfade)
  // The object's world half-size is supplied via the boxHalf_ member, set by the caller immediately
  // before each call, so the signature stays compact and the box snaps to fit each photo/dream.
  void drawReticle(al::Graphics& g, const al::Vec3f& center, const al::Vec3f& camR,
                   const al::Vec3f& camU, int node, double simTime,
                   float age, float conf01, float bright);

  // ---- head-locked SORT STRIP: N markers continuously re-ordering by key ----
  // keys[i]   : the per-marker sort key (e.g. clusterOf or densityOf of a sampled node)
  // hue[i]    : a 0..1 hue (e.g. cluster id normalised) so each marker keeps its colour as it moves
  // n         : marker count (small, fixed cap)
  // The displayed order = a deterministic comparison sort on (keys[i] + slow simTime ripple), so the
  // row is perpetually migrating but identical on every node. Anchored to the synced camera basis.
  void drawSortStrip(al::Graphics& g, const al::Vec3f& camPos, const al::Vec3f& camFwd,
                     const al::Vec3f& camR, const al::Vec3f& camU, const float* keys,
                     const float* hue, int n, double simTime, float bright);

  // tunable lock-on look (one-line dials)
  float bracketLen_ = 0.34f;   // corner bracket arm length as a fraction of the box half-size
  float dotCount_   = 5;       // tracked feature dots per box (cap)
  float boxHalf_    = 1.0f;    // set by the caller (per target) just before drawReticle; the box
                              // half-size in world units. Kept as a member so the shared builder
                              // body can use it without changing the signature.

 private:
  al::ShaderProgram shader_;
  al::VAOMesh       mesh_;     // ONE reused per-frame additive-line mesh (rebuilt, not reallocated)
  bool shader_ok_ = false;

  // shift every vertex of m by d in world space (cheap per-channel offset for the in-layer chromatic
  // split); caller restores by calling again with -d. Applied BEFORE allolib's dome modelview.
  static void pushOffset_(al::VAOMesh& m, const al::Vec3f& d);

  // deterministic hash -> [0,1): integer mix (no float clock, no rand). Identical on every node.
  static float hash01(uint32_t a, uint32_t b) {
    uint32_t h = a * 0x9E3779B1u ^ (b + 0x85EBCA6Bu + (a << 6) + (a >> 2));
    h ^= h >> 15; h *= 0x2C1B3C6Du; h ^= h >> 12; h *= 0x297A2D39u; h ^= h >> 15;
    return float(h & 0x00FFFFFFu) / float(0x01000000u);
  }
  // add one additive line segment to mesh_ (loc0 pos, loc1 rgba)
  void seg(const al::Vec3f& a, const al::Vec3f& b, float r, float gg, float bb,
           float aa0, float aa1);
};

}  // namespace wosw
