# World of Shadow Work — Implementation Plan (v4)

*An autonomous, distributed generative-art installation for the AlloSphere, on the theme of Re:Agency.*
*Project slug: `reagency`. Title: **World of Shadow Work** (Illich's term — the uncompensated
human labor an industrial system silently extracts; here, our images and words are the shadow
work feeding the machine's "magic").*

> **v4 (2026-05-29)** folds in the infra change + reflects what's actually built:
> - **Offline generation consolidated onto Google Colab Pro+ (A100 ~40 GB)** — no separate box;
>   VRAM is no longer a constraint, so we run the best generators at full quality. One-stop
>   notebook (`factory/wosw_colab.ipynb`).
> - **Vessels are now per-IMAGE** (one dream-body per corpus image → hundreds of morph
>   keyframes), not one per cluster (~19). Bank size auto-capped.
> - **Bigger corpus** (more queries; deeper fetch).
> - **Runtime is built through the galaxy + webs + live wander + the vessel module.**
> - **Architecture honesty:** the "one conserved substance" ideal (v2/v3) became, in the build,
>   *galaxy points* + a *separate generative-splat vessel*. The material link survives because
>   **every vessel body is the machine's 3D hallucination of a specific corpus image that is also
>   a galaxy node** — and the human trace is the on-screen bridge. See §1.1.

---

## 0. Headline decisions + build status

| Axis | Decision | Status |
|---|---|---|
| **Mode** | Autonomous "machine dream"; no live input; audience witnesses. | ✅ built |
| **Decision-making** | The machine walks its own k-NN graph **live**; hesitation = entropy of its next-step choice. | ✅ built (`Conductor`) |
| **Embedding space** | Multimodal CLIP — words + images in one space; proximity = similarity. | ✅ built (877 nodes; grows with corpus) |
| **Galaxy** | Navigable point cloud + **k-NN similarity webs** ("what the model thinks belongs together"). | ✅ built (`ParticleField` + `WebRenderer`) |
| **Vessel** | **Proper generative-splat, now per-IMAGE**: image-to-3D (LGM / TRELLIS) of corpus photos → distilled `WSWV` keyframe bank → live morph + smoky melt. | ✅ runtime built (`VesselSplats`); awaiting the Colab asset |
| **Human trace (thesis)** | A fixated image's photo surfaces, then dissolves into grains that flow into the vessel formed from that same image. | ⏳ next (CUDA-free) |
| **Loop** | Galaxy ↔ vessel by depth crossfade (slow bleed). | ⏳ partial (vessel at center; depth-descent next) |
| **Corpus** | Images + words on **agency / magic / labor**, CC0/PD with attribution (AIC + Cleveland). | ✅ built; bigger now |
| **Offline infra** | **Google Colab Pro+ (A100 ~40 GB)**, one-stop notebook. (was: separate CUDA box.) | ✅ built |
| **Runtime constraint** | allolib + al_ext only; custom GLSL via `al::ShaderProgram`; no raw GL, no runtime ML. | ✅ honored |
| **Deploy** | Linux AlloSphere via `al::DistributedAppWithState` + omni. Dev/preview on this Mac. | ⏳ M7 (dome bring-up) |

**Built + pushed (branch `world-of-shadow-work`):** M0 scaffold, M1 webs, M2 Conductor (live
wander + entropy hesitation, calm camera), `VesselSplats` (per-image morphing vessel + procedural
fallback), galaxy/web look tuning, and the full Colab factory (corpus → CLIP → UMAP → vessels →
`WSWV`). Real galaxy assets (points/edges/atlas) are generated; the per-image vessel bank is the
next Colab run.

---

## 1. Concept spine

The corvid failure was *three effects with no statement*. The spine every shell, sound, and camera
move serves:

> **You are inside a system walking its own learned associations, live, with no human at the wheel
> — and watching it consume human culture into its own private order of meaning.**

### 1.1 The galaxy and the vessel — the honest architecture **[v4]**
The build has two materially-linked layers (the v2/v3 "literally the same particles" ideal was
simplified, because the vessel is now *generated* splats, not the galaxy points re-laid-out):

- **Core (`d≈1`) — the GALAXY.** Every point is one corpus item (image or word) at its UMAP
  position; **k-NN webs** connect what CLIP thinks belongs together. This *is* the model's claim
  about what our things mean.
- **Outer (`d≈0`) — the VESSEL.** A constantly-regenerating Gaussian-splat dream-body. Each
  keyframe is the machine's **image-to-3D hallucination of a specific corpus image** — i.e. a
  specific galaxy node. As the machine wanders, the body **melts** from one image's hallucination
  to the next (impossible geometry, multi-view glitch — the SplatFlow/Anadol register).
- **The bridge (so it's one piece, not two layers):** the vessel is *made of the galaxy's own
  images*, and the **human trace** stages the hand-off on screen — a fixated photo surfaces, then
  dissolves into grains that flow into the vessel forming from that same image. *Human meaning
  handed to, and consumed by, the machine.*
- **Middle (`d≈0.5`) — smoke / latent walk (optional).** The galaxy points curl-noise-advected
  during the descent so the galaxy appears to flow into the vessel. Nice-to-have.

### 1.2 Where the agency is (honesty contract) — built
- **What is near what** (layout, webs, clusters) is **CLIP's learned judgment**, baked offline.
- **The wander** is a **live decision on the primary**: from the current node the `Conductor`
  scores neighbors (weight × novelty × seeded noise) and samples the next. Graph traversal of a
  learned similarity structure, happening on stage — not a baked path (suits the endless wander).
- **Hesitation is a real readout:** the normalized entropy of that choice distribution. When
  several neighbors are equally plausible, entropy is high and the machine visibly lingers. ✅ built.

### 1.3 The cede, staged — the human trace (next)
When the wander fixates an image, its **actual photo surfaces** (from `atlas_0.png`), holds, then
**dissolves into grains** as the machine re-files it. That recurring dissolve is the transfer the
theme is about. Promoted to **thesis**; it is also the perceptual bridge from galaxy to vessel.

### 1.4 Endless wander; slow bleed
No finite show — the live graph-walk wanders endlessly (seedable for reproducible capture). The
galaxy↔vessel transition is a slow depth crossfade; the vessel re-forms from the image the machine
is fixating.

---

## 2. References → lessons
| Reference | Borrow |
|---|---|
| **Anadol — *Machine Hallucinations*** | Latent-walk aesthetic + density-colored webs; camera-as-tracing-the-machine's-decisions. |
| **"Gyre 35700" (Mark J. Stock)** | Smoke motion = curl-noise/vortex advection (cheap GLSL), for the optional middle shell. |
| **cutterkom** | Webs as low-alpha additive accumulation (built); seeded reproducibility. |
| **SplatFlow / LGM / TRELLIS** | The vessel: image-to-3D Gaussian splats, melted between keyframes; the glitch is the model's default failure mode (lean in). |
| **TF Projector / Nomic Atlas** | Autonomous "fixation" UX; depth-gated hierarchical labels; density → nebula glow. |

**Anti-Anadol differentiator (load-bearing):** the corpus is explicitly **agency/magic/labor**, the
**human→machine cede is shown** (human trace), and the decision process is **live**. That critical
content is what this has that Anadol's clouds don't.

---

## 3. Two worlds: offline factory (Colab) vs. allolib runtime
```
  COLAB PRO+ (A100 ~40 GB) ──[ git push: small *.bin / *.png / *.wswv ]──▶ ALLOLIB RUNTIME (this Mac / dome)
  one notebook, run once                                                    allolib only; no ML; no raw GL
```
Generation runs on **Colab Pro+ from this machine** — no separate box. Only small data assets cross
the boundary (via `git push` from Colab → `git pull` here / on the dome nodes). VRAM is no longer a
constraint, so the best generators (LGM / TRELLIS) run at full quality. **`factory/wosw_colab.ipynb`**
is the one-stop pipeline: corpus → CLIP → UMAP galaxy + webs → per-image vessels → `WSWV` → push.

---

## 4. Runtime architecture (allolib) — built

`struct WoSW : al::DistributedAppWithState<WoSWState>` (`onCreate / onAnimate[primary] / onDraw[all]
/ onSound[primary]`). **The bet (built):** heavy geometry loads identically per node at `onCreate`;
only ~1 KB of scalars sync per frame; all motion is GLSL/CPU from synced state. Dome-correctness
guards in: primary-only audio, nav-input disabled on renderers, synced-pose applied on every node,
frame counter for in-order drops.

| Module | Job | Status |
|---|---|---|
| `core/WoSWState` | POD sync payload: frame, time, seed, nav pose, depth, curNode/nextNode, hesitation, focusPos, vesselKf. | ✅ |
| `core/Conductor` (primary) | The live mind: walks the k-NN graph, scores neighbors, entropy = hesitation, advances vessel keyframe. | ✅ |
| `viz/ParticleField` (all) | The galaxy: one `Mesh::POINTS` cloud, additive **bloom** point-sprites (core+halo), cluster color + density; loads `points.bin` or a procedural fallback. | ✅ |
| `viz/WebRenderer` (all) | k-NN webs as low-alpha additive lines; loads `edges.bin` (cosine graph) or a 3D-kNN fallback; exposes the adjacency the Conductor walks. | ✅ |
| `viz/VesselSplats` (all) | Loads `vessel.wswv`, CPU index-lerps between keyframes (Morton-ordered offline → real morph), smoky melt (curl-noise + opacity dip + size swell), additive bloom sprites; dependency-free fp16; procedural fallback. | ✅ |
| Camera (primary; synced) | **Calm slow orbit** (the fixation-following camera read as distracting and was removed). | ✅ |
| `viz/HumanTrace` (all) | Fixated photo surfaces + dissolves into grains. | ⏳ next |
| Depth-descent crossfade | Galaxy ↔ vessel by `depth`; optional smoke shell. | ⏳ next |
| `audio/AudioEngine` (primary) | Musical multisample + pulsaret, spatialized. | ⏳ M5 |
| `viz/LabelLayer` (all) | Depth-gated hierarchical cluster labels. | ⏳ M6 |

Bloom is a per-sprite core+halo in the fragment shader (no FBO/post-process → composes with
omni/dome warp). Point-size is uniform-driven; tuned for dome brightness.

---

## 5. The offline factory (`factory/`, Colab Pro+)

One notebook (`wosw_colab.ipynb`, CONFIG cell → Run All) runs:
- **fetch_corpus.py** — CC0/PD images (AIC + Cleveland) on agency/magic/labor; bigger now (more
  query terms, `--per-query 20`) + `ATTRIBUTION.csv`. (`download_from_csv.py` re-fetches from the
  ledger on a fresh Colab clone.)
- **Stage A — `stage_a_embed.py`** — open_clip **ViT-L/14** image+text embedding (A100), L2-normalized.
- **Stage B — `stage_b_layout.py`** — UMAP→3D `coords`, FAISS cosine **k-NN webs** (`edges`), HDBSCAN/
  KMeans clusters, density, colors, **hero atlas** (`atlas_0.png`), `labels.txt`, `manifest.json`,
  and **`cluster_reps.json`**. → `points.bin` + `edges.bin`.
- **Stage C — anchors only** — the walk is **live** at runtime; nothing baked here beyond the seed.
- **Stage D — vessels (per-image) [v4]** — `prep_vessel_inputs.py --mode all` collects **every corpus
  image** (→ one vessel each; `--mode reps`/`--limit` to scope). **LGM** (simple/fast, trivial on the
  A100) or **TRELLIS** (top quality, now fits) → `.ply` per image. `stage_d_vessel.py` inverts the
  3DGS activations, prunes to a fixed `G`, **global-AABB normalizes**, imposes a **Morton canonical
  order** (so index-lerp is a real morph, not swimming), packs **fp16** → `vessel.wswv`. A **`--max-mb`
  cap auto-reduces `G`** so hundreds of keyframes stay ≤ ~60 MB.
- **Stage E — audio bank** — multisamples (CC0/recorded). ⏳ M5.

---

## 6. Data contracts
Little-endian, 4-byte magic + int32 header + payload.
- **`points.bin`** `WSWP`: N×10 f32 — `xyz, rgb, density, cluster, type(0=img/1=word), atlas_idx`.
- **`edges.bin`** `WSWE`: E×(u32 i, u32 j, f32 w) — cosine k-NN (the webs **and** the Conductor's graph).
- **`vessel.wswv`** `WSWV` **[v4]**: header `[magic, ver, K, G, stride=8] + 6×f32 AABB`, then
  `K×G×8 fp16` — `pos3, rgb3, opacity1, sigma1` (quat dropped; isotropic sprites). K = #vessel images
  (hundreds); `G` auto-sized so total ≤ ~60 MB.
- **`atlas_0.png`** (hero thumbnails, for the human trace), **`labels.txt`**, **`manifest.json`**,
  **`cluster_reps.json`**.
- Assets are git-tracked (a few MB + the atlas) so they reach the Mac + every dome node via `git pull`.

---

## 7. Audio (M5, todo)
Lock-free voice pool (salvaged from `pulsar_cern_v2`); pooled `gam::SamplePlayer` multisamples +
pulsaret layer; scale-quantized to a root+mode from `manifest.json`; mapping: fixation→onset,
web-ignition→arpeggio over k-NN, density→voicing, hesitation→tremolo, depth→timbre morph; `al::Lbap`
spatialization over the **54-speaker** dome, `al::Reverb`; primary-only, events via a lock-free ring.

## 8. Distribution & determinism — built guards
Primary simulates + broadcasts `WoSWState`; renderers apply the synced pose and draw. Omni is
automatic (shaders use `al_ModelViewMatrix`/`al_ProjectionMatrix`). Mandatory guards (in): primary-
only audio, nav-input off on renderers, pose applied every node, frame counter. **M2.5** two-process
desktop sync check before the dome.

---

## 9. Budget
Galaxy 1k–100k `GL_POINTS` at 60 fps is comfortable; built once. Vessel: only **two** keyframes alive
at a time (CPU-lerped into one ≤~12k-point mesh) — fill-rate-safe; bank ≤ ~60 MB regardless of K.
Assets a few MB + atlas + audio ≪ any node's RAM. No runtime ML / external libs / raw GL.

---

## 10. Salvage (from corvid/pulsar)
Distributed-state pattern, lock-free audio pool, Gaussian point-sprite shader + `.bin` loader idiom,
3D value-noise (→ curl), spatialization/reverb/streamed-audio, the build pattern. *(SpatialHash not
used for the galaxy — the precomputed `edges.bin` adjacency is the graph.)*

---

## 11. Project layout
```
MAT201B_Projects/reagency/
  PLAN.md  CMakeLists.txt  run_demo.sh  .gitignore
  src/main_reagency.cpp           src/flags.cmake   (run.sh single-file hook)
  core/  WoSWState.hpp  Conductor.{hpp,cpp}
  viz/   ParticleField.{hpp,cpp}  WebRenderer.{hpp,cpp}  VesselSplats.{hpp,cpp}
         (HumanTrace, LabelLayer, Skybox — todo)
  audio/ (AudioEngine — todo)
  assets/ points.bin edges.bin labels.txt manifest.json atlas_0.png vessel.wswv  (git-tracked)
  factory/ wosw_colab.ipynb  fetch_corpus.py download_from_csv.py stage_a_embed.py
           stage_b_layout.py prep_vessel_inputs.py stage_d_vessel.py
           VESSEL.md README.md requirements.txt make_assets.sh  corpus/ (images gitignored)
```
Builds two ways: `./run.sh MAT201B_Projects/reagency/src/main_reagency.cpp` (single-file via
`flags.cmake`) or `bash reagency/run_demo.sh` (multi-file CMake). Procedural fallbacks so it always runs.

---

## 12. Status & remaining work
Quality over date (no hard deadline). Ordering is a build sequence, not a triage list.

**Done:** M0 scaffold ✅ · M1 webs ✅ · M2 Conductor (live wander + entropy hesitation, calm camera) ✅ ·
`VesselSplats` runtime ✅ · Colab factory + per-image vessels + bigger corpus ✅ · galaxy/web look tuned ✅.

**Next (runtime, CUDA-free, on this Mac):**
1. **Human trace** — fixated photo surfaces from `atlas_0.png`, dissolves into grains (the thesis + the galaxy→vessel bridge).
2. **Depth-descent crossfade** — galaxy ↔ vessel by `depth` (+ optional curl-noise smoke shell).
3. **Audio (M5)** — voices + mapping + 54-ch spatialization.
4. **Labels (M6)** — depth-gated cluster labels.

**Asset (Colab):** run `wosw_colab.ipynb` → real galaxy (bigger) + per-image `vessel.wswv`; `git pull` here.

**Dome (M7):** two-process sync check → AlloSphere bring-up (omni, multi-node, 54-speaker). Gradeable
artifact = desktop capture, so dome access can't sink it.

**Concept gate (still applies):** show a build with no wall text — can a viewer read that the points
are human-made images/words and that a system is consuming/reordering them on its own? If not, the
concept isn't done regardless of engine quality.

---

## 13. Top risks
1. **Concept slips to pretty demo** → human trace + corpus argument are required; concept gate above.
2. **All-images vessels = long Colab run** (~15 s/image → hundreds = hours). → smoke-test with
   `VESSEL_LIMIT`, scale up; `--max-mb` keeps the bank small regardless.
3. **Vessel "swimming"** between unrelated shapes → offline Morton canonical order + the smoky melt
   hide it; GaussianCube OT is the upgrade if needed.
4. **Dome desync** → §8 guards mandatory; M2.5 + M7 verify.
5. **fp16 position cracks** → positions normalized to ~[-1,1] (global AABB) before fp16; safe.
6. **Galaxy/vessel both additive + busy** → vessel dimmed/small at center until the depth-descent
   gives it its own moment; tunable.

---

## 14. Resolved decisions
Autonomous; multimodal CLIP; live graph-walk; **per-image** generative-splat vessel; human trace
(required, next); slow-bleed depth loop; corpus = agency/magic/labor (CC0, bigger); title *World of
Shadow Work*; endless wander; offline = **Colab Pro+ A100**. Remaining polish: exact corpus/vessel
counts, audio sample source, label styling.

---

## 15. Revision log
- **v4 (2026-05-29):** Offline generation consolidated onto **Google Colab Pro+ (A100 ~40 GB)** via
  one-stop notebook (no separate box; VRAM unconstrained → LGM/TRELLIS at full quality). **Vessels
  now per-IMAGE** (hundreds of keyframes, size-capped) instead of per-cluster. Bigger corpus.
  Reflected the **built** state (galaxy, webs, Conductor, VesselSplats, calm camera, look tuning) and
  corrected the architecture to *galaxy points + separate generative-splat vessel*, bridged by the
  human trace and the fact that each vessel is the machine's hallucination of a galaxy image.
- **v3 (2026-05-29):** Deadline relaxed (quality over date); proper generative-splat vessel promoted
  to REQUIRED (then per-cluster); full scope restored.
- **v2 (2026-05-29):** Author's 5 decisions + 6-dimension adversarial red-team; conserved-particle
  architecture; live walk; human trace as thesis; dome-correctness fixes; corpus fetched.
- **v1:** initial plan from the 6-way research fan-out.
```
